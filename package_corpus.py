#!/usr/bin/env python3
"""
package_corpus.py — Pseudo-label AfriBase-Speech segments and package as HF dataset.

Reads segment JSONL(s) from segment_lessons.py (or quality_filter.py).
For records where text == null, transcribes using Whisper large-v3 to generate
pseudo-labels. Saves a HuggingFace DatasetDict (train/validation split) as
Arrow/Parquet, ready for finetune_whisper.py.

Usage:
    python package_corpus.py --lang lin --segments segments/lin.jsonl \\
        --audio-root segments/ --out corpus/ --device cuda
    python package_corpus.py --segments-dir segments/ --audio-root segments/ \\
        --out corpus/ --device cuda

Outputs per language:
    corpus/{lang}/
        dataset/            <- HuggingFace DatasetDict (train + validation)
        pseudo_labels.jsonl <- updated JSONL with text filled in
        stats.json
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Whisper language codes for the languages this corpus covers. Pseudo-labeling
# MUST force the target language rather than relying on Whisper's
# auto-detection: auto-detect badly misidentifies lower-resource / script-
# distinct languages (confirmed: 98.5% of auto-detected "amh" pseudo-labels
# were actually transcribed in Arabic script, not Ge'ez -- this silently
# poisoned downstream fine-tuning with wrong-language training labels).
WHISPER_LANG_HINT: dict[str, str] = {
    "amh": "am",
    "hau": "ha",
    "lin": "ln",
    "swh": "sw",
    "yor": "yo",
    "som": "so",
    "ibo": "ig",
    "wol": "wo",
    "orm": "om",
    "sna": "sn",
    "hat": "ht",
    "afr": "af",
}


def load_ckpt(path: Path | None) -> dict[str, str]:
    """utterance_id -> text from the per-batch pseudo-label checkpoint JSONL."""
    done: dict[str, str] = {}
    if path is None or not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
            done[rec["utterance_id"]] = rec["text"]
        except (json.JSONDecodeError, KeyError):
            continue  # torn last line from a crash; segment is redone
    return done


def reuse_labels(segments: list[dict], old_jsonl: Path) -> tuple[list[dict], int]:
    """Copy text from an earlier pseudo_labels.jsonl for segments that cover the same audio:
    same source_file and identical span (3-decimal seconds, as written by segment_lessons.py).
    utterance_id is NOT required to match, because ids are a running counter and shift if
    an earlier source file gains segments."""
    if not old_jsonl.exists():
        return segments, 0
    def key(r: dict) -> tuple:
        return (r.get("source_file"), round(r["span_start_s"], 3), round(r["span_end_s"], 3))
    old: dict[tuple, dict] = {}
    for line in old_jsonl.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            old[key(r)] = r
    out, n = [], 0
    for rec in segments:
        o = old.get(key(rec))
        if not rec.get("text") and o and o.get("text"):
            rec = dict(rec)
            rec["text"] = o["text"]
            rec["text_source"] = o.get("text_source", "whisper-large-v3-pseudo")
            n += 1
        out.append(rec)
    return out, n


def _apply_text_map(segments: list[dict], text_map: dict[str, str]) -> list[dict]:
    updated = []
    for rec in segments:
        if not rec.get("text") and rec["utterance_id"] in text_map:
            rec = dict(rec)
            rec["text"] = text_map[rec["utterance_id"]]
            rec["text_source"] = "whisper-large-v3-pseudo"
        updated.append(rec)
    return updated


def pseudo_label_batch(
    segments: list[dict], audio_root: Path, device: str, batch_size: int = 16,
    lang: str | None = None, ckpt_path: Path | None = None,
) -> list[dict]:
    """Fill text=None segments using Whisper large-v3.

    Checkpointing: each finished batch is appended to ckpt_path (flush + fsync);
    on restart, utterance_ids already in it are not re-transcribed.
    """
    import numpy as np
    import torch
    from transformers import pipeline as hf_pipeline
    import librosa

    text_map: dict[str, str] = load_ckpt(ckpt_path)
    unlabeled = [r for r in segments if not r.get("text") and r["utterance_id"] not in text_map]
    if text_map:
        print(f"  Resuming: {len(text_map)} segments already in {ckpt_path}")
    if not unlabeled:
        print(f"  Nothing left to pseudo-label")
        return _apply_text_map(segments, text_map)

    print(f"  Pseudo-labeling {len(unlabeled)} segments with Whisper large-v3 ...")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    lang_hint = WHISPER_LANG_HINT.get(lang) if lang else None
    if lang_hint is None:
        print(f"  WARNING: no Whisper language hint for '{lang}' -- falling back to "
              f"auto-detect, which is known to misidentify some languages.", file=sys.stderr)
    generate_kwargs = {"num_beams": 1, "no_repeat_ngram_size": 3}
    if lang_hint:
        generate_kwargs["language"] = lang_hint
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        device=device,
        torch_dtype=dtype,
        generate_kwargs=generate_kwargs,
    )

    ck = None
    if ckpt_path is not None:
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        ck = open(ckpt_path, "a", encoding="utf-8")
        if ckpt_path.stat().st_size and ckpt_path.read_bytes()[-1:] != b"\n":
            ck.write("\n")  # terminate a torn last line so new records parse
    for i in range(0, len(unlabeled), batch_size):
        batch = unlabeled[i : i + batch_size]
        audios = []
        for rec in batch:
            ap = Path(rec["audio_path"])
            if not ap.is_absolute():
                ap = audio_root / ap
            try:
                wav, _ = librosa.load(str(ap), sr=16_000, mono=True)
                audios.append(wav.astype(np.float32))
            except Exception as exc:
                print(f"    Warning: load failed {ap.name}: {exc}", file=sys.stderr)
                audios.append(np.zeros(1600, dtype=np.float32))

        results = pipe(audios, batch_size=len(audios))
        for rec, result in zip(batch, results):
            text_map[rec["utterance_id"]] = result["text"].strip()
            if ck is not None:
                ck.write(json.dumps({"utterance_id": rec["utterance_id"],
                                     "text": text_map[rec["utterance_id"]]},
                                    ensure_ascii=False) + "\n")
        if ck is not None:
            ck.flush()
            os.fsync(ck.fileno())

        done = min(i + batch_size, len(unlabeled))
        if done % (batch_size * 5) == 0 or done == len(unlabeled):
            print(f"    {done}/{len(unlabeled)} labeled")

    if ck is not None:
        ck.close()
    return _apply_text_map(segments, text_map)


def build_hf_dataset(segments: list[dict], audio_root: Path, val_frac: float = 0.05):
    from datasets import Audio, Dataset, DatasetDict

    valid = [r for r in segments if r.get("text") and r.get("audio_path")]
    if not valid:
        raise ValueError("No labeled segments to package")

    def gen():
        for rec in valid:
            ap = Path(rec["audio_path"])
            if not ap.is_absolute():
                ap = audio_root / ap
            if not ap.exists():
                continue
            yield {
                "audio": str(ap),
                "text": rec["text"],
                "language": rec.get("language", ""),
                "source_type": rec.get("source_type", ""),
                "duration_s": float(rec.get("duration_s", 0.0)),
                "snr_db": float(rec.get("snr_db", 0.0)),
                "utterance_id": rec.get("utterance_id", ""),
                "license": rec.get("license", "PD-US"),
            }

    ds = Dataset.from_generator(gen).cast_column("audio", Audio(sampling_rate=16_000))
    n_val = max(1, int(len(ds) * val_frac))
    splits = ds.train_test_split(test_size=n_val, seed=42)
    return DatasetDict({"train": splits["train"], "validation": splits["test"]})


def process_lang(
    jsonl_path: Path,
    audio_root: Path,
    out_root: Path,
    device: str,
    val_frac: float,
    batch_size: int,
    no_pseudo_label: bool,
    reuse_root: Path | None = None,
    resume: bool = False,
) -> dict:
    lang = jsonl_path.stem
    out_dir = out_root / lang
    if resume and (out_dir / "stats.json").exists() and (out_dir / "dataset").exists():
        stats = json.loads((out_dir / "stats.json").read_text(encoding="utf-8"))
        if stats.get("dataset_built"):
            print(f"[{lang}] already built ({out_dir}/stats.json), skipping")
            return stats
    print(f"\n[{lang}] Loading {jsonl_path} ...")

    segments = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                segments.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    print(f"  {len(segments)} segments")

    if reuse_root is not None:
        segments, n_reused = reuse_labels(segments, reuse_root / lang / "pseudo_labels.jsonl")
        print(f"  Reused {n_reused}/{len(segments)} labels from {reuse_root}/{lang}/pseudo_labels.jsonl")

    if not no_pseudo_label:
        segments = pseudo_label_batch(segments, audio_root, device, batch_size, lang=lang,
                                      ckpt_path=out_dir / "pseudo_labels.partial.jsonl")

    out_dir.mkdir(parents=True, exist_ok=True)

    pseudo_path = out_dir / "pseudo_labels.jsonl"
    with open(pseudo_path, "w", encoding="utf-8") as f:
        for rec in segments:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    labeled = sum(1 for r in segments if r.get("text"))
    total_dur_h = round(sum(r.get("duration_s", 0) for r in segments) / 3600, 2)
    stats: dict = {
        "language": lang,
        "total_segments": len(segments),
        "labeled_segments": labeled,
        "total_duration_h": total_dur_h,
    }

    if labeled == 0:
        print(f"  No labeled segments — skipping HF dataset build (re-run with GPU to pseudo-label)")
        stats["dataset_built"] = False
    else:
        print(f"  Building HuggingFace DatasetDict ({labeled} labeled) ...")
        ds = build_hf_dataset(segments, audio_root, val_frac)
        ds_path = out_dir / "dataset"
        tmp_path = out_dir / "dataset.tmp"
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        ds.save_to_disk(str(tmp_path))
        if ds_path.exists():
            shutil.rmtree(ds_path)
        tmp_path.rename(ds_path)
        stats["train_size"] = len(ds["train"])
        stats["validation_size"] = len(ds["validation"])
        stats["dataset_built"] = True
        print(
            f"  Saved: {ds_path}  "
            f"({stats['train_size']} train / {stats['validation_size']} val / "
            f"{total_dur_h}h)"
        )

    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Package AfriBase-Speech corpus + pseudo-label")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--segments", nargs="+", help="JSONL file(s) from segment_lessons.py")
    group.add_argument("--segments-dir", help="Directory of *.jsonl files (one per language)")
    ap.add_argument("--lang", nargs="+", help="Filter to these codes when using --segments-dir")
    ap.add_argument("--audio-root", default="segments",
                    help="Root for resolving relative audio_path values (default: segments/)")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--reuse-labels-from", default=None,
                    help="Earlier corpus root (e.g. corpus_v2/): copy text for segments whose "
                         "utterance_id, source_file and span match exactly; only the rest is "
                         "pseudo-labeled")
    ap.add_argument("--resume", action="store_true",
                    help="Skip languages whose dataset is already built (stats.json)")
    ap.add_argument("--no-pseudo-label", action="store_true",
                    help="Skip Whisper pseudo-labeling (keep text=null)")
    args = ap.parse_args()

    if args.segments_dir:
        jsonls = sorted(Path(args.segments_dir).glob("*.jsonl"))
        if args.lang:
            jsonls = [j for j in jsonls if j.stem in args.lang]
        jsonl_paths = jsonls
    else:
        jsonl_paths = [Path(p) for p in args.segments]

    audio_root = Path(args.audio_root)
    out_root = Path(args.out)
    all_stats = {}

    for jsonl_path in jsonl_paths:
        if not jsonl_path.exists():
            print(f"Not found: {jsonl_path}", file=sys.stderr)
            continue
        stats = process_lang(
            jsonl_path, audio_root, out_root, args.device,
            args.val_frac, args.batch_size, args.no_pseudo_label,
            Path(args.reuse_labels_from) if args.reuse_labels_from else None, args.resume,
        )
        all_stats[jsonl_path.stem] = stats

    summary_path = out_root / "package_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
