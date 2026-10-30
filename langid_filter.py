#!/usr/bin/env python3
"""
langid_filter.py — Drop English instructor speech from corpus_v2 segments.

corpus_v2 was segmented with --skip-langid, so it contains English speech. This
runs Whisper large-v3 language identification (faster-whisper, CUDA) on every
segment of train+validation and writes corpus_v3 without the segments whose top
detected language is English. Pseudo-labels and splits are untouched.

Checkpointing: one JSONL line per segment is appended to
results/langid_v3/<lang>.jsonl (flush + fsync every --batch-size segments);
on restart, utterance_ids already present are skipped. The corpus_v3 dataset
is written to a temp dir and renamed, and langid_stats.json is written last, so
a finished language is skipped on rerun.

Usage:
    python langid_filter.py --lang amh hau lin swh yor
    python langid_filter.py --lang lin --max-segments 30 \\
        --results-dir scratch/langid --out-root scratch/corpus_v3   # smoke test
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

TARGET_CODE = {"amh": "am", "hau": "ha", "lin": "ln", "swh": "sw", "yor": "yo"}


def load_done(jsonl: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not jsonl.exists():
        return done
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line from a crash; segment is redone
            done[rec["utterance_id"]] = rec
    return done


def decode_audio(audio_field: dict) -> np.ndarray:
    import librosa
    import soundfile as sf

    arr, sr = sf.read(io.BytesIO(audio_field["bytes"]), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16_000:
        arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
    return arr.astype(np.float32)


def classify(model, lang: str, ds_all, jsonl: Path, batch_size: int,
             max_segments: int | None) -> dict[str, dict]:
    target = TARGET_CODE[lang]
    done = load_done(jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = len(ds_all) if max_segments is None else min(max_segments, len(ds_all))
    print(f"[{lang}] {n} segments, {len(done)} already done")

    with open(jsonl, "a", encoding="utf-8") as out:
        if jsonl.stat().st_size and jsonl.read_bytes()[-1:] != b"\n":
            out.write("\n")  # terminate a torn last line so new records parse
        for start in range(0, n, batch_size):
            rows = ds_all[start:min(start + batch_size, n)]
            wrote = 0
            for i in range(len(rows["utterance_id"])):
                uid = rows["utterance_id"][i]
                if uid in done:
                    continue
                wav = decode_audio(rows["audio"][i])
                top, top_p, all_probs = model.detect_language(audio=wav)
                probs = dict(all_probs)
                rec = {
                    "utterance_id": uid,
                    "split": rows["split"][i],
                    "duration_s": float(rows["duration_s"][i]),
                    "top_lang": top,
                    "top_prob": round(float(top_p), 5),
                    "p_en": round(float(probs.get("en", 0.0)), 5),
                    "p_target": round(float(probs.get(target, 0.0)), 5),
                    "target": target,
                }
                out.write(json.dumps(rec) + "\n")
                done[uid] = rec
                wrote += 1
            if wrote:
                out.flush()
                os.fsync(out.fileno())
            if (start // batch_size) % 20 == 0:
                print(f"  [{lang}] {min(start + batch_size, n)}/{n}", flush=True)
    return done


def build_v3(lang: str, src: Path, dst_root: Path, recs: dict[str, dict]) -> None:
    from datasets import Audio, DatasetDict, load_from_disk

    ds = load_from_disk(str(src))
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    out_dir = dst_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)

    kept_splits = {}
    for split in ds:
        keep = [i for i, uid in enumerate(ds[split]["utterance_id"])
                if uid in recs and recs[uid]["top_lang"] != "en"]
        kept_splits[split] = ds[split].select(keep)
    new = DatasetDict(kept_splits).cast_column("audio", Audio(sampling_rate=16_000))

    tmp = out_dir / "dataset.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    new.save_to_disk(str(tmp))
    final = out_dir / "dataset"
    if final.exists():
        shutil.rmtree(final)
    tmp.rename(final)

    all_recs = list(recs.values())
    kept = [r for r in all_recs if r["top_lang"] != "en"]
    dropped = [r for r in all_recs if r["top_lang"] == "en"]
    kept_h = sum(r["duration_s"] for r in kept) / 3600
    drop_h = sum(r["duration_s"] for r in dropped) / 3600
    total_h = kept_h + drop_h
    target = TARGET_CODE[lang]
    top_dist = Counter(r["top_lang"] for r in kept).most_common(10)
    stats = {
        "language": lang,
        "target_code": target,
        "total_segments": len(all_recs),
        "dropped_english_segments": len(dropped),
        "kept_segments": len(kept),
        "kept_hours": round(kept_h, 3),
        "dropped_hours": round(drop_h, 3),
        "english_share_by_count": round(len(dropped) / max(1, len(all_recs)), 4),
        "english_share_by_duration": round(drop_h / total_h, 4) if total_h else 0.0,
        "kept_top_language_distribution": [
            {"lang": k, "count": v, "share": round(v / max(1, len(kept)), 4)}
            for k, v in top_dist
        ],
        "kept_fraction_top_equals_target": round(
            sum(1 for r in kept if r["top_lang"] == target) / max(1, len(kept)), 4),
        "kept_per_split": {s: len(kept_splits[s]) for s in kept_splits},
        "rule": "drop if top detected language == 'en'; Whisper large-v3 (faster-whisper)",
    }
    (out_dir / "langid_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[{lang}] kept {stats['kept_segments']}/{stats['total_segments']}, "
          f"English {stats['english_share_by_count']:.1%} by count / "
          f"{stats['english_share_by_duration']:.1%} by duration")


def main() -> None:
    ap = argparse.ArgumentParser(description="Whisper LangID filter: corpus_v2 -> corpus_v3")
    ap.add_argument("--lang", nargs="+", required=True, choices=sorted(TARGET_CODE))
    ap.add_argument("--in-root", default="corpus_v2")
    ap.add_argument("--out-root", default="corpus_v3")
    ap.add_argument("--results-dir", default="results/langid_v3")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Segments per fsync checkpoint (default 16)")
    ap.add_argument("--max-segments", type=int, default=None,
                    help="Smoke test: classify only first N; skips corpus build unless output dirs are scratch")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute-type", default="float16")
    args = ap.parse_args()

    from datasets import Audio, concatenate_datasets, load_from_disk
    from faster_whisper import WhisperModel

    todo = [l for l in args.lang
            if not (Path(args.out_root) / l / "langid_stats.json").exists()]
    for l in set(args.lang) - set(todo):
        print(f"[{l}] already finished ({args.out_root}/{l}/langid_stats.json), skipping")
    if not todo:
        return

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    for lang in todo:
        src = Path(args.in_root) / lang / "dataset"
        dd = load_from_disk(str(src))
        dd = dd.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
        parts = []
        for split in dd:
            parts.append(dd[split].add_column("split", [split] * len(dd[split])))
        ds_all = concatenate_datasets(parts)
        recs = classify(model, lang, ds_all, Path(args.results_dir) / f"{lang}.jsonl",
                        args.batch_size, args.max_segments)
        if args.max_segments is not None:
            # Smoke test: restrict the build to the classified subset.
            print(f"[{lang}] smoke test: {len(recs)} classified, skipping corpus build")
            continue
        total = sum(len(dd[s]) for s in dd)
        missing = total - len(recs)
        if missing:
            sys.exit(f"[{lang}] {missing} segments unclassified; refusing to build corpus_v3")
        build_v3(lang, src, Path(args.out_root), recs)


if __name__ == "__main__":
    main()
