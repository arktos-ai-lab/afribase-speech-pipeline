#!/usr/bin/env python3
"""
eval_fleurs.py — Evaluate ASR models on FLEURS test splits.

Generates Table 2 from the AfriBase-Speech paper:
  rows = study languages
  columns = [zero-shot Whisper-small, fine-tuned Whisper-small, Whisper-large-v3 (oracle)]

Supports Whisper (via HF pipeline) and Wav2Vec2/MMS (via AutoProcessor).
Results are appended to a CSV for incremental runs.

Usage:
    # Zero-shot baseline (Condition A in the paper)
    python eval_fleurs.py --model openai/whisper-small --lang lin hau swh amh yor \\
        --system whisper --label whisper_small_zs --out results/fleurs.csv

    # Fine-tuned checkpoint (Condition B)
    python eval_fleurs.py --model runs/whisper-small-lin/best --lang lin \\
        --system whisper --label whisper_small_ft --out results/fleurs.csv

    # Oracle ceiling
    python eval_fleurs.py --model openai/whisper-large-v3 --lang lin hau swh amh \\
        --system whisper --label whisper_large_zs --out results/fleurs.csv

    # Print table from accumulated CSV
    python eval_fleurs.py --table results/fleurs.csv
"""

import argparse
import csv
import io
import json
import os
import sys
from pathlib import Path

import numpy as np


def _load_audio(audio_field: dict, sr: int = 16_000):
    """Decode a datasets Audio(decode=False) field without torchcodec.

    Prefers an on-disk path when one actually resolves (e.g. real cached
    files); falls back to embedded bytes (e.g. our own packaged corpora,
    which store WAV bytes inline rather than an external path).
    """
    import librosa
    import soundfile as sf

    path = audio_field.get("path")
    if path and os.path.exists(path):
        arr, file_sr = librosa.load(path, sr=sr, mono=True)
        return arr, file_sr
    arr, file_sr = sf.read(io.BytesIO(audio_field["bytes"]))
    if file_sr != sr:
        arr = librosa.resample(arr.astype(np.float32), orig_sr=file_sr, target_sr=sr)
        file_sr = sr
    return arr, file_sr


FLEURS_CODE: dict[str, str] = {
    "lin": "ln_cd",
    "hau": "ha_ng",
    "swh": "sw_ke",
    "amh": "am_et",
    "som": "so_so",
    "yor": "yo_ng",
    "ibo": "ig_ng",
    "wol": "wo_sn",
    "orm": "om_et",
    "sna": "sn_zw",
    "hat": "ht_ht",
    "afr": "af_za",
}

WHISPER_LANG_HINT: dict[str, str | None] = {
    "lin": "ln",
    "hau": "ha",
    "swh": "sw",
    "amh": "am",
    "som": "so",
    "yor": "yo",
    "ibo": "ig",
    "wol": "wo",
    "orm": "om",
    "sna": "sn",
    "hat": "ht",
    "afr": "af",
}

RESULT_FIELDS = ("label", "language", "fleurs_code", "model", "system", "n_utt", "wer", "cer")


def compute_metrics(refs: list[str], hyps: list[str]) -> dict:
    import jiwer
    tr = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    pairs = [(tr(r), tr(h)) for r, h in zip(refs, hyps) if tr(r).strip()]
    if not pairs:
        return {"wer": float("nan"), "cer": float("nan"), "n_utt": 0}
    rs = [r for r, _ in pairs]
    hs = [h for _, h in pairs]
    return {
        "wer": round(jiwer.wer(rs, hs), 4),
        "cer": round(jiwer.cer(rs, hs), 4),
        "n_utt": len(pairs),
    }


def _load_ckpt(path: Path | None) -> dict[int, tuple[str, str]]:
    done: dict[int, tuple[str, str]] = {}
    if path is not None and path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line from a crash; utterance is redone
                done[rec["i"]] = (rec["ref"], rec["hyp"])
    return done


def eval_whisper(model_id: str, lang: str, device: str, no_lang_hint: bool = False,
                 checkpoint: Path | None = None, max_utt: int | None = None) -> dict:
    import torch
    from datasets import Audio, load_dataset
    from transformers import pipeline as hf_pipeline

    fleurs_code = FLEURS_CODE.get(lang)
    if not fleurs_code:
        return {"error": f"No FLEURS code for '{lang}'"}

    try:
        ds = load_dataset("google/fleurs", fleurs_code, split="test", trust_remote_code=True)
        # decode=False avoids datasets' torchcodec-based lazy decoder (broken on
        # this machine's torch build); decode manually via _load_audio instead.
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
        if max_utt:  # smoke tests only
            ds = ds.select(range(min(max_utt, len(ds))))
    except Exception as exc:
        return {"error": f"FLEURS load failed: {exc}"}

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    lang_hint = None if no_lang_hint else WHISPER_LANG_HINT.get(lang)
    base_kwargs = {"language": lang_hint} if lang_hint else {}
    # no_repeat_ngram_size guards against the repetition-loop hallucination
    # failure mode (model gets stuck repeating a phrase, often in the wrong
    # language, until it hits the generation length ceiling).
    base_kwargs["no_repeat_ngram_size"] = 3
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=dtype,
        generate_kwargs=base_kwargs,
    )

    # Per-utterance checkpoint (flush+fsync): a crash loses at most one utterance.
    done = _load_ckpt(checkpoint)
    ck = None
    if checkpoint is not None:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        ck = open(checkpoint, "a", encoding="utf-8")
        if checkpoint.stat().st_size and checkpoint.read_bytes()[-1:] != b"\n":
            ck.write("\n")  # terminate a torn last line so new records parse
    if done:
        print(f"  Resuming: {len(done)} utterances already in {checkpoint}")

    refs, hyps = [], []
    for idx, item in enumerate(ds):
        if idx in done:
            refs.append(done[idx][0])
            hyps.append(done[idx][1])
            continue
        ref = item.get("raw_transcription") or item.get("transcription", "")
        wav, _ = _load_audio(item["audio"])
        wav = wav.astype(np.float32)
        # Only force the long-form/timestamp decoding path for genuinely long
        # (>30s) clips -- applying it to every clip (as an earlier fix did)
        # makes ALL clips, including short ones, far more prone to the
        # repetition-loop failure above; short-form decoding is more stable.
        long_form = (len(wav) / 16_000) > 30
        hyp = pipe(wav, return_timestamps=True if long_form else False)["text"]
        if ck is not None:
            ck.write(json.dumps({"i": idx, "ref": ref, "hyp": hyp}) + "\n")
            ck.flush()
            os.fsync(ck.fileno())
        refs.append(ref)
        hyps.append(hyp)
    if ck is not None:
        ck.close()

    m = compute_metrics(refs, hyps)
    return {**m, "fleurs_code": fleurs_code}


def eval_wav2vec2(model_id: str, lang: str, device: str) -> dict:
    import torch
    from datasets import Audio, load_dataset
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    fleurs_code = FLEURS_CODE.get(lang)
    if not fleurs_code:
        return {"error": f"No FLEURS code for '{lang}'"}

    try:
        ds = load_dataset("google/fleurs", fleurs_code, split="test", trust_remote_code=True)
        ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    except Exception as exc:
        return {"error": f"FLEURS load failed: {exc}"}

    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device).eval()

    refs, hyps = [], []
    with torch.no_grad():
        for item in ds:
            ref = item.get("raw_transcription") or item.get("transcription", "")
            wav, _ = _load_audio(item["audio"])
            wav = wav.astype(np.float32)
            inputs = processor(wav, sampling_rate=16_000, return_tensors="pt").to(device)
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
            hyp = processor.decode(pred_ids[0])
            refs.append(ref)
            hyps.append(hyp)

    m = compute_metrics(refs, hyps)
    return {**m, "fleurs_code": fleurs_code}


def append_results(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RESULT_FIELDS))
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


def print_table(csv_path: Path) -> None:
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))
    labels = sorted({r["label"] for r in rows})
    langs = sorted({r["language"] for r in rows})

    # Pivot: lang × label → WER
    table: dict[str, dict[str, str]] = {lang: {} for lang in langs}
    for row in rows:
        table[row["language"]][row["label"]] = row.get("wer", "")

    header = ["Language"] + labels
    col_w = [max(len(h), 10) for h in header]

    def fmt_row(vals):
        return " | ".join(str(v).ljust(w) for v, w in zip(vals, col_w))

    print("\n# FLEURS WER Table\n")
    print(fmt_row(header))
    print("-" * (sum(col_w) + 3 * len(col_w)))
    for lang in langs:
        vals = [lang] + [table[lang].get(lbl, "—") for lbl in labels]
        print(fmt_row(vals))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate ASR models on FLEURS test splits")
    ap.add_argument("--model", help="HF model ID or local path")
    ap.add_argument("--lang", nargs="+", help="ISO 639-3 language codes")
    ap.add_argument("--system", default="whisper", choices=["whisper", "wav2vec2"],
                    help="Inference backend (default: whisper)")
    ap.add_argument("--label", default=None,
                    help="Label written to results CSV (e.g. whisper_small_ft)")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--out", help="Results CSV path (appended)")
    ap.add_argument("--table", help="Print pivot table from existing CSV and exit")
    ap.add_argument("--max-utt", type=int, default=None,
                    help="Evaluate only the first N utterances (smoke tests only)")
    ap.add_argument("--no-lang-hint", action="store_true",
                    help="Disable forced-language decoding (true auto-detect condition)")
    args = ap.parse_args()

    if args.table:
        print_table(Path(args.table))
        return

    if not args.model or not args.lang:
        ap.print_help()
        sys.exit(1)

    label = args.label or Path(args.model).name.replace("/", "_")
    rows = []

    for lang in args.lang:
        print(f"\n[{lang}]")
        if args.out and Path(args.out).exists():
            with open(args.out, encoding="utf-8") as f:
                if any(r["label"] == label and r["language"] == lang for r in csv.DictReader(f)):
                    print(f"  Already in {args.out} (label={label}); skipping")
                    continue
        if args.system == "whisper":
            ckpt = (Path(args.out).parent / "ckpt" / f"{label}_{lang}.jsonl") if args.out else None
            result = eval_whisper(args.model, lang, args.device, no_lang_hint=args.no_lang_hint,
                                  checkpoint=ckpt, max_utt=args.max_utt)
        else:
            result = eval_wav2vec2(args.model, lang, args.device)

        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        print(f"  WER={result['wer']:.4f}  CER={result['cer']:.4f}  n={result['n_utt']}")
        rows.append({
            "label": label,
            "language": lang,
            "fleurs_code": result.get("fleurs_code", ""),
            "model": args.model,
            "system": args.system,
            "n_utt": result["n_utt"],
            "wer": result["wer"],
            "cer": result["cer"],
        })

    if args.out and rows:
        out_path = Path(args.out)
        append_results(out_path, rows)
        print(f"\nAppended {len(rows)} rows -> {out_path}")

    # Quick summary table
    if rows:
        print(f"\n{'Lang':<6} {'WER':>8} {'CER':>8} {'n':>6}")
        print("-" * 30)
        for row in rows:
            print(f"{row['language']:<6} {row['wer']:>8.4f} {row['cer']:>8.4f} {row['n_utt']:>6}")


if __name__ == "__main__":
    main()
