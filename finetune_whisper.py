#!/usr/bin/env python3
"""
finetune_whisper.py — Fine-tune openai/whisper-small on AfriBase-Speech Tier 1a audio.

Supports:
  - Single language or batch across multiple languages
  - Validation on held-out split from package_corpus.py
  - Optional FLEURS evaluation after training
  - Saves best WER checkpoint

Usage:
    # Single language
    python finetune_whisper.py --lang lin --corpus corpus/lin/dataset \\
        --out runs/whisper-small-lin --device cuda

    # Batch (all languages with packaged corpus)
    python finetune_whisper.py --corpus-root corpus/ --out-root runs/ \\
        --device cuda --epochs 15

Output per language:
    runs/whisper-small-{lang}/
        best/           <- best WER checkpoint (load with WhisperProcessor + WhisperForConditionalGeneration)
        logs/           <- TensorBoard training logs
        eval_result.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np


WHISPER_BASE = "openai/whisper-small"

# FLEURS language codes (ISO 639-3 → FLEURS identifier)
FLEURS_CODE: dict[str, str] = {
    "lin": "ln_cd",
    "hau": "hau_NG",
    "swh": "sw_KE",
    "amh": "am_ET",
    "som": "so_SO",
    "yor": "yo_NG",
    "ibo": "ig_NG",
    "wol": "wo_SN",
    "orm": "om_ET",
    "sna": "sn_ZW",
    "hat": "ht_HT",
    "kri": None,
    "ewe": None,
}

# Whisper language tags (may differ from FLEURS codes)
WHISPER_LANG_TAG: dict[str, str | None] = {
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
    "kri": None,
    "ewe": None,
}


def compute_wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer
    tr = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])
    pairs = [(tr(r), tr(h)) for r, h in zip(refs, hyps) if tr(r).strip()]
    if not pairs:
        return 1.0
    return jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


@dataclasses.dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], "torch.Tensor"]]]
    ) -> Dict[str, "torch.Tensor"]:
        import torch
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip leading BOS token if present (Whisper-specific)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def load_fleurs_test(lang: str):
    from datasets import load_dataset
    fleurs_code = FLEURS_CODE.get(lang)
    if not fleurs_code:
        print(f"  No FLEURS code for '{lang}' — skipping FLEURS eval")
        return None
    try:
        return load_dataset("google/fleurs", fleurs_code, split="test", trust_remote_code=True)
    except Exception as exc:
        print(f"  Warning: could not load FLEURS/{fleurs_code}: {exc}")
        return None


def eval_on_fleurs(model_dir: str, lang: str, device: str) -> float | None:
    import torch
    from transformers import pipeline as hf_pipeline

    fleurs_ds = load_fleurs_test(lang)
    if fleurs_ds is None:
        return None

    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model_dir,
        device=device,
        torch_dtype=dtype,
        generate_kwargs={"language": WHISPER_LANG_TAG.get(lang)} if WHISPER_LANG_TAG.get(lang) else {},
    )
    refs, hyps = [], []
    for item in fleurs_ds:
        ref = item.get("raw_transcription") or item.get("transcription", "")
        wav = item["audio"]["array"].astype(np.float32)
        hyp = pipe(wav)["text"]
        refs.append(ref)
        hyps.append(hyp)

    wer = compute_wer(refs, hyps)
    print(f"  FLEURS WER ({FLEURS_CODE.get(lang)}): {wer:.4f}  ({len(refs)} utterances)")
    return round(wer, 4)


def finetune(
    lang: str,
    corpus_path: Path,
    out_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    eval_fleurs: bool,
    max_steps: int | None = None,
) -> dict:
    import torch
    from datasets import Audio, load_from_disk
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from transformers.trainer_utils import get_last_checkpoint

    import io
    import soundfile as sf

    print(f"\n[{lang}] Loading corpus from {corpus_path} ...")
    ds = load_from_disk(str(corpus_path))
    # decode=False avoids datasets' torchcodec-based lazy decoder (broken on this
    # machine's torch build); audio is embedded as WAV bytes (not an external
    # path), so we decode manually with soundfile instead.
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    print(f"  {len(ds['train'])} train / {len(ds['validation'])} validation")

    lang_tag = WHISPER_LANG_TAG.get(lang)
    processor = WhisperProcessor.from_pretrained(
        WHISPER_BASE,
        language=lang_tag,
        task="transcribe",
    )

    def prepare(batch):
        array, sr = sf.read(io.BytesIO(batch["audio"]["bytes"]))
        batch["input_features"] = processor.feature_extractor(
            array, sampling_rate=sr
        ).input_features[0]
        batch["labels"] = processor.tokenizer(
            batch["text"], max_length=448, truncation=True
        ).input_ids
        return batch

    ds = ds.map(prepare, remove_columns=ds["train"].column_names, num_proc=1)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": compute_wer(label_str, pred_str)}

    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_BASE)
    model.generation_config.language = lang_tag
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    out_dir.mkdir(parents=True, exist_ok=True)
    best_dir = out_dir / "best"
    logs_dir = out_dir / "logs"
    # transformers 5.x dropped the logging_dir argument; TensorBoard reads this env var instead
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", str(logs_dir))
    steps_per_epoch = max(1, len(ds["train"]) // batch_size)
    total_steps = max_steps if max_steps else epochs * steps_per_epoch  # max_steps: smoke tests
    ckpt_every = max_steps if max_steps else max(100, steps_per_epoch)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=max(1, 8 // batch_size),
        learning_rate=lr,
        warmup_steps=min(200, steps_per_epoch),
        max_steps=total_steps,
        gradient_checkpointing=True,
        fp16=device.startswith("cuda"),
        eval_strategy="steps",
        per_device_eval_batch_size=batch_size,
        save_steps=ckpt_every,
        save_total_limit=2,  # best checkpoint is always kept with load_best_model_at_end
        eval_steps=ckpt_every,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    last_checkpoint = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    if last_checkpoint:
        print(f"  Resuming from checkpoint: {last_checkpoint}")
    print(f"  Training for {epochs} epochs ({total_steps} steps) ...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    result: dict = {
        "language": lang,
        "model": WHISPER_BASE,
        "train_size": len(ds["train"]),
        "val_size": len(ds["validation"]),
        "epochs": epochs,
    }

    if eval_fleurs:
        try:
            wer = eval_on_fleurs(str(best_dir), lang, device)
            if wer is not None:
                result["fleurs_wer"] = wer
        except Exception as exc:
            # Non-essential (the separate eval_fleurs.py pass covers this properly
            # afterward) -- never let this crash a finished language's save.
            print(f"  Warning: internal FLEURS eval failed for {lang}: {exc}")

    result_path = out_dir / "eval_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"  Saved: {best_dir}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune Whisper-small on AfriBase-Speech")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", help="Path to HF DatasetDict (from package_corpus.py)")
    group.add_argument("--corpus-root", help="Root with {lang}/dataset/ subdirs")
    ap.add_argument("--lang", nargs="+",
                    help="Language code(s) — required with --corpus, filter with --corpus-root")
    ap.add_argument("--out", help="Output dir (single-lang mode)")
    ap.add_argument("--out-root", default="runs", help="Output root (batch mode)")
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override total steps (smoke tests only)")
    ap.add_argument("--no-fleurs-eval", action="store_true",
                    help="Skip FLEURS evaluation after training")
    args = ap.parse_args()

    if args.corpus:
        if not args.lang:
            sys.exit("--lang is required with --corpus")
        lang = args.lang[0]
        out_dir = Path(args.out or f"runs/whisper-small-{lang}")
        finetune(lang, Path(args.corpus), out_dir, args.device, args.epochs,
                 args.batch_size, args.lr, not args.no_fleurs_eval, args.max_steps)
    else:
        root = Path(args.corpus_root)
        langs = args.lang or [p.name for p in sorted(root.iterdir()) if p.is_dir()]
        out_root = Path(args.out_root)
        all_results: dict[str, dict] = {}

        for lang in langs:
            corpus_path = root / lang / "dataset"
            if not corpus_path.exists():
                print(f"Skipping {lang}: {corpus_path} not found")
                continue
            out_dir = out_root / f"whisper-small-{lang}"
            done_json = out_dir / "eval_result.json"
            if (out_dir / "best").exists() and done_json.exists():
                print(f"Skipping {lang}: already finished ({done_json})")
                all_results[lang] = json.loads(done_json.read_text(encoding="utf-8"))
                continue
            result = finetune(lang, corpus_path, out_dir, args.device, args.epochs,
                              args.batch_size, args.lr, not args.no_fleurs_eval, args.max_steps)
            all_results[lang] = result

        summary = out_root / "finetune_summary.json"
        summary.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nSummary: {summary}")

        # Print quick WER table
        print("\n# FLEURS WER Summary")
        print(f"{'Lang':<6} {'WER':>8}")
        print("-" * 16)
        for lang, r in sorted(all_results.items()):
            wer = r.get("fleurs_wer")
            print(f"{lang:<6} {f'{wer:.4f}':>8}" if wer is not None else f"{lang:<6} {'n/a':>8}")


if __name__ == "__main__":
    main()
