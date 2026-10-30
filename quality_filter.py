#!/usr/bin/env python3
"""
quality_filter.py — Re-filter AfriBase-Speech segment manifests post-hoc.

Reads JSONL produced by segment_lessons.py; applies configurable thresholds;
writes a filtered copy. Useful for generating stricter subsets for experiments
without re-running the full LangID pipeline.

Usage:
    python quality_filter.py --input segments/lin.jsonl --out filtered/
    python quality_filter.py --input segments/lin.jsonl --out filtered/ \\
        --min-snr 15 --min-dur 2.0 --max-dur 12.0 --min-speech 0.70
    python quality_filter.py --input-dir segments/ --out filtered/ --min-snr 15

Outputs:
    filtered/{lang}.jsonl              — records passing all thresholds
    filtered/{lang}_filter_stats.json  — filter summary
"""

import argparse
import json
import sys
from pathlib import Path


def filter_jsonl(in_path: Path, out_dir: Path, args: argparse.Namespace) -> dict:
    audio_root = Path(args.audio_root) if args.audio_root else in_path.parent

    records = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    total = len(records)
    kept = []
    rejected = {"snr": 0, "duration": 0, "speech": 0, "missing_file": 0}

    for rec in records:
        dur = rec.get("duration_s", 0.0)
        snr = rec.get("snr_db", 0.0)
        sf = rec.get("speech_fraction", 0.0)

        if dur < args.min_dur or dur > args.max_dur:
            rejected["duration"] += 1
            continue
        if snr < args.min_snr:
            rejected["snr"] += 1
            continue
        if sf < args.min_speech:
            rejected["speech"] += 1
            continue

        if args.verify_files:
            audio_path = Path(rec.get("audio_path", ""))
            if not audio_path.is_absolute():
                audio_path = audio_root / audio_path
            if not audio_path.exists():
                rejected["missing_file"] += 1
                continue

        kept.append(rec)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / in_path.name
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "input": str(in_path),
        "total": total,
        "kept": len(kept),
        "retention_pct": round(len(kept) / max(total, 1) * 100, 1),
        "rejected": rejected,
        "total_duration_h": round(
            sum(r.get("duration_s", 0) for r in kept) / 3600, 2
        ),
        "thresholds": {
            "min_snr_db": args.min_snr,
            "min_dur_s": args.min_dur,
            "max_dur_s": args.max_dur,
            "min_speech_fraction": args.min_speech,
        },
    }
    stats_path = out_dir / f"{in_path.stem}_filter_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-hoc quality filter for AfriBase-Speech manifests")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Single JSONL file")
    group.add_argument("--input-dir", help="Directory of *.jsonl files (one per language)")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--lang", nargs="+", help="Filter to these lang codes (with --input-dir)")
    ap.add_argument("--min-snr", type=float, default=10.0)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=15.0)
    ap.add_argument("--min-speech", type=float, default=0.60)
    ap.add_argument("--verify-files", action="store_true",
                    help="Skip records whose WAV file does not exist")
    ap.add_argument("--audio-root", default=None,
                    help="Root dir for resolving relative audio_path values")
    args = ap.parse_args()

    out_dir = Path(args.out)

    if args.input:
        paths = [Path(args.input)]
    else:
        paths = sorted(Path(args.input_dir).glob("*.jsonl"))
        if args.lang:
            paths = [p for p in paths if p.stem in args.lang]

    for in_path in paths:
        if not in_path.exists():
            print(f"Not found: {in_path}", file=sys.stderr)
            continue
        stats = filter_jsonl(in_path, out_dir, args)
        print(
            f"{in_path.stem}: {stats['kept']}/{stats['total']} "
            f"({stats['retention_pct']}%) -> {stats['total_duration_h']}h  "
            f"rejected={stats['rejected']}"
        )


if __name__ == "__main__":
    main()
