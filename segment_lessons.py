#!/usr/bin/env python3
"""
segment_lessons.py  —  Extract target-language segments from FSI/Peace Corps lesson audio

Lesson recordings interleave English instructor narration with target-language speech.
This script uses Whisper language identification to locate target-language spans, then
applies silence-based splitting and quality filters to produce ASR-ready clips.

Pipeline per file:
  1. Load MP3/WAV at 16 kHz mono (Whisper's native rate)
  2. Slide 5s windows (50% overlap); run Whisper LangID on each window
  3. Mark windows as target-language by lang match + confidence threshold
     (for Whisper-99 languages) or by non-English heuristic (others)
  4. Merge adjacent target windows → coarse target-language spans
  5. Apply pydub silence splitting within each span → candidate segments
  6. Quality filter: duration 1–15s, SNR ≥ 10 dB, speech fraction ≥ 0.60
  7. Export 16 kHz mono WAV + append JSONL metadata record

Usage:
    pip install faster-whisper pydub numpy soundfile
    python segment_lessons.py --lang lin --input fsi_audio/lin/ --out segments/
    python segment_lessons.py --lang hau --input fsi_audio/hau/ --out segments/ --model base

Output:
    segments/
      lin/           <- WAV files: lin_u000001.wav, lin_u000002.wav, ...
      lin.jsonl      <- one JSONL record per segment (Gebru et al. 2018 format)

Resumable: files already listed in the output JSONL are skipped on re-run.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

try:
    from faster_whisper import WhisperModel
except ImportError:
    sys.exit(
        "faster-whisper not installed.\n"
        "Run:  pip install faster-whisper\n"
        "GPU:  pip install faster-whisper[cuda]"
    )

try:
    from pydub import AudioSegment
    from pydub.silence import detect_silence
except ImportError:
    sys.exit(
        "pydub not installed.\n"
        "Run:  pip install pydub\n"
        "Also requires ffmpeg on PATH: https://ffmpeg.org/download.html"
    )

try:
    import soundfile as sf  # noqa: F401 — imported for explicit dependency check
except ImportError:
    sys.exit("soundfile not installed.  Run:  pip install soundfile")


# ---------------------------------------------------------------------------
# Language configuration
#
# Maps our ISO 639-3 working codes → Whisper's internal language tags.
# None = language not in Whisper's 99; fallback: keep any non-English window.
# ---------------------------------------------------------------------------
WHISPER_LANG: dict[str, str | None] = {
    "lin": None,   # Lingala — not in Whisper 99; use non-English heuristic
    "hau": "ha",   # Hausa
    "swh": "sw",   # Swahili
    "amh": "am",   # Amharic
    "som": "so",   # Somali
    "yor": "yo",   # Yoruba
    "ibo": "ig",   # Igbo
    "ewe": None,   # Ewe — not in Whisper 99
    "kri": None,   # Krio — not in Whisper 99
    "hat": "ht",   # Haitian Creole
    "wol": "wo",   # Wolof (in Whisper large-v3; uncertain in smaller models)
    "kin": None,   # Kinyarwanda — not reliably in Whisper 99
    "bam": None,   # Bambara — not in Whisper 99
    "sna": None,   # Shona
    "nde": None,   # Ndebele
    "tir": None,   # Tigrinya
    "orm": None,   # Oromo
    "ny":  None,   # Chichewa/Nyanja
    "lg":  None,   # Luganda
    "tn":  None,   # Setswana
    "ss":  None,   # SiSwati
    "mnk": None,   # Mandinka
    "fuc": None,   # Pulaar
    "dje": None,   # Zarma
    "snk": None,   # Soninke
    "dyo": None,   # Jola-Fonyi
}

SAMPLE_RATE = 16_000  # Whisper's native sample rate (Hz)


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------
def load_audio(path: Path) -> tuple[AudioSegment, np.ndarray]:
    """
    Load any audio file as 16 kHz mono 16-bit PCM.
    Returns (pydub AudioSegment, numpy float32 array normalised to [-1, 1]).
    """
    seg = (
        AudioSegment.from_file(str(path))
        .set_channels(1)
        .set_frame_rate(SAMPLE_RATE)
        .set_sample_width(2)
    )
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
    return seg, samples


def export_wav(seg: AudioSegment, path: Path) -> None:
    seg.export(str(path), format="wav")


# ---------------------------------------------------------------------------
# Language identification
# ---------------------------------------------------------------------------
def find_target_spans(
    audio_np: np.ndarray,
    target_whisper_lang: str | None,
    model: WhisperModel,
    window_s: float,
    conf_thresh: float,
    merge_gap_s: float = 0.5,
) -> list[tuple[float, float]]:
    """
    Slide a window over audio; keep windows classified as target-language.
    Merge adjacent and gap-separated target windows into contiguous spans.

    For languages in Whisper 99 (target_whisper_lang is a string):
        Keep window if detected_lang == target_whisper_lang AND prob >= conf_thresh.

    For other languages (target_whisper_lang is None):
        Keep window if detected_lang != "en".
        Rationale: FSI/Peace Corps instructor always speaks English, so any
        non-English window is almost certainly the target language.

    Returns list of (start_s, end_s) tuples in seconds.
    """
    n = len(audio_np)
    window_n = int(window_s * SAMPLE_RATE)
    stride_n = window_n // 2  # 50% overlap

    if n < window_n:
        # File shorter than one window; classify the whole thing
        window_n = n
        stride_n = n

    target_windows: list[tuple[float, float]] = []

    for start in range(0, n - window_n + 1, stride_n):
        chunk = audio_np[start : start + window_n]
        start_s = start / SAMPLE_RATE
        end_s = (start + window_n) / SAMPLE_RATE

        try:
            result = model.detect_language(chunk)
            # faster-whisper returns (lang: str, prob: float)
            # Older versions may return (lang, {lang: prob, ...}) — handle both
            if isinstance(result, (tuple, list)) and len(result) == 2:
                lang, prob_or_dict = result
                if isinstance(prob_or_dict, dict):
                    top_prob = prob_or_dict.get(lang, 0.0)
                else:
                    top_prob = float(prob_or_dict)
            else:
                logging.debug(f"Unexpected detect_language output at {start_s:.1f}s: {result}")
                continue
        except Exception as exc:
            logging.debug(f"LangID failed at {start_s:.1f}s: {exc}")
            continue

        if target_whisper_lang is not None:
            if lang == target_whisper_lang and top_prob >= conf_thresh:
                target_windows.append((start_s, end_s))
        else:
            # Non-English heuristic
            if lang != "en":
                target_windows.append((start_s, end_s))

    if not target_windows:
        return []

    # Merge overlapping or close windows
    merged: list[tuple[float, float]] = [target_windows[0]]
    for s, e in target_windows[1:]:
        ps, pe = merged[-1]
        if s <= pe + merge_gap_s:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))

    return merged


# ---------------------------------------------------------------------------
# Silence-based segmentation within a span
# ---------------------------------------------------------------------------
def segment_span(
    audio_seg: AudioSegment,
    start_s: float,
    end_s: float,
    silence_thresh_db: int,
    silence_min_ms: int,
) -> list[tuple[AudioSegment, float, float]]:
    """
    Slice the (start_s, end_s) portion and split on silence.
    Returns list of (chunk_AudioSegment, abs_start_s, abs_end_s).
    """
    start_ms = int(start_s * 1000)
    end_ms = int(end_s * 1000)
    span = audio_seg[start_ms:end_ms]

    silences = detect_silence(
        span,
        min_silence_len=silence_min_ms,
        silence_thresh=silence_thresh_db,
        seek_step=10,
    )

    if not silences:
        return [(span, start_s, end_s)]

    chunks: list[tuple[AudioSegment, float, float]] = []
    prev_end_ms = 0

    for sil_start_ms, sil_end_ms in silences:
        if sil_start_ms > prev_end_ms:
            chunk = span[prev_end_ms:sil_start_ms]
            c_start = start_s + prev_end_ms / 1000.0
            c_end = start_s + sil_start_ms / 1000.0
            chunks.append((chunk, c_start, c_end))
        prev_end_ms = sil_end_ms

    # Trailing speech after last silence
    if prev_end_ms < len(span):
        chunk = span[prev_end_ms:]
        c_start = start_s + prev_end_ms / 1000.0
        chunks.append((chunk, c_start, end_s))

    return chunks


# ---------------------------------------------------------------------------
# Quality metrics
# ---------------------------------------------------------------------------
def compute_snr_db(chunk_np: np.ndarray, frame_ms: int = 20) -> float:
    """
    Frame-level SNR estimate.
    Signal = mean RMS of loudest 25% of frames.
    Noise  = mean RMS of quietest 25% of frames.
    Returns dB, or 0.0 if chunk is too short to measure.
    """
    frame_n = int(frame_ms / 1000 * SAMPLE_RATE)
    if len(chunk_np) < frame_n * 4:
        return 0.0

    frames = [
        chunk_np[i : i + frame_n]
        for i in range(0, len(chunk_np) - frame_n + 1, frame_n)
    ]
    rms_values = np.array([np.sqrt(np.mean(f ** 2)) for f in frames])
    rms_values = rms_values[rms_values > 1e-9]  # discard near-silence frames

    if len(rms_values) < 4:
        return 0.0

    rms_sorted = np.sort(rms_values)
    q = len(rms_sorted) // 4
    signal_rms = float(np.mean(rms_sorted[3 * q :]))
    noise_rms = float(np.mean(rms_sorted[:q]))

    if noise_rms < 1e-9:
        return 99.0
    return 20.0 * float(np.log10(signal_rms / noise_rms))


def compute_speech_fraction(
    chunk: AudioSegment,
    silence_thresh_db: int,
    silence_min_ms: int = 80,
) -> float:
    """Fraction of chunk that is non-silent (proxy for speech fraction)."""
    silences = detect_silence(
        chunk,
        min_silence_len=silence_min_ms,
        silence_thresh=silence_thresh_db,
        seek_step=10,
    )
    silence_ms = sum(e - s for s, e in silences)
    return 1.0 - silence_ms / max(len(chunk), 1)


def chunk_to_numpy(chunk: AudioSegment) -> np.ndarray:
    raw = chunk.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    return np.array(raw.get_array_of_samples(), dtype=np.float32) / 32768.0


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def _infer_source_type(path: Path) -> str:
    p = str(path).lower()
    if "fsi" in p:
        return "FSI"
    if "peace" in p or "/pc" in p or "_pc_" in p:
        return "PeaceCorps"
    if "dli" in p:
        return "DLI"
    return "Unknown"


def process_file(
    src: Path,
    lang: str,
    source_name: str,
    model,  # WhisperModel or None (when --skip-langid)
    wav_dir: Path,
    jsonl_path: Path,
    counter_start: int,
    args: argparse.Namespace,
) -> int:
    """
    Process one lesson file.
    Returns number of segments exported (used to advance the global counter).
    """
    logging.info(f"  [{src.name}]")

    try:
        audio_seg, audio_np = load_audio(src)
    except Exception as exc:
        logging.warning(f"    Load failed: {exc}")
        return 0

    duration_s = len(audio_np) / SAMPLE_RATE
    logging.info(f"    duration={duration_s:.1f}s")

    if model is None:
        # --skip-langid: treat entire file as one target span; rely on VAD + quality filters
        spans = [(0.0, duration_s)]
        logging.info(f"    skip-langid mode: treating full file as target span")
    else:
        target_whisper_lang = WHISPER_LANG.get(lang)
        if target_whisper_lang is None:
            logging.debug(f"    LangID mode: non-English heuristic (lang '{lang}' not in Whisper 99)")

        spans = find_target_spans(
            audio_np,
            target_whisper_lang,
            model,
            window_s=args.window,
            conf_thresh=args.conf,
            merge_gap_s=0.5,
        )
    total_span_s = sum(e - s for s, e in spans)
    logging.info(f"    target spans={len(spans)}  ({total_span_s:.1f}s / {duration_s:.1f}s = {total_span_s/max(duration_s,1)*100:.0f}%)")

    n_exported = 0
    counter = counter_start
    n_short = n_long = n_snr = n_speech = 0

    for span_start, span_end in spans:
        raw_chunks = segment_span(
            audio_seg,
            span_start,
            span_end,
            silence_thresh_db=args.silence_thresh,
            silence_min_ms=args.silence_min_ms,
        )
        for chunk, c_start, c_end in raw_chunks:
            dur_s = len(chunk) / 1000.0

            if dur_s < args.min_dur:
                n_short += 1
                continue
            if dur_s > args.max_dur:
                n_long += 1
                continue

            chunk_np = chunk_to_numpy(chunk)
            snr = compute_snr_db(chunk_np)
            if snr < args.min_snr:
                n_snr += 1
                continue

            sf_val = compute_speech_fraction(chunk, args.silence_thresh)
            if sf_val < args.speech_frac:
                n_speech += 1
                continue

            uid = f"{lang}_u{counter:06d}"
            wav_path = wav_dir / f"{uid}.wav"

            try:
                export_wav(chunk, wav_path)
            except Exception as exc:
                logging.warning(f"    Export failed for {uid}: {exc}")
                continue

            record = {
                "utterance_id": uid,
                "language": lang,
                "source_file": src.name,
                "source_type": source_name or _infer_source_type(src),
                "license": "PD-US",
                "license_basis": "17 U.S.C. § 105",
                "audio_path": f"{lang}/{uid}.wav",
                "duration_s": round(dur_s, 3),
                "snr_db": round(snr, 1),
                "speech_fraction": round(sf_val, 3),
                "span_start_s": round(c_start, 3),
                "span_end_s": round(c_end, 3),
                "text": None,
                "aligned_text": None,
            }
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            counter += 1
            n_exported += 1

    logging.info(
        f"    exported={n_exported}  "
        f"(rejected: short={n_short} long={n_long} snr={n_snr} speech={n_speech})"
    )
    return n_exported


# ---------------------------------------------------------------------------
# Progress tracking (resumable)
# ---------------------------------------------------------------------------
def load_done_files(jsonl_path: Path) -> set[str]:
    """Return source filenames already present in the output JSONL."""
    done: set[str] = set()
    if not jsonl_path.exists():
        return done
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["source_file"])
        except Exception:
            pass
    return done


def count_existing_segments(jsonl_path: Path) -> int:
    if not jsonl_path.exists():
        return 0
    return sum(1 for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Segment FSI/Peace Corps lesson audio into target-language clips"
    )
    ap.add_argument("--lang", required=True,
                    help="ISO 639-3 language code (e.g. lin, hau, swh)")
    ap.add_argument("--input", required=True,
                    help="Directory of source MP3/WAV files")
    ap.add_argument("--out", default="segments",
                    help="Output root directory (default: segments/)")
    ap.add_argument("--source-name", default=None,
                    help="Source label written to JSONL: FSI, PeaceCorps, DLI (auto-detected if omitted)")
    ap.add_argument("--model", default="tiny",
                    choices=["tiny", "base", "small", "medium", "large-v3"],
                    help="Whisper model size for LangID (default: tiny)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Compute device (default: cpu)")
    ap.add_argument("--window", type=float, default=5.0,
                    help="LangID sliding window in seconds (default: 5.0)")
    ap.add_argument("--conf", type=float, default=0.85,
                    help="LangID confidence threshold for Whisper-99 langs (default: 0.85)")
    ap.add_argument("--min-dur", type=float, default=1.0,
                    help="Minimum segment duration in seconds (default: 1.0)")
    ap.add_argument("--max-dur", type=float, default=15.0,
                    help="Maximum segment duration in seconds (default: 15.0)")
    ap.add_argument("--min-snr", type=float, default=10.0,
                    help="Minimum SNR in dB (default: 10.0)")
    ap.add_argument("--speech-frac", type=float, default=0.60,
                    help="Minimum speech fraction 0–1 (default: 0.60)")
    ap.add_argument("--silence-thresh", type=int, default=-35,
                    help="Silence threshold in dBFS for splitting (default: -35)")
    ap.add_argument("--silence-min-ms", type=int, default=500,
                    help="Minimum silence length ms to trigger a split (default: 500)")
    ap.add_argument("--skip-langid", action="store_true",
                    help="Skip Whisper LangID; use VAD+quality filters only (fast CPU mode). "
                         "Includes some English instruction segments; filter later with "
                         "pseudo-labeling or quality_filter.py.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    in_dir = Path(args.input)
    if not in_dir.is_dir():
        sys.exit(f"Input directory not found: {in_dir}")

    out_root = Path(args.out)
    wav_dir = out_root / args.lang
    wav_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / f"{args.lang}.jsonl"

    _seen_paths: set[Path] = set()
    _raw_sources = (
        list(in_dir.glob("*.mp3"))
        + list(in_dir.glob("*.MP3"))
        + list(in_dir.glob("*.wav"))
        + list(in_dir.glob("*.WAV"))
        + list(in_dir.glob("*.m4a"))
    )
    sources = []
    for _p in sorted(_raw_sources):
        _rp = _p.resolve()
        if _rp not in _seen_paths:
            _seen_paths.add(_rp)
            sources.append(_p)
    if not sources:
        sys.exit(f"No audio files found in {in_dir}")

    if not args.skip_langid and args.lang not in WHISPER_LANG:
        logging.warning(
            f"Lang code '{args.lang}' not in WHISPER_LANG map — "
            "treating as non-Whisper-99 (non-English heuristic)"
        )

    logging.info(f"Lang: {args.lang}  |  source files: {len(sources)}  |  model: {'none (skip-langid)' if args.skip_langid else args.model}")
    logging.info(f"Filters: dur {args.min_dur}-{args.max_dur}s  SNR>={args.min_snr}dB  speech>={args.speech_frac}")

    if args.skip_langid:
        model = None
        logging.info("LangID: SKIPPED (fast CPU mode)")
    else:
        logging.info(f"Loading Whisper {args.model} on {args.device} ...")
        model = WhisperModel(args.model, device=args.device, compute_type="int8")

    done_files = load_done_files(jsonl_path)
    counter = count_existing_segments(jsonl_path)
    total_exported = 0

    for src in sources:
        if src.name in done_files:
            logging.info(f"  Skip (already done): {src.name}")
            continue
        n = process_file(
            src, args.lang, args.source_name, model, wav_dir, jsonl_path, counter, args
        )
        counter += n
        total_exported += n

    logging.info(f"\nTotal segments exported: {total_exported}")
    logging.info(f"JSONL:  {jsonl_path}")
    logging.info(f"WAVs:   {wav_dir}/")

    if total_exported == 0 and sources:
        logging.warning(
            "No segments passed filters. Suggestions:\n"
            "  - Lower --conf (try 0.70) if target lang is not in Whisper 99\n"
            "  - Lower --min-snr (try 5.0) for older recordings\n"
            "  - Lower --silence-thresh (try -45) if audio is quiet\n"
            "  - Use --model base or --model small for better LangID accuracy"
        )


if __name__ == "__main__":
    main()
