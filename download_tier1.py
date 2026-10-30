#!/usr/bin/env python3
"""
download_tier1.py  —  Download FSI / Peace Corps / DLI lesson audio (public domain)

All material produced by US federal government employees (FSI = State Dept,
Peace Corps, DLI = Dept of Defense) is public domain under 17 U.S.C. § 105.
No license negotiation required.

Primary source: Internet Archive (archive.org). Each item's files are listed
via the metadata API and MP3s are downloaded from the S3-compatible IA CDN.

Usage:
    pip install requests
    python download_tier1.py --lang lin --out fsi_audio/
    python download_tier1.py --lang all --out fsi_audio/
    python download_tier1.py --lang yor swh lin --out fsi_audio/ --dry-run

Output:
    fsi_audio/
      lin/   <- MP3 files from the FSI Lingala course
      yor/   <- MP3 files from the FSI Yoruba course (4 IA items merged)
      ...

Resumable: skips already-downloaded files by filename.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import requests

IA_BASE = "https://archive.org"
IA_DOWNLOAD = "https://archive.org/download"

# ---------------------------------------------------------------------------
# Course catalog
# identifiers: one or more Internet Archive item identifiers.
#   Multi-item entries (like Yoruba) are merged into one output directory.
# type: "fsi" | "pc" | "dli"
# ---------------------------------------------------------------------------
CATALOG = {
    # === Primary eval languages ===
    "lin": {
        "name": "Lingala",
        "type": "fsi",
        "identifiers": ["Lingala"],
        "note": "FSI Lingala Basic Course, 24 units ~9h (17 U.S.C. § 105)",
    },
    "yor": {
        "name": "Yoruba",
        "type": "fsi",
        # Yoruba is split across 3 IA items: yrb01-02, yrb03, yrb04 = 279 total
        "identifiers": ["yoruba1A", "yoruba2", "yoruba3"],
        "note": "FSI Yoruba Basic Course, 279 files ~18h (split across 3 IA items)",
    },
    "swh": {
        "name": "Swahili",
        "type": "fsi",
        "identifiers": ["SwahiliBasicCourse"],
        "note": "FSI Swahili Basic Course, 132 files ~10h",
    },
    "amh": {
        "name": "Amharic",
        "type": "fsi",
        "identifiers": ["AmharicBasicCourse"],
        "note": "FSI Amharic Basic Course, 180 files ~14h",
    },
    "hau": {
        "name": "Hausa",
        "type": "fsi",
        "identifiers": ["HausaCourse"],
        "note": "FSI Hausa Basic Course, 10 files ~3h",
    },
    # === Peace Corps languages ===
    "ibo": {
        "name": "Igbo",
        "type": "fsi",
        "identifiers": ["TODO-ibo"],   # verify IA identifier
        "note": "FSI Igbo Language Course; TODO: find IA identifier",
        "skip": True,
    },
    "ewe": {
        "name": "Ewe",
        "type": "pc",
        "identifiers": ["TODO-ewe"],
        "note": (
            "Peace Corps Ewe course, 104 files ~10h. "
            "NOT found on Archive.org as audio. "
            "Manual download may be required: check livelingua.com after logging in, "
            "or contact livelingua directly."
        ),
        "skip": True,
    },
    "wol": {
        "name": "Wolof",
        "type": "pc",
        "identifiers": ["TODO-wol"],
        "note": "Peace Corps Wolof; TODO: find IA identifier",
        "skip": True,
    },
    # === DLI ===
    "hat": {
        "name": "Haitian Creole",
        "type": "dli",
        "identifiers": ["TODO-hat"],
        "note": (
            "DLI Haitian Creole, 59 files. "
            "Only fragmented IA items found (Tape04Vol01Less0203 etc.). "
            "TODO: enumerate all parts or find consolidated upload."
        ),
        "skip": True,
    },
}

# Languages to run first (have confirmed IA identifiers)
PRIMARY = ["lin", "yor", "swh", "amh", "hau"]

# All languages with confirmed identifiers (no skip flag)
CONFIRMED = [k for k, v in CATALOG.items() if not v.get("skip")]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; research-downloader/1.0; "
        "AfriBase-Speech NLP corpus; contact: see paper)"
    )
}


# ---------------------------------------------------------------------------
# Internet Archive helpers
# ---------------------------------------------------------------------------

def ia_get_mp3s(identifier: str, session: requests.Session) -> list[dict]:
    """Return list of {name, size, url} dicts for all MP3s in an IA item."""
    url = f"{IA_BASE}/metadata/{identifier}"
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logging.warning(f"  IA metadata fetch failed for {identifier}: {exc}")
        return []

    data = resp.json()
    files = data.get("files", [])
    mp3s = []
    for f in files:
        name = f.get("name", "")
        fmt = f.get("format", "")
        if fmt in ("MP3", "VBR MP3", "128Kbps MP3", "64Kbps MP3") or name.lower().endswith(".mp3"):
            mp3s.append({
                "name": name,
                "size": int(f.get("size", 0)),
                "url": f"{IA_DOWNLOAD}/{identifier}/{name}",
            })
    return mp3s


def ia_download_file(url: str, dest: Path, session: requests.Session) -> bool:
    """Download a single file from IA. Returns True on success."""
    try:
        resp = session.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 16):
                f.write(chunk)
        return True
    except Exception as exc:
        logging.warning(f"  Download failed {dest.name}: {exc}")
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Main download logic
# ---------------------------------------------------------------------------

def download_course(
    code: str,
    entry: dict,
    out_root: Path,
    session: requests.Session,
    dry_run: bool = False,
    force: bool = False,
) -> bool:
    if entry.get("skip"):
        logging.warning(f"[{code}] SKIPPED — {entry['note']}")
        return False

    lang_dir = out_root / code
    lang_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n[{code}] {entry['name']} ({entry['type'].upper()})")
    if entry.get("note"):
        logging.info(f"  {entry['note']}")

    identifiers = entry["identifiers"]
    all_files = []
    for ident in identifiers:
        logging.info(f"  Fetching IA metadata: {ident}")
        if not dry_run:
            mp3s = ia_get_mp3s(ident, session)
            logging.info(f"  {ident}: {len(mp3s)} MP3s found")
            all_files.extend(mp3s)

    if dry_run:
        logging.info(f"  DRY RUN: would download from {identifiers}")
        return True

    if not all_files:
        logging.warning(f"  No MP3s found in {identifiers}")
        return False

    # Deduplicate by filename (multi-item may have overlaps)
    seen_names = set()
    unique_files = []
    for f in all_files:
        if f["name"] not in seen_names:
            seen_names.add(f["name"])
            unique_files.append(f)

    logging.info(f"  Total unique MP3s: {len(unique_files)}")

    existing = {p.name for p in lang_dir.glob("*.mp3")}
    to_download = [f for f in unique_files if force or f["name"] not in existing]
    logging.info(f"  Already downloaded: {len(existing)}, to download: {len(to_download)}")

    ok = failed = 0
    for i, f in enumerate(to_download, 1):
        dest = lang_dir / f["name"]
        size_mb = f["size"] / 1048576
        logging.info(f"  [{i}/{len(to_download)}] {f['name']} ({size_mb:.1f}MB)")
        if ia_download_file(f["url"], dest, session):
            ok += 1
        else:
            failed += 1
        time.sleep(0.5)  # polite delay between files

    logging.info(f"  Done: {ok} ok, {failed} failed")
    return failed == 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download FSI / Peace Corps / DLI lesson audio via Internet Archive"
    )
    ap.add_argument(
        "--lang", nargs="+", default=["primary"],
        help=(
            "Language codes to download. "
            "Presets: 'primary' (lin yor swh amh hau — all with confirmed IA IDs), "
            "'all' for everything attempted. "
            "Or specific codes: lin yor swh amh hau"
        ),
    )
    ap.add_argument("--out", default="fsi_audio", help="Output root directory (default: fsi_audio/)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan without downloading")
    ap.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    requested = args.lang
    if requested == ["all"]:
        codes = list(CATALOG)
    elif requested == ["primary"]:
        codes = PRIMARY
    elif requested == ["confirmed"]:
        codes = CONFIRMED
    else:
        codes = requested
        unknown = [c for c in codes if c not in CATALOG]
        if unknown:
            logging.error(f"Unknown language codes: {unknown}")
            logging.error(f"Available: {list(CATALOG)}")
            sys.exit(1)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    logging.info(f"Languages: {codes}")
    logging.info(f"Output: {out_root.resolve()}/")
    if args.dry_run:
        logging.info("DRY RUN — no files will be downloaded")

    session = requests.Session()
    session.headers.update(HEADERS)

    ok = failed = skipped = 0
    for code in codes:
        entry = CATALOG[code]
        if entry.get("skip") and "all" not in requested:
            skipped += 1
            continue
        success = download_course(code, entry, out_root, session, args.dry_run, args.force)
        if success:
            ok += 1
        else:
            failed += 1

    logging.info(f"\n=== Summary: {ok} ok, {failed} failed, {skipped} skipped ===")
    if failed:
        logging.warning("See NOTES above for languages marked SKIPPED or failed.")


if __name__ == "__main__":
    main()
