"""Read-only dataset inspector for AquaRadar (Member 1, Step 1).

Usage:
    python scripts/inspect_dataset.py <dataset_root> [--out outputs/dataset_inspection.txt]

It does NOT train, move, rename, or modify anything. It reports facts found on disk.
Labels are NOT assumed: it only shows *candidate* label sources (folder names,
filename prefixes, metadata tables) for you to confirm against the dataset's own documentation.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path

try:
    import soundfile as sf  # optional; used for flac/ogg/wav headers
except ImportError:  # pragma: no cover
    sf = None

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aiff", ".aif", ".opus", ".wma"}
DOC_NAMES = ("readme", "licen", "copying", "citation", "notice", "metadata", "meta", "terms")
TABLE_SUFFIXES = {".csv", ".tsv", ".json", ".xlsx", ".xls"}
DOC_SUFFIXES = {".md", ".txt", ".rst", ".pdf", ".html"}
MAX_DOC_LINES = 60


def audio_info(path: Path) -> dict:
    """Header-only read. Returns sr, channels, frames, duration, subtype, or error."""
    suffix = path.suffix.lower()
    if sf is not None:
        try:
            i = sf.info(str(path))
            return {"sr": i.samplerate, "ch": i.channels, "frames": i.frames,
                    "dur": i.duration, "subtype": i.subtype, "fmt": i.format}
        except Exception as e:  # noqa: BLE001
            sf_err = f"soundfile: {e}"
        else:
            sf_err = ""
    else:
        sf_err = "soundfile not installed"
    if suffix == ".wav":
        try:
            with wave.open(str(path), "rb") as w:
                sr, n = w.getframerate(), w.getnframes()
                return {"sr": sr, "ch": w.getnchannels(), "frames": n,
                        "dur": n / sr if sr else 0.0,
                        "subtype": f"PCM{8 * w.getsampwidth()}", "fmt": "WAV(wave)"}
        except Exception as e:  # noqa: BLE001
            return {"error": f"{sf_err}; wave: {e!r}"}
    return {"error": sf_err or "unreadable"}


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def group_key(stem: str) -> str:
    """Heuristic 'same-source' key: strip trailing numeric chunks like _001, -12, (3)."""
    prev = None
    s = stem
    while prev != s:
        prev = s
        s = re.sub(r"[\s_\-.]+\(?\d+\)?$", "", s)
    return s or stem


def head(path: Path, n: int = MAX_DOC_LINES) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [next(f).rstrip("\n") for _ in range(n)]
    except StopIteration:
        pass
    except Exception as e:  # noqa: BLE001
        return f"(could not read: {e})"
    else:
        return "\n".join(lines) + f"\n... (truncated at {n} lines)"
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def fmt_stats(vals: list[float]) -> str:
    if not vals:
        return "n/a"
    return (f"min={min(vals):.2f} median={statistics.median(vals):.2f} "
            f"mean={statistics.mean(vals):.2f} max={max(vals):.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("outputs/dataset_inspection.txt"))
    args = ap.parse_args()
    root: Path = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    out: list[str] = []
    P = out.append

    all_files = [p for p in root.rglob("*") if p.is_file() and not p.name.startswith(".")]
    P(f"DATASET ROOT: {root}")
    P(f"Total files: {len(all_files)}")
    P(f"Python soundfile available: {sf is not None}")
    P("")

    # 1. File types
    ext_counts = Counter(p.suffix.lower() or "(none)" for p in all_files)
    P("== 1. FILE EXTENSIONS ==")
    for ext, c in ext_counts.most_common():
        P(f"  {ext:10s} {c}")
    P("")

    audio = sorted(p for p in all_files if p.suffix.lower() in AUDIO_SUFFIXES)
    P(f"Audio files found: {len(audio)}")
    P("")

    # 2. Documentation / licence / metadata files (content shown so it can be read, not assumed)
    P("== 2. DOCUMENTATION, LICENCE & METADATA FILES ==")
    docs = [p for p in all_files if p.suffix.lower() in DOC_SUFFIXES | TABLE_SUFFIXES
            and (p.suffix.lower() in TABLE_SUFFIXES or any(k in p.name.lower() for k in DOC_NAMES))]
    if not docs:
        P("  NONE FOUND -> licence/labels/recording conditions are UNDOCUMENTED in this folder.")
    for p in sorted(docs):
        rel = p.relative_to(root)
        P(f"--- {rel}  ({p.stat().st_size} bytes)")
        if p.suffix.lower() in {".md", ".txt", ".rst", ".csv", ".tsv", ".json", ".html"}:
            P(head(p, MAX_DOC_LINES if p.suffix.lower() not in {".csv", ".tsv"} else 8))
        else:
            P("(binary/other format: open manually)")
    P("")

    # 3. Audio headers
    P("== 3. AUDIO HEADERS ==")
    infos: dict[Path, dict] = {}
    for p in audio:
        infos[p] = audio_info(p)
    ok = {p: i for p, i in infos.items() if "error" not in i}
    bad = {p: i for p, i in infos.items() if "error" in i}
    P(f"Readable: {len(ok)}   Unreadable/corrupt: {len(bad)}")
    for p, i in list(bad.items())[:10]:
        P(f"  UNREADABLE {p.relative_to(root)}: {i['error']}")
    if ok:
        P(f"Sample rates : {dict(Counter(i['sr'] for i in ok.values()))}")
        P(f"Channels     : {dict(Counter(i['ch'] for i in ok.values()))}")
        P(f"Subtypes     : {dict(Counter(i['subtype'] for i in ok.values()))}")
        durs = [i["dur"] for i in ok.values()]
        P(f"Duration (s) : {fmt_stats(durs)}")
        P(f"Zero-length  : {sum(1 for d in durs if d <= 0)}")
        P(f"Total hours  : {sum(durs) / 3600:.2f}")
    P("")

    # 4. CANDIDATE label sources (unconfirmed)
    P("== 4. CANDIDATE LABEL SOURCES (UNCONFIRMED - verify against dataset docs) ==")
    P("4a. Files per immediate parent folder:")
    by_parent: dict[str, list[Path]] = defaultdict(list)
    for p in audio:
        by_parent[str(p.parent.relative_to(root)) or "."].append(p)
    for k in sorted(by_parent):
        durs = [infos[p]["dur"] for p in by_parent[k] if p in ok]
        P(f"  {k:50s} n={len(by_parent[k]):6d}  dur(s): {fmt_stats(durs)}")
    P("4b. Files per first filename token (split on _ - space):")
    tok = Counter(re.split(r"[_\-\s.]", p.stem)[0] for p in audio)
    for t, c in tok.most_common(25):
        P(f"  {t:30s} {c}")
    if len(tok) > 25:
        P(f"  ... {len(tok) - 25} more distinct tokens")
    P("")

    # 5. Leakage risks (heuristic evidence only)
    P("== 5. LEAKAGE RISK CHECKS ==")
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in audio:
        by_size[p.stat().st_size].append(p)
    dup_groups = []
    for ps in by_size.values():
        if len(ps) > 1:
            by_hash: dict[str, list[Path]] = defaultdict(list)
            for p in ps:
                by_hash[md5(p)].append(p)
            dup_groups += [g for g in by_hash.values() if len(g) > 1]
    P(f"5a. Byte-identical duplicate groups: {len(dup_groups)} "
      f"(files involved: {sum(len(g) for g in dup_groups)})")
    for g in dup_groups[:10]:
        P("   " + " == ".join(str(p.relative_to(root)) for p in g[:4]))
    cross = [g for g in dup_groups if len({p.parent for p in g}) > 1]
    P(f"    ...of which span different folders (possible cross-class duplicates): {len(cross)}")

    name_dirs: dict[str, set[str]] = defaultdict(set)
    for p in audio:
        name_dirs[p.name].add(str(p.parent.relative_to(root)))
    same_name = {n: d for n, d in name_dirs.items() if len(d) > 1}
    P(f"5b. Same filename in more than one folder: {len(same_name)}")
    for n, d in list(same_name.items())[:5]:
        P(f"   {n}: {sorted(d)}")

    groups: dict[str, list[Path]] = defaultdict(list)
    for p in audio:
        groups[group_key(p.stem)].append(p)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    P(f"5c. Filename-stem groups (numeric suffix stripped): {len(groups)} groups for {len(audio)} files; "
      f"{len(multi)} groups contain >1 file. Largest groups (possible same source/session):")
    for k, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:10]:
        folders = sorted({str(p.parent.relative_to(root)) for p in v})
        P(f"   '{k}': {len(v)} files, folders={folders[:4]}")
    P("   NOTE: 5c is a filename heuristic only. Whether files share a source/session must be "
      "confirmed in the dataset documentation or metadata.")
    P("")
    P("== 6. EXAMPLE PATHS (first 3 audio files per folder) ==")
    for k in sorted(by_parent):
        for p in by_parent[k][:3]:
            P(f"  {p.relative_to(root)}")

    text = "\n".join(out)
    print(text)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"\n[saved to {args.out.resolve()}]")


if __name__ == "__main__":
    main()
