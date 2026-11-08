"""Draft manifest + confound/leakage report for the Zenodo leak-audio dataset (read-only).

Usage:
    python scripts/build_manifest_draft.py data/raw/zenodo_leak_audio

Writes:
    data/processed/manifest_draft.csv     (one row per audio file)
    outputs/manifest_summary.txt          (what to send back)

Nothing is trained, moved or modified. Filename fields are PARSED, not trusted:
the summary shows where parsing fails and where duplicate audio carries conflicting metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import wave
from pathlib import Path

import pandas as pd

try:
    import soundfile as sf
except ImportError:  # pragma: no cover
    sf = None

# Folder -> label comes from the Zenodo record's folder structure (to be confirmed by you).
def label_from_path(rel: Path) -> str:
    top = rel.parts[0].lower()
    if top.startswith("no leak"):
        return "no_leak"
    if top.startswith("environmental"):
        return "noise"
    if top.startswith("leak"):
        return "leak"
    return "unknown"


# leak / no_leak: material-zone-pressure-flow-device[-_]a-b[_copy]
RE_PIPE = re.compile(
    r"^(?P<material>[^-]+)-(?P<zone>[^-]+)-(?P<pressure>[^-]+)-(?P<flow>[^-]+)-"
    r"(?P<device>[^-_]+)[-_](?P<a>\d+)-(?P<b>\d+)(?:_(?P<copy>\d+))?$"
)
# noise: category-region-device[-_]a-b[_copy]
RE_NOISE = re.compile(
    r"^(?P<category>[^-]+)-(?P<zone>[^-]+)-(?P<device>[^-_]+)[-_](?P<a>\d+)-(?P<b>\d+)"
    r"(?:_(?P<copy>\d+))?$"
)


def num(s: str | None) -> float | None:
    if not s or s.upper() == "NA":
        return None
    m = re.match(r"^\s*([0-9.]+)", s)
    return float(m.group(1)) if m else None


def parse_name(stem: str, label: str) -> dict:
    m = (RE_NOISE if label == "noise" else RE_PIPE).match(stem)
    if not m:
        return {"parse_ok": False}
    g = m.groupdict()
    d = {"parse_ok": True, "zone": g["zone"], "device": g["device"],
         "seg_a": int(g["a"]), "seg_b": int(g["b"]),
         "copy_idx": int(g["copy"]) if g["copy"] else 0}
    if label == "noise":
        d["noise_category"] = g["category"].split("_")[0]
        d["material"] = None
        d["pressure_mpa"] = None
        d["flow"] = None
        rec = f"{label}|{d['noise_category']}|{g['zone']}|{g['device']}"
    else:
        d["noise_category"] = None
        d["material"] = g["material"]
        d["pressure_mpa"] = num(g["pressure"])
        d["flow"] = num(g["flow"])
        rec = f"{label}|{g['material']}|{g['zone']}|{g['pressure']}|{g['flow']}|{g['device']}"
    d["recording_key"] = rec  # filename minus segment/copy suffix: HYPOTHESIS for 'same recording'
    return d


def audio_hdr(path: Path) -> tuple[int | None, float | None]:
    try:
        if sf is not None:
            i = sf.info(str(path))
            return i.samplerate, i.duration
        with wave.open(str(path), "rb") as w:
            return w.getframerate(), w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001
        return None, None


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_draft.csv"))
    ap.add_argument("--summary", type=Path, default=Path("outputs/manifest_summary.txt"))
    a = ap.parse_args()
    root = a.root.resolve()

    rows = []
    for p in sorted(root.rglob("*.wav")):
        rel = p.relative_to(root)
        label = label_from_path(rel)
        sr, dur = audio_hdr(p)
        row = {"path": rel.as_posix(), "label": label, "sr": sr, "dur_s": dur,
               "md5": md5(p), **parse_name(p.stem, label)}
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No .wav files found.")

    # union-find: link files sharing a recording_key or identical bytes
    parent = list(range(len(df)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for col in ("recording_key", "md5"):
        for _, idx in df.groupby(col).groups.items():
            idx = list(idx)
            for j in idx[1:]:
                union(idx[0], j)
    df["group_id"] = [find(i) for i in range(len(df))]
    df["dup_group_size"] = df.groupby("md5")["md5"].transform("size")

    a.manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.manifest, index=False)

    out: list[str] = []
    P = out.append
    P(f"FILES: {len(df)}   (manifest: {a.manifest.resolve()})")
    P("\n== label counts (from top-level folder) ==")
    P(df["label"].value_counts().to_string())
    P(f"\n== filename parse failures: {(~df['parse_ok'].fillna(False).astype(bool)).sum()} ==")
    for pth in df.loc[~df["parse_ok"].fillna(False).astype(bool), "path"].head(15):
        P("  " + pth)

    ok = df[df["parse_ok"].fillna(False).astype(bool)]
    pipe = ok[ok["label"] != "noise"]
    P("\n== CONFOUND CHECK: label x device ==")
    P(pd.crosstab(ok["label"], ok["device"]).to_string())
    P("\n== label x material (leak / no_leak only) ==")
    P(pd.crosstab(pipe["label"], pipe["material"].fillna("NA")).to_string())
    P("\n== label x zone ==")
    P(pd.crosstab(ok["label"], ok["zone"]).to_string())
    P("\n== label x device x material (leak / no_leak only) ==")
    P(pd.crosstab([pipe["label"], pipe["device"]], pipe["material"].fillna("NA")).to_string())
    P("\n== missing pressure/flow (NA) per label ==")
    P(pipe.groupby("label")[["pressure_mpa", "flow"]].agg(lambda s: int(s.isna().sum())).to_string())
    P("\n== noise categories ==")
    P(ok[ok["label"] == "noise"]["noise_category"].value_counts().to_string())

    P("\n== GROUPS (hypothesis: same recording_key or identical bytes = same group) ==")
    gs = df.groupby("group_id").agg(n=("path", "size"), labels=("label", "nunique"), label=("label", "first"))
    P(f"groups: {len(gs)}   groups spanning >1 label: {(gs['labels'] > 1).sum()}")
    P("groups per label:\n" + gs.groupby("label").size().to_string())
    P("group size per label (min / median / max):")
    P(gs.groupby("label")["n"].agg(["min", "median", "max"]).to_string())
    P("largest groups share of its label's files:")
    for lab, sub in gs.groupby("label"):
        tot = sub["n"].sum()
        top = sub["n"].sort_values(ascending=False).head(3).tolist()
        P(f"  {lab}: top-3 group sizes {top} of {tot} files")

    P("\n== DUPLICATE AUDIO ==")
    dup = df[df["dup_group_size"] > 1]
    P(f"files in byte-identical groups: {len(dup)} in {dup['md5'].nunique()} groups")
    conflict = 0
    for _, g in dup.groupby("md5"):
        if g["recording_key"].nunique() > 1:
            conflict += 1
    P(f"duplicate groups whose filenames describe DIFFERENT recordings (metadata conflict): {conflict}")

    P("\n== segment tokens (a-b) ==")
    P(ok["seg_a"].astype(int).astype(str).add("-").add(ok["seg_b"].astype(int).astype(str)).value_counts().head(15).to_string())

    text = "\n".join(out)
    print(text)
    a.summary.parent.mkdir(parents=True, exist_ok=True)
    a.summary.write_text(text, encoding="utf-8")
    print(f"\n[saved to {a.summary.resolve()}]")


if __name__ == "__main__":
    main()
