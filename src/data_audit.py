"""Task 1.2 - dataset audit for VisionGuard.

Walks the raw dataset folder and builds a table with ONE ROW PER FILE:
class, file size, width, height, colour mode, format, SHA-256 hash and whether the file decodes.
It then reports: files per class, min/median/max of width/height/file size per class,
unreadable (corrupt) files, and exact duplicates (identical bytes => identical SHA-256).

Usage (from the repository root):
    python src/data_audit.py --root data/raw --out reports
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

# The label of an image is the folder it sits in (first path component under --root).
CLASS_FOLDERS = {"AiArtData": "ai_generated", "RealArt": "real"}
HASH_CHUNK_BYTES = 1024 * 1024  # read files 1 MiB at a time so big files never fill memory


def sha256_of(path: Path) -> str:
    """Content hash of the raw bytes. Two files with the same hash are exact duplicates."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_image(path: Path) -> dict:
    """Open one image and return its properties; `error` is empty when the file decodes fine."""
    try:
        with Image.open(path) as img:
            width, height = img.size
            info = {"width": width, "height": height, "mode": img.mode, "format": img.format}
            # open() only reads the header. load() forces a FULL decode, which is what catches
            # truncated or corrupt pixel data that a header-only check would miss.
            img.load()
        return {**info, "error": ""}
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        return {"width": None, "height": None, "mode": None, "format": None,
                "error": f"{type(exc).__name__}: {exc}"}


def build_table(root: Path) -> pd.DataFrame:
    """Walk every class folder under `root` and return the per-file audit table."""
    rows = []
    for folder, label in CLASS_FOLDERS.items():
        class_dir = root / folder
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        for path in sorted(p for p in class_dir.rglob("*") if p.is_file()):
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "class": label,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
                **inspect_image(path),
                "sha256": sha256_of(path),
            })
    df = pd.DataFrame(rows)
    df["readable"] = df["error"] == ""
    # keep="first": the first file with a given hash is the original, every later one is an extra copy.
    df["is_extra_copy"] = df.duplicated(subset="sha256", keep="first")
    df["duplicate_group_size"] = df.groupby("sha256")["sha256"].transform("size")
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Numbers the report needs, as plain Python types so they can be written to JSON."""
    ok = df[df["readable"]]
    per_class = {}
    for label, group in df.groupby("class"):
        good = group[group["readable"]]
        per_class[label] = {
            "files": len(group),
            "unreadable": int((~group["readable"]).sum()),
            **{f"{col}_{stat}": float(getattr(good[col], stat)())
               for col in ("width", "height", "size_bytes") for stat in ("min", "median", "max")},
        }
    dup_groups = df[df["duplicate_group_size"] > 1].groupby("sha256")
    cross_class = [h for h, g in dup_groups if g["class"].nunique() > 1]
    return {
        "total_files": len(df),
        "per_class": per_class,
        "unreadable_files": df.loc[~df["readable"], ["path", "error"]].to_dict("records"),
        "duplicate_groups": int(dup_groups.ngroups),
        "extra_copies_to_remove": int(df["is_extra_copy"].sum()),
        "duplicate_groups_across_classes": len(cross_class),
        "formats": {k: int(v) for k, v in ok["format"].value_counts().items()},
        "colour_modes": {k: int(v) for k, v in ok["mode"].value_counts().items()},
        "extension_vs_real_format_mismatches": int(
            (ok["extension"].str.lstrip(".").str.replace("jpg", "jpeg") != ok["format"].str.lower()).sum()
        ),
    }


def print_summary(summary: dict) -> None:
    print(f"Total files: {summary['total_files']}")
    print("\nPer class (min / median / max):")
    for label, s in summary["per_class"].items():
        print(f"  {label}: {s['files']} files, {s['unreadable']} unreadable")
        for col in ("width", "height", "size_bytes"):
            print(f"    {col:<11} {s[col + '_min']:>12,.0f} / {s[col + '_median']:>12,.0f} / {s[col + '_max']:>12,.0f}")
    print(f"\nFormats: {summary['formats']}")
    print(f"Colour modes: {summary['colour_modes']}")
    print(f"Extension != real format: {summary['extension_vs_real_format_mismatches']} files")
    print(f"\nUnreadable files: {len(summary['unreadable_files'])}")
    for item in summary["unreadable_files"]:
        print(f"  {item['path']}  ({item['error']})")
    print(f"Duplicate groups: {summary['duplicate_groups']}  "
          f"(extra copies: {summary['extra_copies_to_remove']}, "
          f"groups spanning both classes: {summary['duplicate_groups_across_classes']})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("data/raw"), help="folder holding AiArtData/ and RealArt/")
    parser.add_argument("--out", type=Path, default=Path("reports"), help="where to write the CSV and JSON")
    args = parser.parse_args()

    df = build_table(args.root)
    summary = summarise(df)

    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "dataset_audit.csv", index=False)
    (args.out / "dataset_audit_summary.json").write_text(json.dumps(summary, indent=2))
    print_summary(summary)
    print(f"\nWrote {args.out / 'dataset_audit.csv'} and {args.out / 'dataset_audit_summary.json'}")


if __name__ == "__main__":
    main()
