#!/usr/bin/env python3
"""Create one HDF5 external-link table for each phase-field train/validation split."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "phase_field"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "phase_field" / "external_links"
SPLIT_SUFFIXES = {"train": "_train", "validation": "_validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one external-link table for the phase-field training data and one "
            "external-link table for the phase-field validation data."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Phase-field data root (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Directory for link tables (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the tables that would be created without writing them.",
    )
    return parser.parse_args()


def collect_split_links(source_dir: Path, output_dir: Path) -> dict[str, list[tuple[str, str]]]:
    link_tables: dict[str, list[tuple[str, str]]] = {"train": [], "validation": []}
    link_names: set[str] = set()

    for child in sorted(source_dir.iterdir()):
        if not child.is_dir():
            continue

        folder_name = child.name
        split = None
        material = None
        for candidate_split, suffix in SPLIT_SUFFIXES.items():
            if folder_name.endswith(suffix):
                split = candidate_split
                material = folder_name[: -len(suffix)]
                break
        if split is None or not material:
            continue

        for source_file in sorted(child.glob("*.h5")):
            stem_parts = source_file.stem.split("_")
            if len(stem_parts) < 4 or stem_parts[0] != "frac" or stem_parts[1] != "pull":
                raise ValueError(
                    f"Filename does not match expected pattern 'frac_pull_<orientation>_<id>.h5': {source_file}"
                )
            orientation = stem_parts[2]
            if orientation == "xz":
                bc_type = "combined"
            elif orientation == "z":
                bc_type = "vertical"
            else:
                raise ValueError(f"Unsupported orientation for BC type mapping: {orientation}")

            sample = stem_parts[-1]
            if not sample.isdigit():
                raise ValueError(f"Filename does not end in a numeric sample ID: {source_file}")

            link_name = f"link_{material}_{bc_type}_{orientation}_{sample}"
            if link_name in link_names:
                raise ValueError(f"Duplicate link name {link_name!r} for file {source_file}")
            link_names.add(link_name)
            target = os.path.relpath(source_file.resolve(), start=output_dir.resolve())
            link_tables[split].append((link_name, target))

    return link_tables


def write_link_table(archive_path: Path, links: list[tuple[str, str]], dry_run: bool = False) -> int:
    if dry_run:
        return len(links)

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(archive_path, mode="w") as link_table:
        for link_name, target in links:
            link_table[link_name] = h5py.ExternalLink(target, "/")
    return len(links)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    link_tables = collect_split_links(source_dir, output_dir)
    if not link_tables["train"] and not link_tables["validation"]:
        raise SystemExit(f"No train or validation .h5 files found below: {source_dir}")

    action = "Would create" if args.dry_run else "Created"

    train_archive = output_dir / "train_link.h5"
    validation_archive = output_dir / "validation_link.h5"

    train_count = write_link_table(train_archive, link_tables["train"], dry_run=args.dry_run)
    print(f"{action} {train_archive} with {train_count} links.")

    validation_count = write_link_table(validation_archive, link_tables["validation"], dry_run=args.dry_run)
    print(f"{action} {validation_archive} with {validation_count} links.")

    print(f"{action} 2 link tables containing {train_count + validation_count} links total.")


if __name__ == "__main__":
    main()
