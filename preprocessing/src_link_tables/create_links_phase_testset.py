#!/usr/bin/env python3
"""Create one HDF5 external-link table for every phase-field test folder."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import h5py


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "phase_field" / "test_set" / "PHASE-FIELD"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "phase_field" / "external_links"

MATERIAL_NAMES = {"aluminum": "al"}
FRACTURE_NAMES = {"dense": "highDensity", "sparse": "lowDensity"}
BOUNDARY_NAMES = {
    "axial_bc": ("axial", "vertical", "z"),
    "biaxial_bc": ("biaxial", "combined", "xz"),
}
SAMPLE_ID_PATTERN = re.compile(r"_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one external-link table for each directory below PHASE-FIELD "
            "that directly contains HDF5 files."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Phase-field test-set root (default: {DEFAULT_SOURCE})",
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


def natural_sort_key(path: Path) -> list[str | int]:
    """Sort numeric filename components numerically."""
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def source_folders(source_dir: Path) -> list[Path]:
    """Return only directories that directly contain one or more .h5 files."""
    folders = {path.parent for path in source_dir.rglob("*.h5") if path.is_file()}
    return sorted(folders, key=lambda path: path.relative_to(source_dir).parts)


def folder_metadata(folder: Path, source_dir: Path) -> tuple[str, str, str, str]:
    """Return material, fracture type, table BC, and link-key BC for a source folder."""
    relative_parts = folder.relative_to(source_dir).parts
    if len(relative_parts) < 4:
        raise ValueError(f"Unexpected test folder layout: {folder}")

    material_dir, boundary_dir = relative_parts[0], relative_parts[1]
    batch_dir = relative_parts[-1]
    if boundary_dir not in BOUNDARY_NAMES:
        raise ValueError(f"Unsupported boundary-condition folder: {folder}")

    fracture_type = batch_dir.split("_", maxsplit=1)[0]
    if not fracture_type:
        raise ValueError(f"Cannot determine fracture type from: {folder}")

    table_bc, orientation, link_bc = BOUNDARY_NAMES[boundary_dir]
    material = MATERIAL_NAMES.get(material_dir, material_dir)
    fracture_name = FRACTURE_NAMES.get(fracture_type, fracture_type)
    return material, fracture_name, table_bc, f"{orientation}_{link_bc}"


def sample_id(source_file: Path) -> str:
    match = SAMPLE_ID_PATTERN.search(source_file.stem)
    if match is None:
        raise ValueError(f"Filename does not end in a numeric sample ID: {source_file}")
    return match.group(1)


def create_link_table(
    folder: Path, source_dir: Path, output_dir: Path, dry_run: bool = False
) -> tuple[Path, int]:
    material, fracture_name, table_bc, link_bc = folder_metadata(folder, source_dir)
    archive = output_dir / f"{material}_{fracture_name}_{table_bc}_link.h5"
    source_files = sorted(folder.glob("*.h5"), key=natural_sort_key)

    link_names: set[str] = set()
    links: list[tuple[str, str]] = []
    for source_file in source_files:
        link_name = f"link_{material}_{link_bc}_{sample_id(source_file)}"
        if link_name in link_names:
            raise ValueError(f"Duplicate link name {link_name!r} in {folder}")
        link_names.add(link_name)
        target = os.path.relpath(source_file.resolve(), start=output_dir.resolve())
        links.append((link_name, target))

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with h5py.File(archive, mode="w") as link_table:
            for link_name, target in links:
                link_table[link_name] = h5py.ExternalLink(target, "/")

    return archive, len(links)


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    folders = source_folders(source_dir)
    if not folders:
        raise SystemExit(f"No .h5 files found below: {source_dir}")

    total_links = 0
    action = "Would create" if args.dry_run else "Created"
    for folder in folders:
        archive, link_count = create_link_table(
            folder, source_dir, output_dir, dry_run=args.dry_run
        )
        total_links += link_count
        print(f"{action} {archive} with {link_count} links from {folder}")

    print(f"{action} {len(folders)} link tables containing {total_links} links total.")


if __name__ == "__main__":
    main()
