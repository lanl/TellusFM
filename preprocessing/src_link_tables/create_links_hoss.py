#!/usr/bin/env python3
"""Create an HDF5 external-link table for HOSS simulation files.

Each HOSS simulation is a file named ``frac_pull_z_<id>.h5`` that contains the datasets
``model <id> fracture initial``, ``model <id> fracture final`` and ``model <id> break time``.
The link table has one entry ``link_frac_pull_z_<id>`` per file, pointing at the file with a
path relative to the link table, so the data directory and the table can be moved together.

Example
-------
python preprocessing/src_link_tables/create_links_hoss.py \
    --source-dir ../data/hoss \
    --output data/hoss/external_links/validation_link.h5
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import h5py

FILE_PATTERN = re.compile(r"^frac_pull_z_(\d+)\.h5$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an external-link table for HOSS simulations.")
    parser.add_argument("--source-dir", type=Path, required=True,
                        help="Directory containing frac_pull_z_<id>.h5 files")
    parser.add_argument("--output", type=Path, required=True, help="Link table to write (.h5)")
    parser.add_argument("--ids-file", type=Path, default=None,
                        help="Optional text file with one file name (or id) per line; "
                             "only these simulations are linked")
    parser.add_argument("--check", action="store_true",
                        help="Open every linked simulation and check the expected datasets exist")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created")
    return parser.parse_args()


def select_files(source_dir: Path, ids_file: Path | None) -> list[tuple[int, Path]]:
    files = {}
    for path in sorted(source_dir.glob("frac_pull_z_*.h5")):
        match = FILE_PATTERN.match(path.name)
        if match:
            files[int(match.group(1))] = path

    if ids_file is not None:
        wanted = []
        for line in ids_file.read_text().split():
            match = re.search(r"(\d+)", line)
            if match:
                wanted.append(int(match.group(1)))
        missing = [i for i in wanted if i not in files]
        if missing:
            raise SystemExit(f"{len(missing)} ids from {ids_file} not found in {source_dir}: {missing[:5]}...")
        return [(i, files[i]) for i in wanted]

    return sorted(files.items())


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    selected = select_files(source_dir, args.ids_file)
    if not selected:
        raise SystemExit(f"No frac_pull_z_<id>.h5 files found in {source_dir}")

    if args.check:
        for sim_id, path in selected:
            with h5py.File(path, "r") as f:
                for name in (f"model {sim_id} fracture initial", f"model {sim_id} fracture final",
                             f"model {sim_id} break time"):
                    if name not in f:
                        raise SystemExit(f"{path} is missing dataset '{name}'")

    action = "Would create" if args.dry_run else "Created"
    if not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as table:
            for sim_id, path in selected:
                target = os.path.relpath(path, start=output.parent)
                table[f"link_frac_pull_z_{sim_id}"] = h5py.ExternalLink(target, "/")
    print(f"{action} {output} with {len(selected)} links.")


if __name__ == "__main__":
    main()
