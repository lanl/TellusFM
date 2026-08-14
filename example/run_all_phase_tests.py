#!/usr/bin/env python3
"""Run every phase-field HDF5 test-link set and collect MAE summaries.
# Run only high and low density tests (either pattern, OR semantics)
python example/run_all_phase_tests.py --pattern "lowDensity|highDensity" --dry-run
# Or supply patterns separately (equivalent)
python example/run_all_phase_tests.py --pattern "lowDensity" --pattern "highDensity" --dry-run
# Run only ortho and curved tests
python example/run_all_phase_tests.py --pattern "ortho|curved"
# Run with verbose and save outputs
python example/run_all_phase_tests.py --pattern "ortho|curved" --verbose
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import h5py
import yaml
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any




SKIP_FILENAMES = {"train_link.h5", "test_link.h5", "validation_link.h5"}
PATH_KEYS_BY_SECTION = {
    "model_params": ("path_working_directory",),
    "datasets": ("path_phase", "path_val_phase", "path_hoss", "path_val_hoss", "path_test"),
    "embeddings": ("emb_rule_based", "emb_phase", "emb_hoss"),
    "checkpoints": ("load_model_num",),
}
SUMMARY_FIELDS = (
    "test_name",
    "h5_path",
    "status",
    "exit_code",
    "samples",
    "mae",
    "mse",
    "ssim",
    "loss_target",
    "duration_seconds",
    "run_dir",
    "log_file",
)


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def default_links_dir(repo_root: Path) -> Path:
    requested_layout = repo_root / "data" / "external_links" 
    phase_layout = repo_root / "data" / "phase_field" / "external_links"
    if requested_layout.exists():
        return requested_layout
    return phase_layout


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(
        description="Run all phase-field test HDF5 link files and summarize MAE values."
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=repo_root / "example" / "test_phase_field.yaml",
        help="Template config to use for each test run.",
    )
    parser.add_argument(
        "--links-dir",
        type=Path,
        default=default_links_dir(repo_root),
        help="Directory containing HDF5 external-link test sets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "example" / "test_runs" / timestamp,
        help="Directory where per-test folders and summary files will be written.",
    )
    parser.add_argument(
        "--main-script",
        type=Path,
        default=repo_root / "example" / "main.py",
        help="TellusFM entrypoint to run for each generated config.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch each test run.",
    )
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Additional HDF5 filename to skip. May be supplied more than once.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N discovered test files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the discovered runs without launching tests.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the sweep after the first failed test run.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Pass --verbose through to example/main.py.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help=(
            "Regex pattern to INCLUDE matching HDF5 filenames. "
            "May be supplied more than once; runs matching any pattern are kept."
        ),
    )
    parser.add_argument(
        "--test-bc",
        choices=["auto", "vertical", "horizontal", "combined", "all"],
        default=None,
        help="PHASE test BC/orientation to run for every generated config.",
    )
    return parser.parse_args()


def normalize_path(value: Any, base_dir: Path) -> Any:
    if value is None or not isinstance(value, (str, Path)):
        return value

    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def load_base_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")

    base_dir = config_path.resolve().parent
    for section_name, keys in PATH_KEYS_BY_SECTION.items():
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in keys:
            if key in section:
                section[key] = normalize_path(section[key], base_dir)

    return config


def _hdf5_key_count(path: Path) -> int:
    with h5py.File(path, "r") as hdf5_file:
        return len(hdf5_file.keys())


def discover_test_links(links_dir: Path, skip_names: set[str]) -> list[Path]:
    if not links_dir.is_dir():
        raise FileNotFoundError(f"HDF5 link directory not found: {links_dir}")

    test_links = sorted(
        path.resolve()
        for path in links_dir.glob("*.h5")
        if path.name not in skip_names
    )

    if not test_links:
        raise FileNotFoundError(f"No HDF5 test links found in {links_dir}")

    return test_links


def filter_empty_links(test_links: list[Path]) -> list[Path]:
    non_empty_links = []
    empty_links = []

    for test_link in test_links:
        if _hdf5_key_count(test_link) == 0:
            empty_links.append(test_link.name)
        else:
            non_empty_links.append(test_link)

    if empty_links:
        print(
            "Skipping empty HDF5 test link(s): "
            + ", ".join(empty_links)
        )

    return non_empty_links


def make_run_config(
    base_config: dict[str, Any],
    test_link: Path,
    run_dir: Path,
    test_bc: str | None = None,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config.setdefault("model_params", {})
    config.setdefault("datasets", {})
    config.setdefault("checkpoints", {})
    config.setdefault("embeddings", {})

    config["model_params"]["sim_type"] = "PHASE"
    config["model_params"]["sim_weights"] = {"PHASE": 1.0}
    config["model_params"]["path_working_directory"] = str(run_dir.resolve())
    if test_bc is not None:
        config["model_params"]["test_bc"] = test_bc

    config["datasets"]["path_phase"] = None
    config["datasets"]["path_val_phase"] = None
    config["datasets"]["path_test"] = str(test_link.resolve())

    config["checkpoints"]["run_type"] = "test"
    return config


def write_yaml(path: Path, config: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)


def parse_test_summary(log_path: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Test summary:\s*(\{.*?\})", text)
    if not matches:
        return {}

    try:
        summary = ast.literal_eval(matches[-1])
    except (SyntaxError, ValueError):
        return {}

    return summary if isinstance(summary, dict) else {}


def write_summaries(output_dir: Path, results: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field) for field in SUMMARY_FIELDS})

    json_path = output_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)


def run_one_test(
    args: argparse.Namespace,
    base_config: dict[str, Any],
    test_link: Path,
    output_dir: Path,
) -> dict[str, Any]:
    test_name = test_link.stem
    run_dir = output_dir / test_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.yaml"
    log_path = run_dir / "run.log"
    tellusfm_log_stem = run_dir / "tellusfm"
    run_config = make_run_config(base_config, test_link, run_dir, args.test_bc)
    write_yaml(config_path, run_config)

    command = [
        args.python,
        str(args.main_script.resolve()),
        "--config",
        str(config_path),
        "--log_filename",
        str(tellusfm_log_stem),
    ]
    if args.verbose:
        command.append("--verbose")

    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {' '.join(command)}\n")
        log_file.write(f"Test link: {test_link}\n")
        log_file.write(f"Run directory: {run_dir}\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=repo_root_from_script(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    duration = time.monotonic() - started
    parsed_summary = parse_test_summary(log_path)
    status = "passed" if completed.returncode == 0 else "failed"

    result = {
        "test_name": test_name,
        "h5_path": str(test_link),
        "status": status,
        "exit_code": completed.returncode,
        "duration_seconds": round(duration, 3),
        "run_dir": str(run_dir),
        "log_file": str(log_path),
        "config_file": str(config_path),
        "figures_dir": str(run_dir / "test_samples"),
        "tellusfm_log": str(tellusfm_log_stem) + ".log",
    }
    result.update(parsed_summary)

    with (run_dir / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)

    return result


def print_result(result: dict[str, Any]) -> None:
    mae = result.get("mae")
    mse = result.get("mse")
    ssim = result.get("ssim")
    mae_text = "n/a" if mae is None else f"{mae:.3g}"
    mse_text = "n/a" if mse is None else f"{mse:.4g}"
    ssim_text = "n/a" if ssim is None else f"{ssim:.3g}"
    print(
        f"{result['status'].upper()}: {result['test_name']} "
        f"mae={mae_text} mse={mse_text} ssim={ssim_text}"
    )


def main() -> int:
    args = parse_args()
    args.base_config = args.base_config.resolve()
    args.links_dir = args.links_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.main_script = args.main_script.resolve()

    skip_names = SKIP_FILENAMES | set(args.skip)
    test_links = discover_test_links(args.links_dir, skip_names)

    # If include patterns were provided, filter the discovered links to only those
    # whose filename matches at least one provided regex pattern.
    if getattr(args, "pattern", None):
        compiled = [re.compile(p) for p in args.pattern]
        filtered = [p for p in test_links if any(c.search(p.name) for c in compiled)]
        print(f"Filtering runs with patterns: {args.pattern}")
        print(f"Runs before: {len(test_links)}, after filter: {len(filtered)}")
        test_links = filtered
        if not test_links:
            raise FileNotFoundError(
                f"No HDF5 test links remain after applying patterns: {args.pattern}"
            )

    test_links = filter_empty_links(test_links)
    if not test_links:
        pattern_text = f" after applying patterns: {args.pattern}" if args.pattern else ""
        raise FileNotFoundError(f"No non-empty HDF5 test links remain{pattern_text}")

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be greater than zero")
        test_links = test_links[:args.limit]

    print(f"Base config: {args.base_config}")
    print(f"Links dir:   {args.links_dir}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Skipping:    {', '.join(sorted(skip_names))}")
    print(f"Test BC:     {args.test_bc or 'config/default'}")
    print(f"Runs found:  {len(test_links)}")

    if args.dry_run:
        for test_link in test_links:
            print(f"DRY RUN: {test_link.name}")
        return 0

    base_config = load_base_config(args.base_config)
    results: list[dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, test_link in enumerate(test_links, start=1):
        print(f"\n[{index}/{len(test_links)}] Running {test_link.name}")
        result = run_one_test(args, base_config, test_link, args.output_dir)
        results.append(result)
        write_summaries(args.output_dir, results)
        print_result(result)

        if args.stop_on_error and result["status"] != "passed":
            print("Stopping after first failed run because --stop-on-error was set.")
            return result["exit_code"] or 1

    failed = [result for result in results if result["status"] != "passed"]
    print(f"\nWrote summary: {args.output_dir / 'summary.csv'}")
    print(f"\nWrote summary: {args.output_dir / 'summary.json'}")

    if failed:
        print(f"Completed with {len(failed)} failed run(s).")
        return 1

    print("Completed all test runs successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
