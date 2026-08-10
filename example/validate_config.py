import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = Path(__file__).resolve().parent
CONFIG_MODULE_PATH = REPO_ROOT / "tellusfm" / "read_input_yaml_file.py"


def load_config_module():
    spec = importlib.util.spec_from_file_location("tellusfm_config_loader", CONFIG_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tfm_config = load_config_module()


DEFAULT_CONFIGS = (
    EXAMPLE_DIR / "train_rule_based.yaml",
    EXAMPLE_DIR / "train_phase_field.yaml",
    EXAMPLE_DIR / "test_rule_based.yaml",
    EXAMPLE_DIR / "test_phase_field.yaml",
    EXAMPLE_DIR / "train_mixed.yaml"
)


def resolve_config_path(config_path):
    path = Path(config_path).expanduser()
    if path.is_absolute() or path.exists():
        return path

    example_path = EXAMPLE_DIR / path
    if example_path.exists():
        return example_path

    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Validate TellusFM YAML config files")
    parser.add_argument(
        "configs",
        nargs="*",
        help="Config files to validate. Defaults to the example configs.",
    )
    parser.add_argument(
        "--skip-path-check",
        action="store_true",
        help="Validate structure without requiring dataset and embedding paths to exist.",
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print normalized paths used by each config.",
    )
    args = parser.parse_args(argv)

    config_paths = [resolve_config_path(path) for path in args.configs] if args.configs else DEFAULT_CONFIGS
    check_paths = not args.skip_path_check
    has_error = False

    for config_path in config_paths:
        try:
            config = tfm_config.read_yaml_config(str(config_path))
            summary = tfm_config.validate_config(config, check_paths=check_paths)
        except Exception as exc:
            has_error = True
            print(f"ERROR {config_path}:\n{exc}", file=sys.stderr)
            continue

        sim_types = ", ".join(summary["sim_types"])
        print(f"OK {config_path}: {sim_types}")

        for warning in summary["warnings"]:
            print(f"  WARN {warning}")

        if args.show_paths:
            for label, value in summary["checked_paths"].items():
                print(f"  {label}: {value}")

    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
