import argparse
from pathlib import Path
from pprint import pformat
from typing import Any

from dinosam.project import require_paths, resolve_project_path


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", ""}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        key, separator, value = raw_line.strip().partition(":")
        if not separator:
            raise ValueError(f"Invalid config line: {raw_line}")

        if indent == 0:
            if value.strip():
                data[key] = parse_scalar(value)
                current_section = None
            else:
                current_section = {}
                data[key] = current_section
            continue

        if current_section is None:
            raise ValueError(f"Nested key without a section: {raw_line}")
        current_section[key] = parse_scalar(value)

    return data


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        print("[WARN] PyYAML is not installed; using the limited smoke-config parser.")
        return load_simple_yaml(path)

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise TypeError(f"Config must be a mapping: {path}")
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DINOv3 + SAM2 training entrypoint.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to a training config file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config_path = resolve_project_path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    config = load_config(config_path)
    print(f"Loaded config: {config_path}")
    print(pformat(config, sort_dicts=False))
    print()

    paths = config.get("paths", {})
    third_party = config.get("third_party", {})
    required_paths = {
        "data_dir": paths.get("data_dir", "data"),
        "weights_dir": paths.get("weights_dir", "weights"),
        "outputs_dir": paths.get("outputs_dir", "outputs"),
        "DINOv3": third_party.get("dinov3_dir", "third_party/dinov3"),
        "SAM2": third_party.get("sam2_dir", "third_party/sam2"),
    }

    ok = require_paths(required_paths)
    if not ok:
        print()
        print("Smoke check failed. Make sure local folders and submodules exist.")
        return 1

    print()
    print("Smoke train entrypoint is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
