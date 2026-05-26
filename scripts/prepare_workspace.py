import argparse
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dinosam.train import load_config  # noqa: E402


def resolve_path(path_value: str | Path) -> Path:
    """把配置中的相对路径转换成基于仓库根目录的绝对路径。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def create_dir(path: Path) -> None:
    """创建目录；如果目录已经存在，则保持原样。"""
    path.mkdir(parents=True, exist_ok=True)
    print(f"[DIR] {path}")


def collect_workspace_dirs(model_config: dict[str, Any]) -> list[Path]:
    """根据模型配置收集需要提前创建的工作目录。"""
    dirs = [
        REPO_ROOT / "data",
        REPO_ROOT / "weights",
        REPO_ROOT / "outputs",
        REPO_ROOT / "outputs" / "runs",
        REPO_ROOT / "outputs" / "predictions",
        REPO_ROOT / "outputs" / "visualizations",
    ]

    for section_name in ("dinov3", "sam2"):
        section = model_config.get(section_name, {})
        if not isinstance(section, dict):
            continue

        for key in ("weights", "checkpoint"):
            value = section.get(key)
            if value:
                dirs.append(resolve_path(value).parent)

    return sorted(set(dirs))


def collect_expected_files(model_config: dict[str, Any]) -> dict[str, Path]:
    """从模型配置中收集后续需要手动放置或下载的权重文件路径。"""
    expected: dict[str, Path] = {}

    dinov3 = model_config.get("dinov3", {})
    if isinstance(dinov3, dict) and dinov3.get("weights"):
        expected["DINOv3 weights"] = resolve_path(dinov3["weights"])

    sam2 = model_config.get("sam2", {})
    if isinstance(sam2, dict) and sam2.get("checkpoint"):
        expected["SAM2 checkpoint"] = resolve_path(sam2["checkpoint"])

    return expected


def print_expected_files(expected_files: dict[str, Path]) -> None:
    """打印权重文件是否已经放在配置指定的位置。"""
    if not expected_files:
        print("No weight files are declared in the model config.")
        return

    print()
    print("Expected weight files:")
    for name, path in expected_files.items():
        status = "OK" if path.exists() else "MISSING"
        print(f"[{status}] {name}: {path}")


def build_parser() -> argparse.ArgumentParser:
    """构建工作区准备脚本的命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Prepare local folders for dinosam runs.")
    parser.add_argument(
        "--model-config",
        default="configs/model/dinov3_sam2.yaml",
        help="Model config used to infer checkpoint folders.",
    )
    return parser


def main() -> int:
    """创建数据、权重和输出目录，并提示配置中声明的权重文件位置。"""
    args = build_parser().parse_args()
    model_config_path = resolve_path(args.model_config)

    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config file does not exist: {model_config_path}")

    model_config = load_config(model_config_path)
    print(f"Preparing workspace from: {model_config_path}")
    print()

    for path in collect_workspace_dirs(model_config):
        create_dir(path)

    print_expected_files(collect_expected_files(model_config))
    print()
    print("Workspace folders are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
