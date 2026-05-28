import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from dinosam.models import DINOv3Wrapper, build_dinov3_config  # noqa: E402
from dinosam.project import resolve_project_path  # noqa: E402
from dinosam.train import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建 DINOv3 单图 smoke test 的命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Run a one-image DINOv3 smoke test.")
    parser.add_argument(
        "--model-config",
        default="configs/model/dinov3_sam2.yaml",
        help="Model config containing the DINOv3 loading settings.",
    )
    return parser


def require_file(path: Path, label: str) -> None:
    """检查必须文件是否存在，缺失时给出明确的错误位置。"""
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def require_dir(path: Path, label: str) -> None:
    """检查必须目录是否存在，缺失时给出明确的错误位置。"""
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory: {path}")


def require_hf_model_dir(path: Path) -> None:
    """检查 Hugging Face 本地模型目录是否包含 DINOv3 加载所需文件。"""
    require_dir(path, "DINOv3 Hugging Face model directory")
    for filename in ("config.json", "model.safetensors", "preprocessor_config.json"):
        require_file(path / filename, f"DINOv3 Hugging Face {filename}")


def make_synthetic_satellite_image(size: int = 224) -> np.ndarray:
    """生成一张简化卫星风格 RGB 图，用于验证 DINOv3 前向流程。"""
    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size, 3), dtype=np.uint8)

    image[..., 0] = 70 + (xx % 48)
    image[..., 1] = 95 + (yy % 64)
    image[..., 2] = 55 + ((xx + yy) % 40)

    image[32:96, 36:160] = (80, 130, 80)
    image[118:196, 50:188] = (150, 135, 95)
    image[104:118, :] = (55, 85, 120)
    image[:, 108:122] = (60, 90, 125)
    return image


def normalize_sat493m(image: np.ndarray) -> torch.Tensor:
    """按 DINOv3 SAT-493M 参数把 RGB 图转换成 torch.hub 路线的输入张量。"""
    mean = torch.tensor([0.430, 0.411, 0.296]).view(3, 1, 1)
    std = torch.tensor([0.213, 0.156, 0.143]).view(3, 1, 1)

    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    tensor = (tensor - mean) / std
    return tensor.unsqueeze(0)


def describe_features(value: Any, prefix: str = "features") -> None:
    """递归打印 DINOv3 输出对象中的张量 shape，方便人工确认结果。"""
    if torch.is_tensor(value):
        print(f"{prefix}: shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device}")
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            describe_features(item, f"{prefix}.{key}")
        return

    if hasattr(value, "to_tuple"):
        for index, item in enumerate(value.to_tuple()):
            describe_features(item, f"{prefix}[{index}]")
        return

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            describe_features(item, f"{prefix}[{index}]")
        return

    print(f"{prefix}: {type(value).__name__}")


def print_dinov3_source(dinov3_config: Any) -> None:
    """根据加载方式打印当前 DINOv3 smoke test 将读取的模型来源。"""
    load_from = dinov3_config.load_from.lower()
    print(f"DINOv3 load_from: {load_from}")

    if load_from == "huggingface":
        if dinov3_config.hf_model_dir is None:
            raise ValueError("DINOv3 hf_model_dir is not configured.")
        model_dir = resolve_project_path(dinov3_config.hf_model_dir)
        require_hf_model_dir(model_dir)
        print(f"Loading DINOv3 Hugging Face model from: {model_dir}")
        return

    if load_from == "torch_hub":
        if dinov3_config.weights is None:
            raise ValueError("DINOv3 weights are not configured.")
        weight_path = resolve_project_path(dinov3_config.weights)
        require_file(weight_path, "DINOv3 .pth weights")
        print(f"Loading DINOv3 torch.hub model from: {weight_path}")
        return

    raise ValueError(f"Unsupported DINOv3 load_from value: {dinov3_config.load_from}")


def main() -> int:
    """加载 DINOv3 卫星权重，并对一张合成图执行一次特征提取。"""
    args = build_parser().parse_args()

    model_config_path = resolve_project_path(args.model_config)
    require_file(model_config_path, "Model config")

    model_config = load_config(model_config_path)
    dinov3_config = build_dinov3_config(model_config)
    print_dinov3_source(dinov3_config)

    dinov3 = DINOv3Wrapper.from_config(dinov3_config)
    image = make_synthetic_satellite_image()

    if dinov3_config.load_from.lower() == "huggingface":
        inputs = Image.fromarray(image)
        print(f"Input image: size={inputs.size}, mode={inputs.mode}")
    else:
        inputs = normalize_sat493m(image).to(dinov3_config.device)
        print(f"Input tensor: shape={tuple(inputs.shape)}, dtype={inputs.dtype}, device={inputs.device}")

    with torch.inference_mode():
        features = dinov3(inputs)

    describe_features(features.raw)
    print("DINOv3 smoke test is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
