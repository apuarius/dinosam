import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dinosam.data.instance_tiles import (  # noqa: E402
    list_instance_tile_pairs,
    load_rgb_image,
    resolve_instance_dataset_root,
)
from dinosam.features import (  # noqa: E402
    boundary_map_from_feature_grid,
    extract_patch_feature_grid,
    image_to_sat493m_tensor,
    resize_float_map,
)
from dinosam.models import DINOv3Wrapper, build_dinov3_config  # noqa: E402
from dinosam.project import resolve_project_path  # noqa: E402
from dinosam.train import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建 DINOv3 边界热图导出脚本的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Export DINOv3 feature-distance boundary maps for instance tiles."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/sam2-dataset-1024",
        help="Dataset root containing Image/Instance or an All/Image + All/Instance layout.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/dinov3_sam2.yaml",
        help="Model config containing DINOv3 loading settings.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/dinov3_boundary_maps",
        help="Directory used to save boundary maps and overlays.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of image tiles to process.",
    )
    parser.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.55,
        help="Maximum red overlay opacity used for boundary visualization.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the low-resolution patch boundary map as .npy.",
    )
    return parser


def require_file(path: Path, label: str) -> None:
    """检查必需文件是否存在，缺失时给出明确错误。"""
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def prepare_dinov3_input(dinov3: DINOv3Wrapper, image: np.ndarray) -> Any:
    """根据 DINOv3 加载方式准备模型输入。"""
    load_from = dinov3.config.load_from.lower()
    if load_from == "huggingface":
        return dinov3.prepare_inputs(Image.fromarray(image))
    if load_from == "torch_hub":
        return image_to_sat493m_tensor(image, dinov3.config.device)
    raise ValueError(f"Unsupported DINOv3 load_from value: {dinov3.config.load_from}")


def save_gray_map(value: np.ndarray, output_path: Path) -> None:
    """把 0 到 1 的热图保存为灰度 PNG。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray((np.clip(value, 0.0, 1.0) * 255).astype(np.uint8), mode="L")
    image.save(output_path)


def save_boundary_overlay(
    image: np.ndarray,
    boundary: np.ndarray,
    output_path: Path,
    alpha: float,
) -> None:
    """把边界热图以红色半透明形式叠加到原始 RGB 图上。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    mask_alpha = np.clip(boundary[..., None] * alpha, 0.0, 1.0)
    composite = base * (1.0 - mask_alpha) + red * mask_alpha
    Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8)).save(output_path)


def main() -> int:
    """加载 DINOv3 并导出切片的特征距离边界热图。"""
    args = build_parser().parse_args()

    dataset_root = resolve_instance_dataset_root(args.dataset_root)
    model_config_path = resolve_project_path(args.model_config)
    output_dir = resolve_project_path(args.output_dir)
    require_file(model_config_path, "Model config")

    model_config = load_config(model_config_path)
    dinov3_config = build_dinov3_config(model_config)
    pairs = list_instance_tile_pairs(dataset_root)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "boundary_maps.jsonl"

    print(f"Dataset root: {dataset_root}")
    print(f"Tiles to process: {len(pairs)}")
    print("Loading DINOv3...")
    dinov3 = DINOv3Wrapper.from_config(dinov3_config)

    import torch

    with metadata_path.open("w", encoding="utf-8") as handle:
        for index, pair in enumerate(pairs):
            image = load_rgb_image(pair.image_path)
            model_input = prepare_dinov3_input(dinov3, image)

            with torch.inference_mode():
                raw_features = dinov3(model_input).raw

            feature_grid = extract_patch_feature_grid(raw_features)
            patch_boundary = boundary_map_from_feature_grid(feature_grid)
            image_boundary = resize_float_map(
                patch_boundary,
                size=(image.shape[1], image.shape[0]),
            )

            stem = pair.image_path.stem
            gray_path = output_dir / "gray" / f"{stem}.png"
            overlay_path = output_dir / "overlay" / f"{stem}.png"
            save_gray_map(image_boundary, gray_path)
            save_boundary_overlay(image, image_boundary, overlay_path, alpha=args.overlay_alpha)

            npy_path = None
            if args.save_npy:
                npy_path = output_dir / "npy" / f"{stem}.npy"
                npy_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(npy_path, patch_boundary)

            row = {
                "image": pair.name,
                "feature_grid_shape": list(feature_grid.shape),
                "patch_boundary_shape": list(patch_boundary.shape),
                "gray_path": str(gray_path),
                "overlay_path": str(overlay_path),
                "npy_path": str(npy_path) if npy_path else None,
                "boundary_min": float(patch_boundary.min()),
                "boundary_max": float(patch_boundary.max()),
                "boundary_mean": float(patch_boundary.mean()),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{index + 1}/{len(pairs)}] {pair.name}: "
                f"feature_grid={tuple(feature_grid.shape)}"
            )

    print(f"Saved metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
