import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from dinosam.data.instance_tiles import (  # noqa: E402
    list_instance_tile_pairs,
    load_instance_mask,
    load_rgb_image,
    resolve_instance_dataset_root,
)
from dinosam.evaluation import dilate_binary_mask, label_boundary_map, score_map_metrics  # noqa: E402
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
        "--gt-boundary-dilation",
        type=int,
        default=4,
        help="Dilate GT boundaries before downsampling them to DINOv3 patch resolution.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save the low-resolution patch boundary map as .npy.",
    )
    return parser


def require_file(path: Path, label: str) -> None:
    """检查必须文件是否存在，缺失时给出明确错误。"""
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


def colorize_heatmap(value: np.ndarray) -> np.ndarray:
    """把单通道热图转换成更容易观察的蓝绿黄红伪彩色图。"""
    values = np.clip(value, 0.0, 1.0).astype(np.float32)
    anchors = np.asarray(
        [
            [8, 24, 64],
            [0, 112, 192],
            [72, 184, 112],
            [248, 220, 96],
            [232, 64, 48],
        ],
        dtype=np.float32,
    )
    scaled = values * (len(anchors) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.clip(left + 1, 0, len(anchors) - 1)
    weight = (scaled - left)[..., None]
    colors = anchors[left] * (1.0 - weight) + anchors[right] * weight
    return np.clip(colors, 0, 255).astype(np.uint8)


def save_heat_map(value: np.ndarray, output_path: Path) -> None:
    """把 0 到 1 的边界分数图保存为伪彩色热力图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(colorize_heatmap(value)).save(output_path)


def save_boundary_overlay(
    image: np.ndarray,
    boundary: np.ndarray,
    output_path: Path,
    alpha: float,
) -> None:
    """把 DINOv3 边界热图以红色半透明形式叠加到原始 RGB 图上。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base = image.astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255.0
    mask_alpha = np.clip(boundary[..., None] * alpha, 0.0, 1.0)
    composite = base * (1.0 - mask_alpha) + red * mask_alpha
    Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8)).save(output_path)


def save_gt_boundary_overlay(image: np.ndarray, boundary: np.ndarray, output_path: Path) -> None:
    """把 GT 实例边界以绿色叠加到原始 RGB 图上，方便和 DINOv3 热图对照。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visible_boundary = dilate_binary_mask(boundary, radius=1)
    composite = image.copy()
    composite[visible_boundary] = np.asarray([0, 255, 120], dtype=np.uint8)
    Image.fromarray(composite).save(output_path)


def save_visual_panel(
    image: np.ndarray,
    heat_map: np.ndarray,
    dinov3_overlay: np.ndarray,
    gt_overlay: np.ndarray,
    output_path: Path,
) -> None:
    """保存原图、DINOv3 热图、DINOv3 叠加图和 GT 边界图的横向对比图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        Image.fromarray(image),
        Image.fromarray(colorize_heatmap(heat_map)),
        Image.fromarray(dinov3_overlay),
        Image.fromarray(gt_overlay),
    ]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    canvas.save(output_path)


def build_patch_boundary_target(
    instance_mask: np.ndarray,
    patch_shape: tuple[int, int],
    dilation_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把像素级实例边界转换成和 DINOv3 patch 网格同尺度的评价目标。"""
    gt_boundary = label_boundary_map(instance_mask)
    dilated_boundary = dilate_binary_mask(gt_boundary, radius=dilation_radius)
    patch_target_score = resize_float_map(
        dilated_boundary.astype(np.float32),
        size=(patch_shape[1], patch_shape[0]),
    )
    return patch_target_score > 0.0, gt_boundary


def summarize_metric(rows: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    """汇总 DINOv3 边界图单个评价指标的均值和中位数。"""
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {f"mean_{key}": None, f"median_{key}": None}
    return {f"mean_{key}": mean(values), f"median_{key}": median(values)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 DINOv3 边界热图在当前样本上的 patch 级指标。"""
    summary: dict[str, Any] = {"tiles": len(rows)}
    for key in (
        "roc_auc",
        "average_precision",
        "best_f1",
        "score_contrast",
        "mean_score_on_boundary",
        "mean_score_off_boundary",
        "positive_patches",
    ):
        summary.update(summarize_metric(rows, key))
    return summary


def main() -> int:
    """加载 DINOv3 并导出切片的特征距离边界热图和评价指标。"""
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
    summary_path = output_dir / "boundary_maps_summary.json"

    print(f"Dataset root: {dataset_root}")
    print(f"Tiles to process: {len(pairs)}")
    print("Loading DINOv3...")
    dinov3 = DINOv3Wrapper.from_config(dinov3_config)

    import torch

    rows: list[dict[str, Any]] = []
    with metadata_path.open("w", encoding="utf-8") as handle:
        for index, pair in enumerate(pairs):
            image = load_rgb_image(pair.image_path)
            instance_mask = load_instance_mask(pair.instance_path)
            model_input = prepare_dinov3_input(dinov3, image)

            with torch.inference_mode():
                raw_features = dinov3(model_input).raw

            feature_grid = extract_patch_feature_grid(raw_features)
            patch_boundary = boundary_map_from_feature_grid(feature_grid)
            patch_target, gt_boundary = build_patch_boundary_target(
                instance_mask,
                patch_shape=patch_boundary.shape,
                dilation_radius=args.gt_boundary_dilation,
            )
            image_boundary = resize_float_map(
                patch_boundary,
                size=(image.shape[1], image.shape[0]),
            )
            metrics = score_map_metrics(patch_boundary, patch_target)

            stem = pair.image_path.stem
            gray_path = output_dir / "gray" / f"{stem}.png"
            heat_path = output_dir / "heat" / f"{stem}.png"
            overlay_path = output_dir / "overlay" / f"{stem}.png"
            gt_overlay_path = output_dir / "gt_boundary" / f"{stem}.png"
            panel_path = output_dir / "panel" / f"{stem}.png"

            save_gray_map(image_boundary, gray_path)
            save_heat_map(image_boundary, heat_path)
            save_boundary_overlay(image, image_boundary, overlay_path, alpha=args.overlay_alpha)

            dinov3_overlay = np.asarray(Image.open(overlay_path).convert("RGB"))
            save_gt_boundary_overlay(image, gt_boundary, gt_overlay_path)
            gt_overlay = np.asarray(Image.open(gt_overlay_path).convert("RGB"))
            save_visual_panel(image, image_boundary, dinov3_overlay, gt_overlay, panel_path)

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
                "heat_path": str(heat_path),
                "overlay_path": str(overlay_path),
                "gt_overlay_path": str(gt_overlay_path),
                "panel_path": str(panel_path),
                "npy_path": str(npy_path) if npy_path else None,
                "boundary_min": float(patch_boundary.min()),
                "boundary_max": float(patch_boundary.max()),
                "boundary_mean": float(patch_boundary.mean()),
                "gt_boundary_dilation": args.gt_boundary_dilation,
                **metrics,
            }
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{index + 1}/{len(pairs)}] {pair.name}: "
                f"feature_grid={tuple(feature_grid.shape)}, "
                f"ap={metrics['average_precision']}, auc={metrics['roc_auc']}"
            )

    summary = summarize(rows)
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "gt_boundary_dilation": args.gt_boundary_dilation,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
