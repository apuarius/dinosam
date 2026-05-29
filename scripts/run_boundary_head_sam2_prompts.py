import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from dinosam.data.instance_tiles import (  # noqa: E402
    InstanceTilePair,
    list_instance_tile_pairs,
    load_instance_mask,
    load_rgb_image,
)
from dinosam.evaluation import binary_mask_stats, boundary_f1, first_binary_mask, mask_dice, mask_iou  # noqa: E402
from dinosam.models import (  # noqa: E402
    DINOv3Wrapper,
    PatchDetectionHead,
    PatchDetectionHeadConfig,
    SAM2ImageWrapper,
    build_dinov3_config,
    build_sam2_config,
)
from dinosam.project import resolve_project_path  # noqa: E402
from dinosam.prompting import InstancePrompt, build_sam2_prompt_kwargs  # noqa: E402
from dinosam.prompting.instance_prompts import prompt_from_binary_mask  # noqa: E402
from dinosam.train import load_config  # noqa: E402
from train_dinov3_boundary_head import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    feature_cache_paths,
    load_or_extract_features,
    load_training_values,
    resize_probability_map,
)


def build_parser() -> argparse.ArgumentParser:
    """构建自动 prompt 串联实验的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Run boundary-head proposals through SAM2 on validation tiles."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split-root", default=None)
    parser.add_argument("--output-dir", default="outputs/dinov3_sam2_auto_prompts")
    parser.add_argument(
        "--limit-val",
        type=int,
        default=8,
        help="Number of validation images to run; use <=0 to process the full split.",
    )
    parser.add_argument("--max-prompts-per-image", type=int, default=24)
    parser.add_argument("--min-proposal-area", type=int, default=512)
    parser.add_argument("--box-margin", type=int, default=4)
    parser.add_argument("--prompt-mode", choices=("box", "point", "box_point"), default="box_point")
    parser.add_argument("--boundary-threshold", type=float, default=None)
    parser.add_argument("--foreground-threshold", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--multimask-output", action=argparse.BooleanOptionalAction, default=False)
    return parser


def limit_pairs(pairs: list[InstanceTilePair], limit: int) -> list[InstanceTilePair]:
    """按需截断验证集，默认只跑少量图片来确认整条链路。"""
    if limit <= 0:
        return pairs
    return pairs[:limit]


def checkpoint_metric(metrics: dict[str, Any], name: str, default: float) -> float:
    """从 checkpoint 指标里读取阈值，缺失时使用保守默认值。"""
    value = metrics.get(name)
    if value is None:
        return default
    return float(value)


def head_config_from_state(state_dict: dict[str, torch.Tensor], checkpoint_args: dict[str, Any]) -> PatchDetectionHeadConfig:
    """从 checkpoint 权重形状恢复检测头结构，避免手动填写通道数。"""
    first_weight = state_dict["0.weight"]
    last_weight = state_dict["6.weight"]
    return PatchDetectionHeadConfig(
        input_channels=int(first_weight.shape[1]),
        hidden_channels=int(first_weight.shape[0]),
        dropout=float(checkpoint_args.get("dropout", 0.1)),
        output_channels=int(last_weight.shape[0]),
    )


def load_boundary_head(checkpoint_path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    """加载训练好的轻量检测头，并返回 checkpoint 中保存的指标。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["head_state_dict"]
    head = PatchDetectionHead.build(
        head_config_from_state(state_dict, checkpoint.get("args", {}))
    ).to(device)
    head.load_state_dict(state_dict)
    head.eval()
    return head, dict(checkpoint.get("metrics", {}))


def connected_patch_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """在 DINOv3 patch 网格上找连通块，作为粗粒度田块 proposal。"""
    source = mask.astype(bool, copy=False)
    visited = np.zeros(source.shape, dtype=bool)
    height, width = source.shape
    components: list[list[tuple[int, int]]] = []

    for y in range(height):
        for x in range(width):
            if visited[y, x] or not source[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            component: list[tuple[int, int]] = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or nx < 0 or ny >= height or nx >= width:
                        continue
                    if visited[ny, nx] or not source[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
            components.append(component)

    return components


def component_to_pixel_mask(
    component: Iterable[tuple[int, int]],
    grid_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """把 patch 连通块映射回原图像素区域，供 SAM2 生成 box 和正点。"""
    grid_height, grid_width = grid_shape
    image_height, image_width = image_shape
    mask = np.zeros(image_shape, dtype=bool)
    for patch_y, patch_x in component:
        y0 = patch_y * image_height // grid_height
        y1 = (patch_y + 1) * image_height // grid_height
        x0 = patch_x * image_width // grid_width
        x1 = (patch_x + 1) * image_width // grid_width
        mask[y0:y1, x0:x1] = True
    return mask


def proposal_score(
    component: list[tuple[int, int]],
    boundary_probability: np.ndarray,
    foreground_probability: np.ndarray,
) -> float:
    """用区域置信度减去边界置信度，为 proposal 排序。"""
    ys = np.asarray([item[0] for item in component], dtype=np.int64)
    xs = np.asarray([item[1] for item in component], dtype=np.int64)
    foreground = float(foreground_probability[ys, xs].mean())
    boundary = float(boundary_probability[ys, xs].mean())
    return foreground - boundary


def build_auto_prompts(
    *,
    boundary_probability: np.ndarray,
    foreground_probability: np.ndarray,
    image_shape: tuple[int, int],
    boundary_threshold: float,
    foreground_threshold: float,
    min_area: int,
    box_margin: int,
    max_prompts: int,
) -> list[InstancePrompt]:
    """把边界头输出转换成一组粗 box/point prompts。"""
    candidate = (foreground_probability >= foreground_threshold) & (
        boundary_probability < boundary_threshold
    )
    if not candidate.any():
        candidate = foreground_probability >= foreground_threshold
    if not candidate.any():
        y, x = np.unravel_index(int(np.argmax(foreground_probability)), foreground_probability.shape)
        candidate = np.zeros_like(foreground_probability, dtype=bool)
        candidate[y, x] = True

    scored_components = [
        (
            proposal_score(component, boundary_probability, foreground_probability),
            component,
        )
        for component in connected_patch_components(candidate)
    ]
    scored_components.sort(key=lambda item: item[0], reverse=True)

    prompts: list[InstancePrompt] = []
    grid_shape = tuple(int(value) for value in foreground_probability.shape)
    for _, component in scored_components:
        pixel_mask = component_to_pixel_mask(
            component=component,
            grid_shape=grid_shape,
            image_shape=image_shape,
        )
        if int(pixel_mask.sum()) < min_area:
            continue
        prompts.append(
            prompt_from_binary_mask(
                pixel_mask,
                instance_id=len(prompts) + 1,
                box_margin=box_margin,
            )
        )
        if len(prompts) >= max_prompts:
            break
    return prompts


def select_sam2_mask(prediction: Any) -> tuple[np.ndarray, float | None]:
    """从 SAM2 返回结果中选择分数最高的一张 mask。"""
    masks = np.asarray(prediction.masks)
    scores = np.asarray(prediction.scores).reshape(-1)
    if masks.ndim == 3 and scores.size == masks.shape[0]:
        index = int(np.argmax(scores))
        return masks[index].astype(bool), float(scores[index])
    return first_binary_mask(masks), float(scores[0]) if scores.size else None


def run_sam2_prompts(
    *,
    sam2: SAM2ImageWrapper,
    image: np.ndarray,
    prompts: list[InstancePrompt],
    prompt_mode: str,
    multimask_output: bool,
) -> tuple[np.ndarray, list[float]]:
    """对一张图的所有自动 prompt 逐个调用 SAM2，并合并成前景 mask。"""
    sam2.set_image(image)
    union_mask = np.zeros(image.shape[:2], dtype=bool)
    scores: list[float] = []
    for prompt in prompts:
        kwargs = build_sam2_prompt_kwargs(prompt, prompt_mode)  # type: ignore[arg-type]
        prediction = sam2.predict(**kwargs, multimask_output=multimask_output)
        mask, score = select_sam2_mask(prediction)
        union_mask |= mask
        if score is not None:
            scores.append(score)
    sam2.reset()
    return union_mask, scores


def blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> Image.Image:
    """把二值 mask 半透明叠加到 RGB 图像上，便于快速查看结果。"""
    base = image.astype(np.float32).copy()
    color_array = np.asarray(color, dtype=np.float32)
    base[mask.astype(bool, copy=False)] = base[mask.astype(bool, copy=False)] * 0.45 + color_array * 0.55
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def draw_prompts(image: np.ndarray, prompts: list[InstancePrompt]) -> Image.Image:
    """把自动生成的 box 和正点画到原图上。"""
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for prompt in prompts:
        box = [float(value) for value in prompt.box]
        draw.rectangle(box, outline=(0, 255, 160), width=3)
        for x, y in prompt.point_coords:
            radius = 4
            draw.ellipse(
                [float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius],
                fill=(255, 220, 0),
                outline=(0, 0, 0),
            )
    return canvas


def save_visualization(
    *,
    image: np.ndarray,
    prompts: list[InstancePrompt],
    prediction: np.ndarray,
    target: np.ndarray,
    output_path: Path,
) -> None:
    """保存原图、自动 prompt、SAM2 合并结果和 GT 前景的四联图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        Image.fromarray(image),
        draw_prompts(image, prompts),
        blend_mask(image, prediction, (255, 80, 80)),
        blend_mask(image, target, (0, 255, 120)),
    ]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    canvas.save(output_path)


def write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出每张图的验证指标，方便后续用表格查看。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """汇总平均指标，只作为打通链路阶段的粗略参考。"""
    summary: dict[str, float] = {}
    metric_names = (
        "iou",
        "dice",
        "precision",
        "recall",
        "f1",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "sam2_score",
    )
    for name in metric_names:
        values = [float(row[name]) for row in rows if row.get(name) not in (None, "")]
        if values:
            summary[f"mean_{name}"] = float(np.mean(values))
    return summary


def ensure_feature_source(
    *,
    pairs: list[InstanceTilePair],
    cache_dir: Path,
    cache_features: bool,
    model_config: dict[str, Any],
    device: torch.device,
) -> DINOv3Wrapper | None:
    """缓存完整时不加载 DINOv3，节省后续 SAM2 推理显存。"""
    names = [pair.name for pair in pairs]
    cached = cache_features and all(path.exists() for path in feature_cache_paths(cache_dir, "val", names))
    if cached:
        return None

    dinov3_config = replace(build_dinov3_config(model_config), device=str(device))
    return DINOv3Wrapper.from_config(dinov3_config)


def predict_boundary_head(
    *,
    pair: InstanceTilePair,
    image: Image.Image,
    dinov3: DINOv3Wrapper | None,
    head: torch.nn.Module,
    cache_dir: Path,
    cache_features: bool,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """读取或提取 DINOv3 特征，然后用检测头输出边界和前景概率。"""
    if dinov3 is None:
        path = feature_cache_paths(cache_dir, "val", [pair.name])[0]
        features = torch.load(path, map_location=device).float()[None, ...]
    else:
        features = load_or_extract_features(
            dinov3=dinov3,
            images=[image],
            names=[pair.name],
            split="val",
            cache_dir=cache_dir,
            cache_features=cache_features,
            device=device,
        )

    with torch.no_grad():
        probabilities = torch.sigmoid(head(features.to(device))).detach().float().cpu().numpy()[0]
    return probabilities[0], probabilities[1]


def main() -> int:
    """执行边界头到 SAM2 的最小闭环验证。"""
    args = build_parser().parse_args()
    training_values = load_training_values(resolve_project_path(args.config))
    train_output_dir = resolve_project_path(training_values["output_dir"])
    checkpoint_path = (
        resolve_project_path(args.checkpoint)
        if args.checkpoint
        else train_output_dir / "checkpoints" / "best.pt"
    )
    output_dir = resolve_project_path(args.output_dir)
    cache_dir = (
        resolve_project_path(args.feature_cache_dir)
        if args.feature_cache_dir
        else resolve_project_path(training_values["feature_cache_dir"])
        if training_values.get("feature_cache_dir")
        else train_output_dir / "feature_cache"
    )
    cache_features = bool(training_values["cache_features"] if args.cache_features is None else args.cache_features)
    split_root = args.split_root or training_values["val_root"]
    device = torch.device(args.device or training_values.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    model_config = load_config(resolve_project_path(training_values["model_config"]))
    pairs = limit_pairs(list_instance_tile_pairs(split_root), args.limit_val)
    if not pairs:
        raise RuntimeError("No validation tiles selected.")

    head, checkpoint_metrics = load_boundary_head(checkpoint_path, device=device)
    boundary_threshold = (
        float(args.boundary_threshold)
        if args.boundary_threshold is not None
        else checkpoint_metric(checkpoint_metrics, "val_boundary_best_threshold", 0.05)
    )
    foreground_threshold = (
        float(args.foreground_threshold)
        if args.foreground_threshold is not None
        else checkpoint_metric(checkpoint_metrics, "val_foreground_best_threshold", 0.5)
    )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Images: {len(pairs)} from {resolve_project_path(split_root)}")
    print(f"Device: {device}")
    print(f"Feature cache: {'on' if cache_features else 'off'} -> {cache_dir}")
    print(f"Boundary threshold: {boundary_threshold:.4f}")
    print(f"Foreground threshold: {foreground_threshold:.4f}")
    print(f"Output dir: {output_dir}")

    dinov3 = ensure_feature_source(
        pairs=pairs,
        cache_dir=cache_dir,
        cache_features=cache_features,
        model_config=model_config,
        device=device,
    )
    sam2_config = replace(build_sam2_config(model_config), device=str(device))
    sam2 = SAM2ImageWrapper.from_config(sam2_config)

    rows: list[dict[str, Any]] = []
    prediction_dir = output_dir / "predictions"
    visualization_dir = output_dir / "visualizations"
    for pair in tqdm(pairs, desc="auto prompts", dynamic_ncols=True):
        pil_image = Image.open(pair.image_path).convert("RGB")
        image = load_rgb_image(pair.image_path)
        instance_mask = load_instance_mask(pair.instance_path)
        target = instance_mask > 0

        boundary_probability, foreground_probability = predict_boundary_head(
            pair=pair,
            image=pil_image,
            dinov3=dinov3,
            head=head,
            cache_dir=cache_dir,
            cache_features=cache_features,
            device=device,
        )
        prompts = build_auto_prompts(
            boundary_probability=boundary_probability,
            foreground_probability=foreground_probability,
            image_shape=image.shape[:2],
            boundary_threshold=boundary_threshold,
            foreground_threshold=foreground_threshold,
            min_area=args.min_proposal_area,
            box_margin=args.box_margin,
            max_prompts=args.max_prompts_per_image,
        )
        prediction, sam2_scores = run_sam2_prompts(
            sam2=sam2,
            image=image,
            prompts=prompts,
            prompt_mode=args.prompt_mode,
            multimask_output=args.multimask_output,
        )

        prediction_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray((prediction.astype(np.uint8) * 255)).save(prediction_dir / pair.name)
        save_visualization(
            image=image,
            prompts=prompts,
            prediction=prediction,
            target=target,
            output_path=visualization_dir / pair.name,
        )

        mask_stats = binary_mask_stats(prediction, target)
        boundary_stats = boundary_f1(prediction, target)
        row = {
            "tile": pair.name,
            "prompts": len(prompts),
            "prediction_pixels": int(prediction.sum()),
            "target_pixels": int(target.sum()),
            "iou": mask_iou(prediction, target),
            "dice": mask_dice(prediction, target),
            "precision": mask_stats["precision"],
            "recall": mask_stats["recall"],
            "f1": mask_stats["f1"],
            "boundary_precision": boundary_stats["boundary_precision"],
            "boundary_recall": boundary_stats["boundary_recall"],
            "boundary_f1": boundary_stats["boundary_f1"],
            "sam2_score": float(np.mean(sam2_scores)) if sam2_scores else None,
        }
        rows.append(row)

    summary = summarize_rows(rows)
    write_metrics_csv(output_dir / "metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "images": len(rows),
                "boundary_threshold": boundary_threshold,
                "foreground_threshold": foreground_threshold,
                "prompt_mode": args.prompt_mode,
                "max_prompts_per_image": args.max_prompts_per_image,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.4f}")
    print(f"Saved metrics: {output_dir / 'metrics.csv'}")
    print(f"Saved visualizations: {visualization_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
