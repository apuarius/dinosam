import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SRC_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dinosam.data.instance_tiles import (  # noqa: E402
    InstanceTilePair,
    list_instance_tile_pairs,
    load_instance_mask,
    load_rgb_image,
)
from dinosam.evaluation import (  # noqa: E402
    PredictionInstance,
    binary_instances_from_label_mask,
    mask_average_precision,
)
from dinosam.models import SAM2ImageWrapper, build_sam2_config  # noqa: E402
from dinosam.project import resolve_project_path  # noqa: E402
from dinosam.train import load_config  # noqa: E402
from run_boundary_head_sam2_prompts import (  # noqa: E402
    blend_mask,
    build_auto_prompts,
    checkpoint_metric,
    draw_prompts,
    ensure_feature_source,
    load_boundary_head,
    predict_boundary_head,
    prompt_point_counts,
    run_sam2_prompts,
)
from train_dinov3_boundary_head import load_training_values, resize_probability_map  # noqa: E402


# =========================
# USER SETTINGS
# =========================
# Edit these paths on the server, then run:
#   python scripts/infer_single_t1_image.py
CONFIG_PATH = "configs/train/dinov3_boundary_head_t1.yaml"
CHECKPOINT_PATH = "outputs/dinov3_boundary_head_t1/checkpoints/best.pt"

# Set IMAGE_PATH to a specific test image path. Keep None to randomly select one
# paired image from TEST_ROOT.
IMAGE_PATH = None
MASK_PATH = None
TEST_ROOT = "data/images/test"

OUTPUT_DIR = "outputs/t1_single_image_inference"
SPLIT_NAME = "test"
SEED = None

# Prompt and runtime settings. Usually keep these unchanged for T1.
MAX_PROMPTS_PER_IMAGE = 24
MIN_PROPOSAL_AREA = 512
BOX_MARGIN = 4
POSITIVE_POINTS_PER_PROMPT = 1
PROMPT_MODE = "box_point"
BOUNDARY_THRESHOLD = None
FOREGROUND_THRESHOLD = None
DEVICE = None
CACHE_FEATURES = None
FEATURE_CACHE_DIR = None
MULTIMASK_OUTPUT = False


def build_settings() -> SimpleNamespace:
    return SimpleNamespace(
        config=CONFIG_PATH,
        checkpoint=CHECKPOINT_PATH,
        image=IMAGE_PATH,
        mask=MASK_PATH,
        test_root=TEST_ROOT,
        split_name=SPLIT_NAME,
        output_dir=OUTPUT_DIR,
        seed=SEED,
        max_prompts_per_image=MAX_PROMPTS_PER_IMAGE,
        min_proposal_area=MIN_PROPOSAL_AREA,
        box_margin=BOX_MARGIN,
        positive_points_per_prompt=POSITIVE_POINTS_PER_PROMPT,
        prompt_mode=PROMPT_MODE,
        boundary_threshold=BOUNDARY_THRESHOLD,
        foreground_threshold=FOREGROUND_THRESHOLD,
        device=DEVICE,
        cache_features=CACHE_FEATURES,
        feature_cache_dir=FEATURE_CACHE_DIR,
        multimask_output=MULTIMASK_OUTPUT,
    )


def choose_input_pair(args: SimpleNamespace, training_values: dict[str, Any]) -> tuple[InstanceTilePair, bool]:
    if args.image:
        image_path = resolve_project_path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")
        mask_path = resolve_project_path(args.mask) if args.mask else image_path
        if args.mask and not mask_path.exists():
            raise FileNotFoundError(f"Input mask does not exist: {mask_path}")
        return InstanceTilePair(image_path=image_path, instance_path=mask_path), bool(args.mask)

    test_root = args.test_root or training_values.get("test_root")
    if not test_root:
        raise ValueError("--test-root is required when config does not define test_root.")
    pairs = list_instance_tile_pairs(test_root)
    if not pairs:
        raise RuntimeError(f"No paired test images found in: {resolve_project_path(test_root)}")
    rng = random.Random(args.seed)
    return rng.choice(pairs), True


def save_probability_map(probability: np.ndarray, image_size: tuple[int, int], output_path: Path) -> None:
    resized = resize_probability_map(probability, image_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(resized, 0.0, 1.0) * 255).astype(np.uint8)).save(output_path)


def save_panel(
    *,
    image: np.ndarray,
    prompts: Any,
    prediction: np.ndarray,
    output_path: Path,
    target: np.ndarray | None = None,
) -> None:
    panels = [
        Image.fromarray(image),
        draw_prompts(image, prompts),
        blend_mask(image, prediction, (255, 80, 80)),
    ]
    if target is not None:
        panels.append(blend_mask(image, target, (0, 255, 120)))
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def build_metrics(
    prediction_instances: list[PredictionInstance],
    target_instance_mask: np.ndarray | None,
    latency_ms: float,
) -> dict[str, float] | None:
    if target_instance_mask is None:
        return None
    target_instances = binary_instances_from_label_mask(target_instance_mask)
    metrics = mask_average_precision([prediction_instances], [target_instances])
    metrics["Latency (ms)"] = latency_ms
    return metrics


def main() -> int:
    if len(sys.argv) > 1:
        raise SystemExit(
            "This script is configured in code. Edit USER SETTINGS at the top of "
            "scripts/infer_single_t1_image.py, then run: python scripts/infer_single_t1_image.py"
        )
    args = build_settings()
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
    split_name = str(args.split_name).strip() or "test"
    cache_features = bool(training_values["cache_features"] if args.cache_features is None else args.cache_features)
    device = torch.device(args.device or training_values.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))

    pair, has_target = choose_input_pair(args, training_values)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    model_config = load_config(resolve_project_path(training_values["model_config"]))
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

    print(f"Image: {pair.image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output dir: {output_dir}")

    dinov3 = ensure_feature_source(
        pairs=[pair],
        split_name=split_name,
        cache_dir=cache_dir,
        cache_features=cache_features,
        model_config=model_config,
        device=device,
    )
    sam2_config = replace(build_sam2_config(model_config), device=str(device))
    sam2 = SAM2ImageWrapper.from_config(sam2_config)

    pil_image = Image.open(pair.image_path).convert("RGB")
    image = load_rgb_image(pair.image_path)
    target_instance_mask = load_instance_mask(pair.instance_path) if has_target else None
    target = target_instance_mask > 0 if target_instance_mask is not None else None

    inference_start = time.perf_counter()
    boundary_probability, foreground_probability = predict_boundary_head(
        pair=pair,
        image=pil_image,
        dinov3=dinov3,
        head=head,
        split_name=split_name,
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
        positive_points_per_prompt=args.positive_points_per_prompt,
    )
    prediction, _, prediction_instances = run_sam2_prompts(
        sam2=sam2,
        image=image,
        prompts=prompts,
        prompt_mode=args.prompt_mode,
        multimask_output=args.multimask_output,
    )
    inference_sec = time.perf_counter() - inference_start
    latency_ms = inference_sec * 1000.0

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pair.image_path.stem
    mask_path = output_dir / f"{stem}_mask.png"
    overlay_path = output_dir / f"{stem}_overlay.png"
    prompts_path = output_dir / f"{stem}_prompts.png"
    panel_path = output_dir / f"{stem}_panel.png"
    boundary_path = output_dir / f"{stem}_boundary_prob.png"
    foreground_path = output_dir / f"{stem}_foreground_prob.png"
    summary_path = output_dir / f"{stem}_summary.json"

    Image.fromarray((prediction.astype(np.uint8) * 255)).save(mask_path)
    blend_mask(image, prediction, (255, 80, 80)).save(overlay_path)
    draw_prompts(image, prompts).save(prompts_path)
    save_panel(image=image, prompts=prompts, prediction=prediction, target=target, output_path=panel_path)
    save_probability_map(boundary_probability, pil_image.size, boundary_path)
    save_probability_map(foreground_probability, pil_image.size, foreground_path)

    positive_points, total_points = prompt_point_counts(prompts)
    metrics = build_metrics(prediction_instances, target_instance_mask, latency_ms)
    summary = {
        "image": str(pair.image_path),
        "mask": str(pair.instance_path) if has_target else None,
        "checkpoint": str(checkpoint_path),
        "split_name": split_name,
        "boundary_threshold": boundary_threshold,
        "foreground_threshold": foreground_threshold,
        "prompt_mode": args.prompt_mode,
        "prompts": len(prompts),
        "positive_points": positive_points,
        "points": total_points,
        "Latency (ms)": latency_ms,
        "prediction_pixels": int(prediction.sum()),
        "metrics": metrics,
        "outputs": {
            "mask": str(mask_path),
            "overlay": str(overlay_path),
            "prompts": str(prompts_path),
            "panel": str(panel_path),
            "boundary_probability": str(boundary_path),
            "foreground_probability": str(foreground_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prompts: {len(prompts)}")
    print(f"Positive points: {positive_points}")
    if metrics is not None:
        print(
            "Metrics: "
            f"mAP@0.5={metrics['mAP@0.5']:.4f}, "
            f"mAP@0.5:0.95={metrics['mAP@0.5:0.95']:.4f}, "
            f"Latency={metrics['Latency (ms)']:.2f} ms"
        )
    else:
        print(f"Latency: {latency_ms:.2f} ms")
    print(f"Saved mask: {mask_path}")
    print(f"Saved panel: {panel_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
