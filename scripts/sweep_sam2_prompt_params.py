import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from dinosam.data.instance_tiles import list_instance_tile_pairs, load_instance_mask, load_rgb_image  # noqa: E402
from dinosam.evaluation import binary_mask_stats, boundary_f1, mask_dice, mask_iou  # noqa: E402
from dinosam.models import DINOv3Wrapper, SAM2ImageWrapper, build_dinov3_config, build_sam2_config  # noqa: E402
from dinosam.project import resolve_project_path  # noqa: E402
from dinosam.train import load_config  # noqa: E402
from run_boundary_head_sam2_prompts import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    build_auto_prompts,
    checkpoint_size_mb,
    feature_cache_paths,
    limit_pairs,
    load_boundary_head,
    load_or_extract_features,
    load_training_values,
    model_size_mb,
    run_sam2_prompts,
    summarize_rows,
    write_metrics_csv,
)


def parse_float_list(text: str) -> list[float]:
    """解析逗号分隔的浮点参数列表。"""
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_list(text: str) -> list[int]:
    """解析逗号分隔的整数参数列表。"""
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    """构建 SAM2 prompt 参数网格搜索脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Sweep SAM2 auto-prompt parameters on a fixed validation subset.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split-root", default=None)
    parser.add_argument("--output-dir", default="outputs/sweeps/sam2_prompt_params")
    parser.add_argument("--limit-val", type=int, default=50, help="Use <=0 to sweep the full validation split.")
    parser.add_argument("--box-margins", default="56,64")
    parser.add_argument("--boundary-thresholds", default="0.10,0.12,0.14")
    parser.add_argument("--foreground-thresholds", default="0.05,0.06")
    parser.add_argument("--max-prompts", default="56,72")
    parser.add_argument("--min-proposal-area", type=int, default=512)
    parser.add_argument("--prompt-mode", choices=("box", "point", "box_point"), default="box_point")
    parser.add_argument("--multimask-output", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--sort-by", default="mean_iou")
    parser.add_argument("--min-precision", type=float, default=0.0)
    return parser


def mean_or_none(values: list[float | None]) -> float | None:
    """计算非空数值平均值。"""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return float(np.mean(clean))


def write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """保存每组参数的汇总指标。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_probability_cache(
    *,
    pairs: list[Any],
    training_values: dict[str, Any],
    model_config: dict[str, Any],
    checkpoint_path: Path,
    device: torch.device,
    cache_dir: Path,
    cache_features: bool,
) -> tuple[list[dict[str, Any]], torch.nn.Module, dict[str, Any]]:
    """预先计算每张图的 boundary/foreground 概率图，避免每组参数重复跑检测头。"""
    head, checkpoint_metrics = load_boundary_head(checkpoint_path, device=device)
    names = [pair.name for pair in pairs]
    cached = cache_features and all(path.exists() for path in feature_cache_paths(cache_dir, "val", names))
    dinov3 = None
    if not cached:
        dinov3_config = replace(build_dinov3_config(model_config), device=str(device))
        dinov3 = DINOv3Wrapper.from_config(dinov3_config)

    items: list[dict[str, Any]] = []
    for pair in tqdm(pairs, desc="precompute maps", dynamic_ncols=True):
        pil_image = Image.open(pair.image_path).convert("RGB")
        image = load_rgb_image(pair.image_path)
        target = load_instance_mask(pair.instance_path) > 0
        if dinov3 is None:
            feature_path = feature_cache_paths(cache_dir, "val", [pair.name])[0]
            features = torch.load(feature_path, map_location=device).float()[None, ...]
        else:
            features = load_or_extract_features(
                dinov3=dinov3,
                images=[pil_image],
                names=[pair.name],
                split="val",
                cache_dir=cache_dir,
                cache_features=cache_features,
                device=device,
            )

        with torch.no_grad():
            probabilities = torch.sigmoid(head(features.to(device))).detach().float().cpu().numpy()[0]
        items.append(
            {
                "name": pair.name,
                "image": image,
                "target": target,
                "boundary_probability": probabilities[0],
                "foreground_probability": probabilities[1],
            }
        )
    return items, head, checkpoint_metrics


def evaluate_combo(
    *,
    items: list[dict[str, Any]],
    sam2: SAM2ImageWrapper,
    box_margin: int,
    boundary_threshold: float,
    foreground_threshold: float,
    max_prompts: int,
    min_proposal_area: int,
    prompt_mode: str,
    multimask_output: bool,
) -> dict[str, Any]:
    """在固定样本集上评估一组 prompt 参数。"""
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for item in items:
        image = item["image"]
        target = item["target"]
        prompts = build_auto_prompts(
            boundary_probability=item["boundary_probability"],
            foreground_probability=item["foreground_probability"],
            image_shape=image.shape[:2],
            boundary_threshold=boundary_threshold,
            foreground_threshold=foreground_threshold,
            min_area=min_proposal_area,
            box_margin=box_margin,
            max_prompts=max_prompts,
        )
        image_start = time.perf_counter()
        prediction, sam2_scores = run_sam2_prompts(
            sam2=sam2,
            image=image,
            prompts=prompts,
            prompt_mode=prompt_mode,
            multimask_output=multimask_output,
        )
        inference_sec = time.perf_counter() - image_start
        mask_stats = binary_mask_stats(prediction, target)
        boundary_stats = boundary_f1(prediction, target)
        rows.append(
            {
                "tile": item["name"],
                "prompts": len(prompts),
                "iou": mask_iou(prediction, target),
                "dice": mask_dice(prediction, target),
                "precision": mask_stats["precision"],
                "recall": mask_stats["recall"],
                "f1": mask_stats["f1"],
                "boundary_precision": boundary_stats["boundary_precision"],
                "boundary_recall": boundary_stats["boundary_recall"],
                "boundary_f1": boundary_stats["boundary_f1"],
                "sam2_score": float(np.mean(sam2_scores)) if sam2_scores else None,
                "inference_sec": inference_sec,
            }
        )

    summary = summarize_rows(rows)
    total_sec = time.perf_counter() - start
    return {
        "rows": rows,
        "summary": summary,
        "total_inference_sec": total_sec,
        "mean_inference_sec": total_sec / max(len(items), 1),
        "mean_prompts": mean_or_none([float(row["prompts"]) for row in rows]),
    }


def main() -> int:
    """执行固定规则的 SAM2 prompt 参数搜索。"""
    args = build_parser().parse_args()
    output_dir = resolve_project_path(args.output_dir)
    training_values = load_training_values(resolve_project_path(args.config))
    train_output_dir = resolve_project_path(training_values["output_dir"])
    checkpoint_path = (
        resolve_project_path(args.checkpoint)
        if args.checkpoint
        else train_output_dir / "checkpoints" / "best.pt"
    )
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

    items, head, checkpoint_metrics = prepare_probability_cache(
        pairs=pairs,
        training_values=training_values,
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        device=device,
        cache_dir=cache_dir,
        cache_features=cache_features,
    )
    del checkpoint_metrics

    sam2_config = replace(build_sam2_config(model_config), device=str(device))
    sam2 = SAM2ImageWrapper.from_config(sam2_config)
    head_size = model_size_mb(head)
    ckpt_size = checkpoint_size_mb(checkpoint_path)

    grid = list(
        product(
            parse_int_list(args.box_margins),
            parse_float_list(args.boundary_thresholds),
            parse_float_list(args.foreground_thresholds),
            parse_int_list(args.max_prompts),
        )
    )
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Images: {len(items)} from {resolve_project_path(split_root)}")
    print(f"Grid size: {len(grid)}")
    print(f"Head model size: {head_size:.2f} MB")
    print(f"Checkpoint size: {ckpt_size:.2f} MB")
    print(f"Output dir: {output_dir}")

    result_rows: list[dict[str, Any]] = []
    for box_margin, boundary_threshold, foreground_threshold, max_prompts in tqdm(
        grid,
        desc="sweep params",
        dynamic_ncols=True,
    ):
        result = evaluate_combo(
            items=items,
            sam2=sam2,
            box_margin=box_margin,
            boundary_threshold=boundary_threshold,
            foreground_threshold=foreground_threshold,
            max_prompts=max_prompts,
            min_proposal_area=args.min_proposal_area,
            prompt_mode=args.prompt_mode,
            multimask_output=args.multimask_output,
        )
        summary = result["summary"]
        row = {
            "box_margin": box_margin,
            "boundary_threshold": boundary_threshold,
            "foreground_threshold": foreground_threshold,
            "max_prompts_per_image": max_prompts,
            "images": len(items),
            "mean_prompts": result["mean_prompts"],
            "total_inference_sec": result["total_inference_sec"],
            "mean_inference_sec": result["mean_inference_sec"],
            "head_model_size_mb": head_size,
            "checkpoint_size_mb": ckpt_size,
        }
        row.update(summary)
        precision = float(row.get("mean_precision") or 0.0)
        row["valid_by_precision"] = precision >= args.min_precision
        row["rank_score"] = float(row.get(args.sort_by) or -1.0) if row["valid_by_precision"] else -1.0
        result_rows.append(row)

    result_rows.sort(
        key=lambda row: (
            float(row.get("rank_score") or -1.0),
            float(row.get("mean_boundary_f1") or -1.0),
        ),
        reverse=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(output_dir / "results.csv", result_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "images": len(items),
                "grid_size": len(grid),
                "sort_by": args.sort_by,
                "min_precision": args.min_precision,
                "head_model_size_mb": head_size,
                "checkpoint_size_mb": ckpt_size,
                "best": result_rows[0] if result_rows else None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Best configuration:")
    if result_rows:
        best = result_rows[0]
        for key in (
            "box_margin",
            "boundary_threshold",
            "foreground_threshold",
            "max_prompts_per_image",
            "mean_iou",
            "mean_dice",
            "mean_boundary_f1",
            "mean_inference_sec",
            "head_model_size_mb",
            "checkpoint_size_mb",
        ):
            print(f"  {key}: {best.get(key)}")
    print(f"Saved sweep results: {output_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
