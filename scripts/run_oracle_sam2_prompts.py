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
from PIL import Image, ImageDraw  # noqa: E402

from dinosam.data.instance_tiles import (  # noqa: E402
    list_instance_tile_pairs,
    load_instance_mask,
    load_rgb_image,
    resolve_instance_dataset_root,
)
from dinosam.evaluation import mask_iou  # noqa: E402
from dinosam.models import SAM2ImageWrapper, build_sam2_config  # noqa: E402
from dinosam.project import resolve_project_path  # noqa: E402
from dinosam.prompting import build_sam2_prompt_kwargs, prompts_from_instance_mask  # noqa: E402
from dinosam.prompting.instance_prompts import InstancePrompt  # noqa: E402
from dinosam.train import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建 oracle prompt 实验的命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Run SAM2 with prompts generated from ground-truth instance masks."
    )
    parser.add_argument(
        "--dataset-root",
        default="data/sam2-dataset-1024",
        help="Dataset root containing Image/Instance or an All/Image + All/Instance layout.",
    )
    parser.add_argument(
        "--model-config",
        default="configs/model/dinov3_sam2.yaml",
        help="Model config containing the SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/oracle_sam2_prompts",
        help="Directory used to save jsonl summaries and optional overlays.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("box", "point", "box_point"),
        default="box_point",
        help="Which oracle prompt fields to pass into SAM2.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of image tiles to process.",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=256,
        help="Ignore instance masks smaller than this pixel area.",
    )
    parser.add_argument(
        "--box-margin",
        type=int,
        default=2,
        help="Pixel margin added around GT boxes before passing them to SAM2.",
    )
    parser.add_argument(
        "--max-instances-per-image",
        type=int,
        default=None,
        help="Optional cap on instances processed per image tile.",
    )
    parser.add_argument(
        "--save-overlays",
        type=int,
        default=3,
        help="Number of image-level prediction overlays to save.",
    )
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Ask SAM2 to return multiple masks and choose the highest-score one.",
    )
    return parser


def require_file(path: Path, label: str) -> None:
    """检查必需文件是否存在，缺失时给出明确错误。"""
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def select_sam2_mask(mask_batch: Any, score_batch: Any) -> tuple[np.ndarray, float | None]:
    """从 SAM2 返回结果中选择最高分二维 mask。"""
    masks = np.asarray(mask_batch)
    scores = np.asarray(score_batch).reshape(-1)

    if masks.ndim == 2:
        score = float(scores[0]) if len(scores) else None
        return masks.astype(bool), score

    while masks.ndim > 3:
        masks = masks[0]

    if masks.ndim == 3:
        index = int(np.argmax(scores)) if len(scores) == masks.shape[0] else 0
        score = float(scores[index]) if len(scores) > index else None
        return masks[index].astype(bool), score

    raise ValueError(f"Cannot select a 2D mask from shape: {masks.shape}")


def color_for_instance(instance_id: int) -> tuple[int, int, int, int]:
    """为实例 ID 生成稳定的半透明可视化颜色。"""
    return (
        60 + (instance_id * 53) % 170,
        60 + (instance_id * 97) % 170,
        60 + (instance_id * 151) % 170,
        120,
    )


def save_overlay(
    image: np.ndarray,
    predicted_masks: list[tuple[int, np.ndarray]],
    prompts: list[InstancePrompt],
    output_path: Path,
) -> None:
    """保存 SAM2 预测 mask、GT box 和正点的叠加图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = image.shape[:2]
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    for instance_id, mask in predicted_masks:
        overlay[mask] = color_for_instance(instance_id)

    base = Image.fromarray(image).convert("RGBA")
    composite = Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))
    draw = ImageDraw.Draw(composite)

    for prompt in prompts:
        draw.rectangle([float(value) for value in prompt.box], outline=(0, 255, 120, 255), width=2)
        x, y = prompt.point_coords[0]
        radius = 3
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(255, 235, 59, 255),
        )

    composite.convert("RGB").save(output_path)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 oracle prompt 实验的关键指标。"""
    ious = [float(row["iou"]) for row in rows]
    if not ious:
        return {"instances": 0, "mean_iou": None, "median_iou": None}
    return {
        "instances": len(ious),
        "mean_iou": mean(ious),
        "median_iou": median(ious),
        "min_iou": min(ious),
        "max_iou": max(ious),
    }


def main() -> int:
    """用 GT 实例 mask 生成 prompt，并调用 SAM2 建立 oracle 基线。"""
    args = build_parser().parse_args()

    dataset_root = resolve_instance_dataset_root(args.dataset_root)
    model_config_path = resolve_project_path(args.model_config)
    output_dir = resolve_project_path(args.output_dir)
    require_file(model_config_path, "Model config")

    model_config = load_config(model_config_path)
    sam2_config = build_sam2_config(model_config)
    if sam2_config.checkpoint is None:
        raise ValueError("SAM2 checkpoint is not configured.")
    require_file(resolve_project_path(sam2_config.checkpoint), "SAM2 checkpoint")

    pairs = list_instance_tile_pairs(dataset_root)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"oracle_{args.prompt_mode}.jsonl"
    summary_path = output_dir / f"oracle_{args.prompt_mode}_summary.json"

    print(f"Dataset root: {dataset_root}")
    print(f"Tiles to process: {len(pairs)}")
    print(f"Prompt mode: {args.prompt_mode}")
    print("Loading SAM2...")
    sam2 = SAM2ImageWrapper.from_config(sam2_config)

    rows: list[dict[str, Any]] = []
    with results_path.open("w", encoding="utf-8") as handle:
        for image_index, pair in enumerate(pairs):
            image = load_rgb_image(pair.image_path)
            instance_mask = load_instance_mask(pair.instance_path)
            prompts = prompts_from_instance_mask(
                instance_mask,
                min_area=args.min_area,
                box_margin=args.box_margin,
                max_instances=args.max_instances_per_image,
            )

            print(f"[{image_index + 1}/{len(pairs)}] {pair.name}: {len(prompts)} prompts")
            sam2.set_image(image)

            predicted_masks: list[tuple[int, np.ndarray]] = []
            for prompt in prompts:
                kwargs = build_sam2_prompt_kwargs(prompt, mode=args.prompt_mode)
                prediction = sam2.predict(
                    **kwargs,
                    multimask_output=args.multimask_output,
                )
                predicted_mask, score = select_sam2_mask(prediction.masks, prediction.scores)
                target_mask = instance_mask == prompt.instance_id
                iou = mask_iou(predicted_mask, target_mask)
                predicted_masks.append((prompt.instance_id, predicted_mask))

                row = {
                    "image": pair.name,
                    "instance_id": prompt.instance_id,
                    "area": prompt.area,
                    "box": [float(value) for value in prompt.box.tolist()],
                    "point": [float(value) for value in prompt.point_coords[0].tolist()],
                    "prompt_mode": args.prompt_mode,
                    "score": score,
                    "iou": iou,
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            if image_index < args.save_overlays:
                overlay_path = output_dir / "overlays" / pair.name
                save_overlay(image, predicted_masks, prompts, overlay_path)

    summary = summarize(rows)
    summary.update(
        {
            "dataset_root": str(dataset_root),
            "tiles": len(pairs),
            "prompt_mode": args.prompt_mode,
            "min_area": args.min_area,
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
