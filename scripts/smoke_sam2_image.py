import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from dinosam.models import SAM2ImageWrapper, build_sam2_config  # noqa: E402
from dinosam.project import resolve_project_path  # noqa: E402
from dinosam.train import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """构建 SAM2 单图 smoke test 的命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Run a one-image SAM2 smoke test.")
    parser.add_argument(
        "--model-config",
        default="configs/model/dinov3_sam2.yaml",
        help="Model config containing the SAM2 checkpoint path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/visualizations/sam2_smoke.png",
        help="Path used to save the visualization image.",
    )
    return parser


def require_file(path: Path, label: str) -> None:
    """检查必需文件是否存在，缺失时给出明确错误。"""
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def make_synthetic_image(size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """生成一张简单 RGB 图片和一个包住目标区域的 box prompt。"""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = (28, 42, 56)

    margin = size // 4
    image[margin : size - margin, margin : size - margin] = (220, 190, 90)
    image[margin + 32 : size - margin - 32, margin + 32 : size - margin - 32] = (
        80,
        170,
        210,
    )

    box = np.array([margin, margin, size - margin, size - margin], dtype=np.float32)
    return image, box


def first_mask(mask_batch: Any) -> np.ndarray:
    """从 SAM2 返回的 mask 批次中取出第一张二维 mask。"""
    masks = np.asarray(mask_batch)
    while masks.ndim > 2:
        masks = masks[0]
    return masks.astype(bool)


def save_visualization(image: np.ndarray, mask: np.ndarray, box: np.ndarray, output: Path) -> None:
    """把输入图、预测 mask 和 box prompt 合成一张便于人工检查的图片。"""
    output.parent.mkdir(parents=True, exist_ok=True)

    base = Image.fromarray(image).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    overlay_pixels = overlay.load()

    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x]:
                overlay_pixels[x, y] = (255, 80, 80, 120)

    draw = ImageDraw.Draw(overlay)
    draw.rectangle([float(value) for value in box], outline=(0, 255, 120, 255), width=4)

    Image.alpha_composite(base, overlay).convert("RGB").save(output)


def main() -> int:
    """加载 SAM2 checkpoint，对合成图片执行一次 box prompt 分割。"""
    args = build_parser().parse_args()

    model_config_path = resolve_project_path(args.model_config)
    output_path = resolve_project_path(args.output)
    require_file(model_config_path, "Model config")

    model_config = load_config(model_config_path)
    sam2_config = build_sam2_config(model_config)
    if sam2_config.checkpoint is None:
        raise ValueError("SAM2 checkpoint is not configured.")

    checkpoint_path = resolve_project_path(sam2_config.checkpoint)
    require_file(checkpoint_path, "SAM2 checkpoint")

    print(f"Loading SAM2 from: {checkpoint_path}")
    sam2 = SAM2ImageWrapper.from_config(sam2_config)

    image, box = make_synthetic_image()
    print(f"Input image shape: {image.shape}")
    print(f"Box prompt: {box.tolist()}")

    sam2.set_image(image)
    prediction = sam2.predict(box=box, multimask_output=False)
    mask = first_mask(prediction.masks)

    print(f"masks shape: {np.asarray(prediction.masks).shape}")
    print(f"scores: {np.asarray(prediction.scores).tolist()}")
    print(f"logits shape: {np.asarray(prediction.logits).shape}")
    print(f"selected mask pixels: {int(mask.sum())}")

    save_visualization(image=image, mask=mask, box=box, output=output_path)
    print(f"Saved visualization: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
