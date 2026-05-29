from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from dinosam.project import resolve_project_path


@dataclass(frozen=True)
class InstanceTilePair:
    """保存一张遥感切片及其同名实例 mask 的路径。"""

    image_path: Path
    instance_path: Path

    @property
    def name(self) -> str:
        """返回切片文件名，便于输出日志和结果文件。"""
        return self.image_path.name


def resolve_instance_dataset_root(dataset_root: str | Path) -> Path:
    """解析实例数据集根目录，并兼容外层目录下还有 All 子目录的情况。"""
    root = resolve_project_path(dataset_root)
    if (root / "Image").is_dir() and (root / "Instance").is_dir():
        return root

    all_root = root / "All"
    if (all_root / "Image").is_dir() and (all_root / "Instance").is_dir():
        return all_root

    raise FileNotFoundError(
        "Dataset root must contain Image/ and Instance/ directories: "
        f"{root}"
    )


def list_instance_tile_pairs(dataset_root: str | Path) -> list[InstanceTilePair]:
    """列出 Image 与 Instance 目录中同名的 PNG 切片配对。"""
    root = resolve_instance_dataset_root(dataset_root)
    image_dir = root / "Image"
    instance_dir = root / "Instance"

    pairs: list[InstanceTilePair] = []
    for image_path in sorted(image_dir.glob("*.png")):
        instance_path = instance_dir / image_path.name
        if instance_path.exists():
            pairs.append(
                InstanceTilePair(
                    image_path=image_path,
                    instance_path=instance_path,
                )
            )

    if not pairs:
        raise FileNotFoundError(f"No paired PNG tiles found under: {root}")
    return pairs


def load_rgb_image(path: str | Path) -> np.ndarray:
    """读取 RGB 图像，并返回 HWC uint8 数组。"""
    return np.asarray(Image.open(path).convert("RGB"))


def load_instance_mask(path: str | Path) -> np.ndarray:
    """读取实例 mask，并返回保持实例 ID 的二维整数数组。"""
    mask = np.asarray(Image.open(path))
    if mask.ndim != 2:
        raise ValueError(f"Instance mask must be a 2D image: {path}")
    return mask


def instance_ids(mask: np.ndarray, min_area: int = 1) -> list[int]:
    """从实例 mask 中提取面积不小于阈值的非背景实例 ID。"""
    if mask.ndim != 2:
        raise ValueError("Instance mask must be a 2D array.")

    counts = np.bincount(mask.astype(np.int64, copy=False).ravel())
    ids: list[int] = []
    for instance_id, area in enumerate(counts):
        if instance_id == 0:
            continue
        if area >= min_area:
            ids.append(instance_id)
    return ids
