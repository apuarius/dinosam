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


@dataclass(frozen=True)
class InstanceDatasetDirs:
    """保存图像目录与实例 mask 目录。"""

    image_dir: Path
    instance_dir: Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def resolve_instance_dataset_dirs(dataset_root: str | Path) -> InstanceDatasetDirs:
    """解析实例数据集目录，兼容 Image/Instance 与 images/split、masks/split 结构。"""
    root = resolve_project_path(dataset_root)

    if root.is_dir() and root.parent.name.lower() == "images":
        mask_dir = root.parent.parent / "masks" / root.name
        if mask_dir.is_dir():
            return InstanceDatasetDirs(image_dir=root, instance_dir=mask_dir)

    if (root / "Image").is_dir() and (root / "Instance").is_dir():
        return InstanceDatasetDirs(image_dir=root / "Image", instance_dir=root / "Instance")

    if (root / "images").is_dir() and (root / "masks").is_dir():
        return InstanceDatasetDirs(image_dir=root / "images", instance_dir=root / "masks")

    all_root = root / "All"
    if (all_root / "Image").is_dir() and (all_root / "Instance").is_dir():
        return InstanceDatasetDirs(image_dir=all_root / "Image", instance_dir=all_root / "Instance")

    raise FileNotFoundError(
        "Dataset root must contain Image/ and Instance/ directories, or point to data/images/<split>: "
        f"{root}"
    )


def list_instance_tile_pairs(dataset_root: str | Path) -> list[InstanceTilePair]:
    """列出图像与实例 mask 目录中同 stem 的切片配对。"""
    dirs = resolve_instance_dataset_dirs(dataset_root)
    image_dir = dirs.image_dir
    instance_dir = dirs.instance_dir

    masks_by_stem: dict[str, Path] = {}
    for mask_path in sorted(instance_dir.iterdir()):
        if mask_path.is_file() and mask_path.suffix.lower() in MASK_EXTENSIONS:
            masks_by_stem.setdefault(mask_path.stem, mask_path)

    pairs: list[InstanceTilePair] = []
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        instance_path = instance_dir / image_path.name
        if not instance_path.exists():
            instance_path = masks_by_stem.get(image_path.stem)
        if instance_path is not None and instance_path.exists():
            pairs.append(
                InstanceTilePair(
                    image_path=image_path,
                    instance_path=instance_path,
                )
            )

    if not pairs:
        raise FileNotFoundError(f"No paired image/mask tiles found under: {image_dir} and {instance_dir}")
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
