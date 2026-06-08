from dataclasses import dataclass
from typing import Literal

import numpy as np

from dinosam.data.instance_tiles import instance_ids


PromptMode = Literal["box", "point", "box_point"]


@dataclass(frozen=True)
class InstancePrompt:
    """保存单个实例转成 SAM2 prompt 所需的 box、正点和面积。"""

    instance_id: int
    box: np.ndarray
    point_coords: np.ndarray
    point_labels: np.ndarray
    area: int


def mask_to_box(mask: np.ndarray, margin: int = 0) -> np.ndarray:
    """把单个二值实例 mask 转换成 SAM2 使用的 xyxy 外接框。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot build a box from an empty mask.")

    height, width = mask.shape
    x0 = max(int(xs.min()) - margin, 0)
    y0 = max(int(ys.min()) - margin, 0)
    x1 = min(int(xs.max()) + margin, width - 1)
    y1 = min(int(ys.max()) + margin, height - 1)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def mask_to_inner_point(mask: np.ndarray) -> np.ndarray:
    """从实例内部选取最靠近质心的正样本点。"""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Cannot build a point from an empty mask.")

    center_x = xs.mean()
    center_y = ys.mean()
    distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
    index = int(np.argmin(distances))
    return np.array([[float(xs[index]), float(ys[index])]], dtype=np.float32)


def prompt_from_binary_mask(
    mask: np.ndarray,
    instance_id: int,
    box_margin: int = 0,
) -> InstancePrompt:
    """把一个二值实例 mask 转换成 box 加正点的基础 prompt。"""
    binary_mask = mask.astype(bool, copy=False)
    point_coords = mask_to_inner_point(binary_mask)
    point_labels = np.array([1], dtype=np.int32)
    return InstancePrompt(
        instance_id=instance_id,
        box=mask_to_box(binary_mask, margin=box_margin),
        point_coords=point_coords,
        point_labels=point_labels,
        area=int(binary_mask.sum()),
    )


def prompts_from_instance_mask(
    instance_mask: np.ndarray,
    min_area: int = 1,
    box_margin: int = 0,
    max_instances: int | None = None,
) -> list[InstancePrompt]:
    """把一张实例 mask 中的每个实例转换成 SAM2 prompt。"""
    prompts: list[InstancePrompt] = []
    for instance_id in instance_ids(instance_mask, min_area=min_area):
        binary_mask = instance_mask == instance_id
        prompts.append(
            prompt_from_binary_mask(
                binary_mask,
                instance_id=instance_id,
                box_margin=box_margin,
            )
        )

        if max_instances is not None and len(prompts) >= max_instances:
            break

    return prompts


def build_sam2_prompt_kwargs(prompt: InstancePrompt, mode: PromptMode) -> dict[str, np.ndarray]:
    """按指定模式把内部 prompt 对象转换成 SAM2 predict 参数。"""
    if mode == "box":
        return {"box": prompt.box}
    if mode == "point":
        return {
            "point_coords": prompt.point_coords,
            "point_labels": prompt.point_labels,
        }
    if mode == "box_point":
        return {
            "box": prompt.box,
            "point_coords": prompt.point_coords,
            "point_labels": prompt.point_labels,
        }
    raise ValueError(f"Unsupported prompt mode: {mode}")
