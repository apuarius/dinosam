from typing import Any

import numpy as np


def first_binary_mask(mask_batch: Any) -> np.ndarray:
    """从模型返回的 mask 批次中取出第一张二维布尔 mask。"""
    mask = np.asarray(mask_batch)
    while mask.ndim > 2:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask after squeezing, got shape: {mask.shape}")
    return mask.astype(bool)


def mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    """计算两个二值 mask 的 IoU。"""
    pred = prediction.astype(bool, copy=False)
    truth = target.astype(bool, copy=False)
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)
