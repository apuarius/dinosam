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
    """计算两个二值 mask 的 IoU，越接近 1 表示重叠越好。"""
    pred = prediction.astype(bool, copy=False)
    truth = target.astype(bool, copy=False)
    intersection = np.logical_and(pred, truth).sum()
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def mask_dice(prediction: np.ndarray, target: np.ndarray) -> float:
    """计算两个二值 mask 的 Dice 系数，对小目标通常比 IoU 更直观。"""
    pred = prediction.astype(bool, copy=False)
    truth = target.astype(bool, copy=False)
    intersection = np.logical_and(pred, truth).sum()
    total = pred.sum() + truth.sum()
    if total == 0:
        return 1.0
    return float(2.0 * intersection / total)


def binary_mask_stats(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """计算二值 mask 的像素级 precision、recall 和 F1。"""
    pred = prediction.astype(bool, copy=False)
    truth = target.astype(bool, copy=False)

    true_positive = float(np.logical_and(pred, truth).sum())
    false_positive = float(np.logical_and(pred, ~truth).sum())
    false_negative = float(np.logical_and(~pred, truth).sum())

    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
