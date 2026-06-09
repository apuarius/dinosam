import numpy as np


def label_boundary_map(label_mask: np.ndarray) -> np.ndarray:
    """从实例 ID 图中提取边界像素，ID 变化的位置会被视为边界。"""
    labels = np.asarray(label_mask)
    if labels.ndim != 2:
        raise ValueError(f"Label mask must be 2D, got shape: {labels.shape}")

    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, 1:] != labels[:, :-1]
    horizontal &= (labels[:, 1:] > 0) | (labels[:, :-1] > 0)
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal

    vertical = labels[1:, :] != labels[:-1, :]
    vertical &= (labels[1:, :] > 0) | (labels[:-1, :] > 0)
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary


def dilate_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """用纯 numpy 对二值 mask 做方形邻域膨胀，避免额外图像处理依赖。"""
    source = mask.astype(bool, copy=False)
    if radius <= 0:
        return source.copy()

    height, width = source.shape
    padded = np.pad(source, radius, mode="constant", constant_values=False)
    result = np.zeros_like(source, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result |= padded[
                radius + dy : radius + dy + height,
                radius + dx : radius + dx + width,
            ]
    return result
def score_map_auc(score_map: np.ndarray, target: np.ndarray) -> float | None:
    """计算连续边界分数图的 ROC-AUC，正负样本缺失时返回 None。"""
    scores = np.asarray(score_map, dtype=np.float64).ravel()
    labels = target.astype(bool, copy=False).ravel()
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    positive_rank_sum = float(ranks[labels].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def score_map_average_precision(score_map: np.ndarray, target: np.ndarray) -> float | None:
    """计算连续边界分数图的 Average Precision，用于衡量高分区域是否落在 GT 边界上。"""
    scores = np.asarray(score_map, dtype=np.float64).ravel()
    labels = target.astype(bool, copy=False).ravel()
    positives = int(labels.sum())
    if positives == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positives = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1, dtype=np.float64)
    precision_at_hits = true_positives[sorted_labels] / ranks[sorted_labels]
    if len(precision_at_hits) == 0:
        return 0.0
    return float(precision_at_hits.sum() / positives)
