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


def binary_boundary_map(mask: np.ndarray) -> np.ndarray:
    """从单个二值 mask 中提取前景轮廓边界。"""
    return label_boundary_map(mask.astype(np.uint8, copy=False))


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


def boundary_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    tolerance: int = 2,
) -> dict[str, float]:
    """计算带像素容差的边界 precision、recall 和 F1。"""
    pred_boundary = binary_boundary_map(prediction)
    target_boundary = binary_boundary_map(target)

    pred_count = float(pred_boundary.sum())
    target_count = float(target_boundary.sum())
    if pred_count == 0 and target_count == 0:
        return {"boundary_precision": 1.0, "boundary_recall": 1.0, "boundary_f1": 1.0}
    if pred_count == 0 or target_count == 0:
        return {"boundary_precision": 0.0, "boundary_recall": 0.0, "boundary_f1": 0.0}

    target_neighborhood = dilate_binary_mask(target_boundary, radius=tolerance)
    pred_neighborhood = dilate_binary_mask(pred_boundary, radius=tolerance)
    precision = float(np.logical_and(pred_boundary, target_neighborhood).sum()) / pred_count
    recall = float(np.logical_and(target_boundary, pred_neighborhood).sum()) / target_count
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
    }


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


def best_f1_from_score_map(
    score_map: np.ndarray,
    target: np.ndarray,
    quantiles: tuple[float, ...] = (0.70, 0.80, 0.85, 0.90, 0.95),
) -> dict[str, float | None]:
    """在若干分位数阈值中寻找连续分数图的最佳二值 F1。"""
    scores = np.asarray(score_map, dtype=np.float32)
    labels = target.astype(bool, copy=False)
    if labels.sum() == 0:
        return {"best_f1": None, "best_threshold": None, "best_quantile": None}

    best = {"best_f1": -1.0, "best_threshold": None, "best_quantile": None}
    for quantile in quantiles:
        threshold = float(np.quantile(scores, quantile))
        prediction = scores >= threshold
        true_positive = float(np.logical_and(prediction, labels).sum())
        false_positive = float(np.logical_and(prediction, ~labels).sum())
        false_negative = float(np.logical_and(~prediction, labels).sum())
        precision = true_positive / max(true_positive + false_positive, 1.0)
        recall = true_positive / max(true_positive + false_negative, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > float(best["best_f1"]):
            best = {
                "best_f1": f1,
                "best_threshold": threshold,
                "best_quantile": quantile,
            }

    return best


def score_map_metrics(score_map: np.ndarray, target: np.ndarray) -> dict[str, float | int | None]:
    """汇总连续边界分数图与目标边界之间的 patch 级评价指标。"""
    scores = np.asarray(score_map, dtype=np.float32)
    labels = target.astype(bool, copy=False)
    positive_scores = scores[labels]
    negative_scores = scores[~labels]

    mean_positive = float(positive_scores.mean()) if positive_scores.size else None
    mean_negative = float(negative_scores.mean()) if negative_scores.size else None
    contrast = None
    if mean_positive is not None and mean_negative is not None:
        contrast = mean_positive - mean_negative

    metrics: dict[str, float | int | None] = {
        "positive_patches": int(labels.sum()),
        "negative_patches": int(labels.size - labels.sum()),
        "mean_score_on_boundary": mean_positive,
        "mean_score_off_boundary": mean_negative,
        "score_contrast": contrast,
        "roc_auc": score_map_auc(scores, labels),
        "average_precision": score_map_average_precision(scores, labels),
    }
    metrics.update(best_f1_from_score_map(scores, labels))
    return metrics
