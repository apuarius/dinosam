from __future__ import annotations

from typing import Iterable

import numpy as np


PredictionInstance = tuple[np.ndarray, float]


def binary_instances_from_label_mask(label_mask: np.ndarray, min_area: int = 1) -> list[np.ndarray]:
    """Extract non-background instance masks from an integer label mask."""
    labels = np.asarray(label_mask)
    if labels.ndim != 2:
        raise ValueError(f"Instance mask must be 2D, got shape: {labels.shape}")

    instances: list[np.ndarray] = []
    for instance_id in sorted(int(value) for value in np.unique(labels) if int(value) != 0):
        mask = labels == instance_id
        if int(mask.sum()) >= min_area:
            instances.append(mask)
    return instances


def _mask_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction).astype(bool, copy=False)
    truth = np.asarray(target).astype(bool, copy=False)
    intersection = float(np.logical_and(pred, truth).sum())
    union = float(np.logical_or(pred, truth).sum())
    if union <= 0.0:
        return 0.0
    return intersection / union


def _interpolated_average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    recall_points = np.linspace(0.0, 1.0, 101)
    values = []
    for recall_threshold in recall_points:
        matched_precision = precision[recall >= recall_threshold]
        values.append(float(matched_precision.max()) if matched_precision.size else 0.0)
    return float(np.mean(values))


def average_precision_at_iou(
    predictions_by_image: Iterable[list[PredictionInstance]],
    targets_by_image: Iterable[list[np.ndarray]],
    iou_threshold: float,
) -> float:
    """Compute single-class mask AP at one IoU threshold over a dataset."""
    predictions = list(predictions_by_image)
    targets = list(targets_by_image)
    if len(predictions) != len(targets):
        raise ValueError("predictions_by_image and targets_by_image must have the same length.")

    total_targets = sum(len(image_targets) for image_targets in targets)
    flat_predictions: list[tuple[float, int, np.ndarray]] = []
    for image_index, image_predictions in enumerate(predictions):
        for mask, score in image_predictions:
            flat_predictions.append((float(score), image_index, np.asarray(mask).astype(bool, copy=False)))
    flat_predictions.sort(key=lambda item: item[0], reverse=True)

    if total_targets == 0:
        return 1.0 if not flat_predictions else 0.0
    if not flat_predictions:
        return 0.0

    matched = [[False] * len(image_targets) for image_targets in targets]
    true_positive: list[float] = []
    false_positive: list[float] = []
    for _, image_index, prediction_mask in flat_predictions:
        image_targets = targets[image_index]
        best_iou = 0.0
        best_target_index = -1
        for target_index, target_mask in enumerate(image_targets):
            if matched[image_index][target_index]:
                continue
            iou = _mask_iou(prediction_mask, target_mask)
            if iou > best_iou:
                best_iou = iou
                best_target_index = target_index

        if best_target_index >= 0 and best_iou >= iou_threshold:
            matched[image_index][best_target_index] = True
            true_positive.append(1.0)
            false_positive.append(0.0)
        else:
            true_positive.append(0.0)
            false_positive.append(1.0)

    tp_cumsum = np.cumsum(np.asarray(true_positive, dtype=np.float64))
    fp_cumsum = np.cumsum(np.asarray(false_positive, dtype=np.float64))
    recall = tp_cumsum / max(float(total_targets), 1.0)
    precision = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1e-12)
    return _interpolated_average_precision(precision, recall)


def mask_average_precision(
    predictions_by_image: Iterable[list[PredictionInstance]],
    targets_by_image: Iterable[list[np.ndarray]],
    iou_thresholds: Iterable[float] | None = None,
) -> dict[str, float]:
    """Compute table-style single-class mAP metrics for instance masks."""
    thresholds = list(iou_thresholds) if iou_thresholds is not None else [round(0.5 + index * 0.05, 2) for index in range(10)]
    predictions = list(predictions_by_image)
    targets = list(targets_by_image)
    ap_by_threshold = {
        float(threshold): average_precision_at_iou(predictions, targets, float(threshold))
        for threshold in thresholds
    }
    return {
        "mAP@0.5": ap_by_threshold[0.5] if 0.5 in ap_by_threshold else average_precision_at_iou(predictions, targets, 0.5),
        "mAP@0.5:0.95": float(np.mean(list(ap_by_threshold.values()))) if ap_by_threshold else 0.0,
    }
