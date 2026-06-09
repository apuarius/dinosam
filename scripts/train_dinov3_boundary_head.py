import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageEnhance  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from dinosam.config import load_config  # noqa: E402
from dinosam.data import (  # noqa: E402
    InstanceTilePair,
    list_instance_tile_pairs,
    load_instance_mask,
)
from dinosam.evaluation import (  # noqa: E402
    dilate_binary_mask,
    label_boundary_map,
    score_map_average_precision,
    score_map_auc,
)
from dinosam.models import (  # noqa: E402
    DINOv3Wrapper,
    PatchDetectionHead,
    PatchDetectionHeadConfig,
    build_dinov3_config,
    extract_patch_feature_grid,
)
from dinosam.project import resolve_project_path  # noqa: E402


DEFAULT_CONFIG_PATH = "configs/train/dinov3_boundary_head_t1.yaml"


DEFAULT_TRAINING_VALUES: dict[str, Any] = {
    "train_root": "data/images/train",
    "val_root": "data/images/val",
    "test_root": "data/images/test",
    "model_config": "configs/model/dinov3_sam2.yaml",
    "output_dir": "outputs/dinov3_boundary_head_t1",
    "epochs": 100,
    "batch_size": 256,
    "num_workers": 8,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "head_type": "t1",
    "hidden_channels": 256,
    "dropout": 0.1,
    "boundary_loss_weight": 1.0,
    "foreground_loss_weight": 0.5,
    "max_pos_weight": 20.0,
    "target_threshold": 0.05,
    "metric_threshold": "auto",
    "gt_boundary_dilation": 4,
    "limit_train": None,
    "limit_val": None,
    "seed": 42,
    "cache_features": True,
    "feature_cache_dir": None,
    "device": None,
    "amp": True,
    "amp_dtype": "bfloat16",
    "preprocess_in_workers": True,
    "dinov3_input_size": 224,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "resume": False,
    "resume_checkpoint": None,
    "visualize_every": 25,
    "visualize_samples": 8,
    "augment_train": True,
    "augment_factor": 4,
    "hflip_prob": 0.5,
    "vflip_prob": 0.5,
    "rotate90_prob": 0.5,
    "small_rotate_prob": 0.5,
    "small_rotate_degrees": 10.0,
    "scale_prob": 0.5,
    "scale_min": 0.9,
    "scale_max": 1.1,
    "translate_prob": 0.5,
    "translate_fraction": 0.08,
    "brightness_prob": 0.5,
    "brightness_min": 0.8,
    "brightness_max": 1.2,
    "contrast_prob": 0.5,
    "contrast_min": 0.8,
    "contrast_max": 1.2,
    "hsv_prob": 0.5,
    "hue_delta": 0.03,
    "saturation_min": 0.85,
    "saturation_max": 1.15,
    "value_min": 0.85,
    "value_max": 1.15,
    "gamma_prob": 0.5,
    "gamma_min": 0.8,
    "gamma_max": 1.2,
}


CONFIG_SECTIONS: dict[str, tuple[str, ...]] = {
    "data": ("train_root", "val_root", "test_root", "limit_train", "limit_val"),
    "model": ("model_config",),
    "output": ("output_dir",),
    "train": ("epochs", "batch_size", "num_workers", "lr", "weight_decay"),
    "head": ("head_type", "hidden_channels", "dropout"),
    "loss": ("boundary_loss_weight", "foreground_loss_weight", "max_pos_weight"),
    "target": ("target_threshold", "metric_threshold", "gt_boundary_dilation"),
    "runtime": (
        "seed",
        "cache_features",
        "feature_cache_dir",
        "device",
        "amp",
        "amp_dtype",
        "preprocess_in_workers",
        "dinov3_input_size",
        "prefetch_factor",
        "persistent_workers",
        "resume",
        "resume_checkpoint",
        "visualize_every",
        "visualize_samples",
    ),
    "augment": (
        "augment_train",
        "augment_factor",
        "hflip_prob",
        "vflip_prob",
        "rotate90_prob",
        "small_rotate_prob",
        "small_rotate_degrees",
        "scale_prob",
        "scale_min",
        "scale_max",
        "translate_prob",
        "translate_fraction",
        "brightness_prob",
        "brightness_min",
        "brightness_max",
        "contrast_prob",
        "contrast_min",
        "contrast_max",
        "hsv_prob",
        "hue_delta",
        "saturation_min",
        "saturation_max",
        "value_min",
        "value_max",
        "gamma_prob",
        "gamma_min",
        "gamma_max",
    ),
}


@dataclass(frozen=True)
class TrainAugmentConfig:
    """训练集在线增强配置。"""

    enabled: bool = True
    factor: int = 4
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate90_prob: float = 0.5
    small_rotate_prob: float = 0.5
    small_rotate_degrees: float = 10.0
    scale_prob: float = 0.5
    scale_min: float = 0.9
    scale_max: float = 1.1
    translate_prob: float = 0.5
    translate_fraction: float = 0.08
    brightness_prob: float = 0.5
    brightness_min: float = 0.8
    brightness_max: float = 1.2
    contrast_prob: float = 0.5
    contrast_min: float = 0.8
    contrast_max: float = 1.2
    hsv_prob: float = 0.5
    hue_delta: float = 0.03
    saturation_min: float = 0.85
    saturation_max: float = 1.15
    value_min: float = 0.85
    value_max: float = 1.15
    gamma_prob: float = 0.5
    gamma_min: float = 0.8
    gamma_max: float = 1.2


@dataclass(frozen=True)
class DINOv3PreprocessConfig:
    """DataLoader worker 内 DINOv3 输入预处理配置。"""

    enabled: bool = True
    image_size: int = 224


def probability(value: Any) -> float:
    """把概率值限制到 0 到 1。"""
    return max(0.0, min(1.0, float(value)))


def build_train_augment_config(args: argparse.Namespace) -> TrainAugmentConfig | None:
    """从训练参数构建在线增强配置。"""
    enabled = bool(args.augment_train)
    factor = max(1, int(args.augment_factor))
    if not enabled or factor <= 1:
        return None
    return TrainAugmentConfig(
        enabled=enabled,
        factor=factor,
        hflip_prob=probability(args.hflip_prob),
        vflip_prob=probability(args.vflip_prob),
        rotate90_prob=probability(args.rotate90_prob),
        small_rotate_prob=probability(args.small_rotate_prob),
        small_rotate_degrees=float(args.small_rotate_degrees),
        scale_prob=probability(args.scale_prob),
        scale_min=float(args.scale_min),
        scale_max=float(args.scale_max),
        translate_prob=probability(args.translate_prob),
        translate_fraction=max(0.0, float(args.translate_fraction)),
        brightness_prob=probability(args.brightness_prob),
        brightness_min=float(args.brightness_min),
        brightness_max=float(args.brightness_max),
        contrast_prob=probability(args.contrast_prob),
        contrast_min=float(args.contrast_min),
        contrast_max=float(args.contrast_max),
        hsv_prob=probability(args.hsv_prob),
        hue_delta=max(0.0, float(args.hue_delta)),
        saturation_min=float(args.saturation_min),
        saturation_max=float(args.saturation_max),
        value_min=float(args.value_min),
        value_max=float(args.value_max),
        gamma_prob=probability(args.gamma_prob),
        gamma_min=float(args.gamma_min),
        gamma_max=float(args.gamma_max),
    )


def _uniform(min_value: float, max_value: float) -> float:
    """从区间内采样；若配置反向，则自动交换。"""
    low = min(float(min_value), float(max_value))
    high = max(float(min_value), float(max_value))
    return random.uniform(low, high)


def _mask_to_image(mask: np.ndarray) -> Image.Image:
    """把实例 ID mask 转成 PIL 图像，后续几何增强使用最近邻插值。"""
    return Image.fromarray(mask.astype(np.int32, copy=False), mode="I")


def _image_fill_color(image: Image.Image) -> tuple[int, int, int]:
    """用图像均值填充增强后的空白边缘，避免黑边过强。"""
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return tuple(int(value) for value in array.reshape(-1, 3).mean(axis=0))


def _apply_affine_pair(
    image: Image.Image,
    mask_image: Image.Image,
    *,
    angle_degrees: float,
    scale: float,
    translate_x: float,
    translate_y: float,
) -> tuple[Image.Image, Image.Image]:
    """对图像和实例 mask 同步执行小角度旋转、缩放和平移。"""
    if abs(angle_degrees) < 1e-6 and abs(scale - 1.0) < 1e-6 and abs(translate_x) < 1e-6 and abs(translate_y) < 1e-6:
        return image, mask_image

    width, height = image.size
    cx = width * 0.5
    cy = height * 0.5
    safe_scale = max(float(scale), 1e-3)
    radians = math.radians(angle_degrees)
    cos_value = math.cos(radians) / safe_scale
    sin_value = math.sin(radians) / safe_scale
    matrix = (
        cos_value,
        sin_value,
        cx - cos_value * (cx + translate_x) - sin_value * (cy + translate_y),
        -sin_value,
        cos_value,
        cy + sin_value * (cx + translate_x) - cos_value * (cy + translate_y),
    )
    image = image.transform(
        image.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.BILINEAR,
        fillcolor=_image_fill_color(image),
    )
    mask_image = mask_image.transform(
        mask_image.size,
        Image.Transform.AFFINE,
        matrix,
        resample=Image.Resampling.NEAREST,
        fillcolor=0,
    )
    return image, mask_image


def _apply_hsv(image: Image.Image, config: TrainAugmentConfig) -> Image.Image:
    """对 RGB 图像做轻量 HSV 增强。"""
    hsv = np.asarray(image.convert("HSV"), dtype=np.uint8).copy()
    hue_shift = int(round(_uniform(-config.hue_delta, config.hue_delta) * 255.0))
    saturation_factor = _uniform(config.saturation_min, config.saturation_max)
    value_factor = _uniform(config.value_min, config.value_max)
    hsv[..., 0] = ((hsv[..., 0].astype(np.int16) + hue_shift) % 256).astype(np.uint8)
    hsv[..., 1] = np.clip(hsv[..., 1].astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
    hsv[..., 2] = np.clip(hsv[..., 2].astype(np.float32) * value_factor, 0, 255).astype(np.uint8)
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def _apply_gamma(image: Image.Image, config: TrainAugmentConfig) -> Image.Image:
    """对 RGB 图像做 Gamma 增强。"""
    gamma = max(_uniform(config.gamma_min, config.gamma_max), 1e-3)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    adjusted = np.power(np.clip(array, 0.0, 1.0), gamma)
    return Image.fromarray(np.clip(adjusted * 255.0, 0, 255).astype(np.uint8), mode="RGB")


def image_to_dinov3_input_tensor(image: Image.Image, image_size: int) -> torch.Tensor:
    """在 worker 内完成 DINOv3 SAT-493M resize、rescale 和 normalize。"""
    resized = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.430, 0.411, 0.296], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.213, 0.156, 0.143], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def apply_train_augmentation(
    image: Image.Image,
    instance_mask: np.ndarray,
    config: TrainAugmentConfig,
) -> tuple[Image.Image, np.ndarray]:
    """对训练样本随机选择增强策略，并同步更新实例 mask。"""
    image = image.convert("RGB")
    mask_image = _mask_to_image(instance_mask)

    if random.random() < config.hflip_prob:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < config.vflip_prob:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask_image = mask_image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if random.random() < config.rotate90_prob:
        transpose_op = random.choice(
            (
                Image.Transpose.ROTATE_90,
                Image.Transpose.ROTATE_180,
                Image.Transpose.ROTATE_270,
            )
        )
        image = image.transpose(transpose_op)
        mask_image = mask_image.transpose(transpose_op)

    angle = _uniform(-config.small_rotate_degrees, config.small_rotate_degrees) if random.random() < config.small_rotate_prob else 0.0
    scale = _uniform(config.scale_min, config.scale_max) if random.random() < config.scale_prob else 1.0
    translate_x = 0.0
    translate_y = 0.0
    if random.random() < config.translate_prob:
        width, height = image.size
        translate_x = _uniform(-config.translate_fraction, config.translate_fraction) * width
        translate_y = _uniform(-config.translate_fraction, config.translate_fraction) * height
    image, mask_image = _apply_affine_pair(
        image,
        mask_image,
        angle_degrees=angle,
        scale=scale,
        translate_x=translate_x,
        translate_y=translate_y,
    )

    if random.random() < config.brightness_prob:
        image = ImageEnhance.Brightness(image).enhance(_uniform(config.brightness_min, config.brightness_max))
    if random.random() < config.contrast_prob:
        image = ImageEnhance.Contrast(image).enhance(_uniform(config.contrast_min, config.contrast_max))
    if random.random() < config.hsv_prob:
        image = _apply_hsv(image, config)
    if random.random() < config.gamma_prob:
        image = _apply_gamma(image, config)

    return image, np.asarray(mask_image, dtype=np.int32)


class InstanceTileDataset(Dataset):
    """读取遥感切片和同名实例 mask，供检测头训练使用。"""

    def __init__(
        self,
        pairs: list[InstanceTilePair],
        augment_config: TrainAugmentConfig | None = None,
        preprocess_config: DINOv3PreprocessConfig | None = None,
        keep_image: bool = True,
    ) -> None:
        """保存已经配对好的 Image/Instance 文件路径列表。"""
        self.pairs = pairs
        self.augment_config = augment_config
        self.preprocess_config = preprocess_config
        self.keep_image = keep_image

    def __len__(self) -> int:
        """返回数据集中的切片数量。"""
        return len(self.pairs) * (self.augment_config.factor if self.augment_config is not None else 1)

    @property
    def base_size(self) -> int:
        """返回未增强前的原始切片数量。"""
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取单个样本，并复制数组以避免只读 numpy 警告。"""
        pair = self.pairs[index % len(self.pairs)]
        image = Image.open(pair.image_path).convert("RGB")
        instance_mask = load_instance_mask(pair.instance_path).copy()
        if self.augment_config is not None:
            image, instance_mask = apply_train_augmentation(image, instance_mask, self.augment_config)
        item = {
            "instance_mask": instance_mask,
            "name": pair.name,
        }
        if self.preprocess_config is not None and self.preprocess_config.enabled:
            item["dinov3_input"] = image_to_dinov3_input_tensor(
                image,
                image_size=self.preprocess_config.image_size,
            )
        if self.keep_image:
            item["image"] = image
        return item


class BinaryMetricAccumulator:
    """累积 patch 级二分类指标，支持进度条实时显示。"""

    def __init__(self, target_threshold: float, metric_threshold: float | str) -> None:
        """初始化计数器和用于 AP/AUC 的分数缓存。"""
        self.target_threshold = target_threshold
        self.metric_threshold = metric_threshold
        self.fast_prediction_threshold = 0.5
        self.true_positive = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0
        self.intersection = 0.0
        self.union = 0.0
        self.scores: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """用一个 batch 的 logits 和 soft target 更新指标。"""
        probabilities = torch.sigmoid(logits).detach().float().cpu().numpy()
        target_values = targets.detach().float().cpu().numpy()
        target_binary = target_values > self.target_threshold
        prediction_binary = probabilities >= self.fast_prediction_threshold

        self.true_positive += float(np.logical_and(prediction_binary, target_binary).sum())
        self.false_positive += float(np.logical_and(prediction_binary, ~target_binary).sum())
        self.false_negative += float(np.logical_and(~prediction_binary, target_binary).sum())
        self.intersection += float(np.logical_and(prediction_binary, target_binary).sum())
        self.union += float(np.logical_or(prediction_binary, target_binary).sum())
        self.scores.append(probabilities.reshape(-1))
        self.targets.append(target_binary.reshape(-1))

    def compute_fast(self) -> dict[str, float]:
        """计算固定 0.5 阈值下的实时 precision、recall、F1 和 IoU。"""
        precision = self.true_positive / max(self.true_positive + self.false_positive, 1.0)
        recall = self.true_positive / max(self.true_positive + self.false_negative, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        iou = self.intersection / max(self.union, 1.0)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    def compute(self) -> dict[str, float | None]:
        """计算完整指标，包括 AP、ROC-AUC 和可选自适应阈值 F1。"""
        fixed_metrics = self.compute_fast()
        metrics: dict[str, float | None] = dict(fixed_metrics)
        if not self.scores:
            metrics.update(
                {
                    "ap": None,
                    "auc": None,
                    "best_f1": None,
                    "best_precision": None,
                    "best_recall": None,
                    "best_threshold": None,
                    "selected_threshold": None,
                }
            )
            return metrics

        scores = np.concatenate(self.scores)
        targets = np.concatenate(self.targets)
        best_metrics = best_binary_f1(scores, targets)
        metrics.update(
            {
                "ap": score_map_average_precision(scores, targets),
                "auc": score_map_auc(scores, targets),
            }
        )
        metrics.update(best_metrics)
        selected_threshold = resolve_metric_threshold(self.metric_threshold, best_metrics)
        selected_metrics = binary_stats_at_threshold(scores, targets, selected_threshold)
        metrics.update(selected_metrics)
        metrics.update({f"selected_{key}": value for key, value in selected_metrics.items()})
        metrics["selected_threshold"] = selected_threshold
        metrics.update({f"fixed_{key}": value for key, value in fixed_metrics.items()})
        return metrics


def best_binary_f1(scores: np.ndarray, targets: np.ndarray) -> dict[str, float | None]:
    """在所有预测分数阈值上搜索最佳 F1 和对应阈值。"""
    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    flat_targets = np.asarray(targets, dtype=bool).reshape(-1)
    positives = int(flat_targets.sum())
    if positives == 0 or flat_scores.size == 0:
        return {
            "best_f1": None,
            "best_precision": None,
            "best_recall": None,
            "best_threshold": None,
        }

    order = np.argsort(-flat_scores, kind="mergesort")
    sorted_targets = flat_targets[order]
    sorted_scores = flat_scores[order]
    true_positive = np.cumsum(sorted_targets)
    predicted_positive = np.arange(1, sorted_targets.size + 1, dtype=np.float64)
    precision = true_positive / predicted_positive
    recall = true_positive / max(float(positives), 1.0)
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_index = int(np.argmax(f1))
    return {
        "best_f1": float(f1[best_index]),
        "best_precision": float(precision[best_index]),
        "best_recall": float(recall[best_index]),
        "best_threshold": float(sorted_scores[best_index]),
    }


def parse_metric_threshold(value: Any) -> float | str:
    """解析评估阈值配置，支持 auto 或 0-1 数值。"""
    if value is None:
        return "auto"
    if isinstance(value, str):
        text = value.strip().lower()
        if text == "auto":
            return "auto"
        value = float(text)
    threshold = float(value)
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"metric_threshold must be 'auto' or a value in [0, 1], got: {value}")
    return threshold


def resolve_metric_threshold(
    metric_threshold: float | str,
    best_metrics: Mapping[str, float | None],
) -> float | None:
    """把 auto 阈值解析成当前 epoch 搜索到的 best_threshold。"""
    if metric_threshold == "auto":
        value = best_metrics.get("best_threshold")
        return None if value is None else float(value)
    return float(metric_threshold)


def binary_stats_at_threshold(
    scores: np.ndarray,
    targets: np.ndarray,
    threshold: float | None,
) -> dict[str, float | None]:
    """按指定预测阈值计算 precision、recall、F1 和 IoU。"""
    if threshold is None:
        return {
            "precision": None,
            "recall": None,
            "f1": None,
            "iou": None,
        }

    flat_scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    flat_targets = np.asarray(targets, dtype=bool).reshape(-1)
    prediction = flat_scores >= threshold
    true_positive = float(np.logical_and(prediction, flat_targets).sum())
    false_positive = float(np.logical_and(prediction, ~flat_targets).sum())
    false_negative = float(np.logical_and(~prediction, flat_targets).sum())
    union = float(np.logical_or(prediction, flat_targets).sum())
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    iou = true_positive / max(union, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
    }


def build_parser() -> argparse.ArgumentParser:
    """构建 DINOv3 patch 检测头训练脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a small boundary head on frozen DINOv3 features.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--train-root", default=None)
    parser.add_argument("--val-root", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--head-type", choices=("t1",), default=None)
    parser.add_argument("--hidden-channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--boundary-loss-weight", type=float, default=None)
    parser.add_argument("--foreground-loss-weight", type=float, default=None)
    parser.add_argument("--max-pos-weight", type=float, default=None)
    parser.add_argument("--target-threshold", type=float, default=None)
    parser.add_argument("--metric-threshold", default=None, help="Use 'auto' or a fixed probability threshold.")
    parser.add_argument("--gt-boundary-dilation", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp-dtype", choices=("float16", "bfloat16"), default=None)
    parser.add_argument("--preprocess-in-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dinov3-input-size", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--visualize-every", type=int, default=None)
    parser.add_argument("--visualize-samples", type=int, default=None)
    parser.add_argument("--augment-train", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--augment-factor", type=int, default=None)
    return parser


def _update_from_section(values: dict[str, Any], section: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    """从配置文件的一个 section 中复制脚本支持的键。"""
    for key in keys:
        if key in section:
            values[key] = section[key]


def load_training_values(config_path: Path) -> dict[str, Any]:
    """读取训练配置文件，并和脚本内置默认值合并。"""
    values = dict(DEFAULT_TRAINING_VALUES)
    if not config_path.exists():
        raise FileNotFoundError(f"Training config file does not exist: {config_path}")

    config = load_config(config_path)
    for key in DEFAULT_TRAINING_VALUES:
        if key in config:
            values[key] = config[key]

    for section_name, keys in CONFIG_SECTIONS.items():
        section = config.get(section_name, {})
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise TypeError(f"Training config section must be a mapping: {section_name}")
        _update_from_section(values, section, keys)
    return values


def parse_training_args() -> argparse.Namespace:
    """合并配置文件和命令行覆盖项，返回训练主流程使用的参数对象。"""
    parser = build_parser()
    cli_args = parser.parse_args()
    config_path = resolve_project_path(cli_args.config)
    values = load_training_values(config_path)

    for key in DEFAULT_TRAINING_VALUES:
        override = getattr(cli_args, key, None)
        if override is not None:
            values[key] = override

    values["metric_threshold"] = parse_metric_threshold(values.get("metric_threshold"))
    values["config"] = str(config_path)
    return argparse.Namespace(**values)


def collate_tiles(items: list[dict[str, Any]]) -> dict[str, Any]:
    """把 DataLoader 读出的样本整理成图片列表、mask 列表和文件名列表。"""
    batch = {
        "instance_masks": [item["instance_mask"] for item in items],
        "names": [item["name"] for item in items],
    }
    if items and "image" in items[0]:
        batch["images"] = [item["image"] for item in items]
    if items and "dinov3_input" in items[0]:
        batch["dinov3_inputs"] = torch.stack([item["dinov3_input"] for item in items], dim=0)
    return batch


def limit_pairs(pairs: list[InstanceTilePair], limit: int | None) -> list[InstanceTilePair]:
    """根据命令行 limit 截断样本列表，便于先做短跑调试。"""
    if limit is None:
        return pairs
    return pairs[:limit]


def seed_worker(worker_id: int) -> None:
    """让 DataLoader worker 内的随机增强可以随全局 seed 复现。"""
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def resolve_amp_dtype(value: str) -> torch.dtype:
    """解析 AMP dtype 配置。"""
    text = str(value).strip().lower()
    if text in {"float16", "fp16", "half"}:
        return torch.float16
    if text in {"bfloat16", "bf16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported amp_dtype: {value}")


def make_grad_scaler(device: torch.device, amp_enabled: bool, amp_dtype: torch.dtype) -> Any | None:
    """仅在 CUDA fp16 AMP 下创建 GradScaler；bf16 不需要缩放。"""
    if not amp_enabled or device.type != "cuda" or amp_dtype != torch.float16:
        return None
    try:
        return torch.amp.GradScaler("cuda", enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def make_loader(
    dataset_root: str,
    limit: int | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    augment_config: TrainAugmentConfig | None = None,
    preprocess_config: DINOv3PreprocessConfig | None = None,
    prefetch_factor: int | None = None,
    persistent_workers: bool = False,
    keep_image: bool = True,
) -> DataLoader:
    """根据数据集根目录构建 PyTorch DataLoader。"""
    pairs = limit_pairs(list_instance_tile_pairs(dataset_root), limit)
    dataset = InstanceTileDataset(
        pairs,
        augment_config=augment_config,
        preprocess_config=preprocess_config,
        keep_image=keep_image,
    )
    loader_kwargs: dict[str, Any] = {}
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor or 2))
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tiles,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker if num_workers > 0 else None,
        **loader_kwargs,
    )


def resize_soft_mask(mask: np.ndarray, patch_shape: tuple[int, int]) -> np.ndarray:
    """把像素级二值图平均池化到 DINOv3 patch 网格大小。"""
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    resized = image.resize((patch_shape[1], patch_shape[0]), Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.float32)


def build_patch_targets(
    instance_masks: list[np.ndarray],
    patch_shape: tuple[int, int],
    device: torch.device,
    boundary_dilation: int,
) -> torch.Tensor:
    """从实例 mask 构建 boundary 和 foreground 两个 patch 级训练目标。"""
    targets: list[np.ndarray] = []
    for instance_mask in instance_masks:
        foreground = instance_mask > 0
        boundary = label_boundary_map(instance_mask)
        boundary = dilate_binary_mask(boundary, radius=boundary_dilation)
        target = np.stack(
            [
                resize_soft_mask(boundary, patch_shape),
                resize_soft_mask(foreground, patch_shape),
            ],
            axis=0,
        )
        targets.append(target)
    array = np.stack(targets, axis=0)
    return torch.from_numpy(array).float().to(device)


def balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_threshold: float,
    max_pos_weight: float,
) -> torch.Tensor:
    """计算带正样本平衡权重的 BCE loss，缓解边界 patch 稀疏问题。"""
    raw_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets > target_threshold
    positive_count = positive.sum().float()
    negative_count = positive.numel() - positive_count
    pos_weight = torch.clamp(negative_count / torch.clamp_min(positive_count, 1.0), max=max_pos_weight)
    weights = torch.where(positive, torch.full_like(targets, pos_weight), torch.ones_like(targets))
    return (raw_loss * weights).sum() / torch.clamp_min(weights.sum(), 1.0)


def candidate_token_tensor(raw_features: Any) -> torch.Tensor:
    """从 DINOv3 输出对象中取出最可能的 patch token 张量。"""
    if hasattr(raw_features, "last_hidden_state"):
        return raw_features.last_hidden_state
    if isinstance(raw_features, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state"):
            value = raw_features.get(key)
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(raw_features):
        return raw_features
    raise TypeError(f"Cannot find token tensor in DINOv3 output: {type(raw_features).__name__}")


def infer_special_token_count(token_count: int) -> int:
    """根据 token 数推断开头需要跳过的 CLS/register token 数量。"""
    for special_tokens in (5, 1, 0, 4, 2, 3, 6, 7, 8):
        patch_count = token_count - special_tokens
        grid_size = int(math.sqrt(patch_count))
        if patch_count > 0 and grid_size * grid_size == patch_count:
            return special_tokens
    raise ValueError(f"Cannot infer square patch grid from token count: {token_count}")


def extract_patch_feature_tensor(raw_features: Any) -> torch.Tensor:
    """把 DINOv3 输出转换成 BCHW 格式的 patch 特征张量。"""
    tokens = candidate_token_tensor(raw_features)
    if tokens.ndim == 4:
        if tokens.shape[-1] > tokens.shape[1]:
            return tokens.permute(0, 3, 1, 2).contiguous().float()
        return tokens.float()

    if tokens.ndim != 3:
        grid = extract_patch_feature_grid(raw_features)
        array = np.moveaxis(grid, -1, 0)[None, ...]
        return torch.from_numpy(array).float()

    special_tokens = infer_special_token_count(tokens.shape[1])
    patch_tokens = tokens[:, special_tokens:, :]
    grid_size = int(math.sqrt(patch_tokens.shape[1]))
    return patch_tokens.reshape(tokens.shape[0], grid_size, grid_size, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous().float()


def feature_cache_paths(cache_dir: Path, split: str, names: list[str]) -> list[Path]:
    """为一个 batch 的切片生成对应的特征缓存路径。"""
    split_dir = cache_dir / split
    return [split_dir / f"{Path(name).stem}.pt" for name in names]


def load_or_extract_features(
    dinov3: DINOv3Wrapper,
    images: list[Image.Image] | None,
    names: list[str],
    split: str,
    cache_dir: Path,
    cache_features: bool,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype | None = None,
    dinov3_inputs: torch.Tensor | None = None,
) -> torch.Tensor:
    """优先读取 DINOv3 特征缓存，不存在时运行冻结编码器并写入缓存。"""
    paths = feature_cache_paths(cache_dir, split, names)
    if cache_features and all(path.exists() for path in paths):
        features = [torch.load(path, map_location=device).float() for path in paths]
        return torch.stack(features, dim=0)

    autocast_dtype = amp_dtype or torch.bfloat16
    with torch.no_grad(), torch.amp.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=bool(amp_enabled and device.type == "cuda"),
    ):
        if dinov3_inputs is None:
            if images is None:
                raise ValueError("images must be provided when dinov3_inputs is not available.")
            if dinov3.processor is None:
                raise RuntimeError("Boundary head training currently expects a Hugging Face DINOv3 model.")
            inputs = dinov3.prepare_inputs(images)
        else:
            inputs = {"pixel_values": dinov3_inputs.to(device, non_blocking=True)}
        raw_features = dinov3(inputs).raw
        features = extract_patch_feature_tensor(raw_features).to(device)

    if cache_features:
        for path, feature in zip(paths, features, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feature.detach().cpu().half(), path)
    return features.float()


def metric_prefix(prefix: str, metrics: dict[str, float | None]) -> dict[str, float | None]:
    """给一组指标统一添加前缀，便于区分 boundary 和 foreground。"""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def format_metric(value: float | None, digits: int = 4) -> str:
    """把可能为空的指标格式化成适合终端显示的字符串。"""
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


def resize_probability_map(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """把 patch 级概率图缩放到原图宽高，便于保存可视化结果。"""
    image = Image.fromarray(value.astype(np.float32), mode="F")
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def colorize_probability_map(value: np.ndarray) -> np.ndarray:
    """把 0 到 1 的概率图转换成蓝绿黄红伪彩色图。"""
    values = np.clip(value, 0.0, 1.0).astype(np.float32)
    anchors = np.asarray(
        [
            [8, 24, 64],
            [0, 112, 192],
            [72, 184, 112],
            [248, 220, 96],
            [232, 64, 48],
        ],
        dtype=np.float32,
    )
    scaled = values * (len(anchors) - 1)
    left = np.floor(scaled).astype(np.int32)
    right = np.clip(left + 1, 0, len(anchors) - 1)
    weight = (scaled - left)[..., None]
    colors = anchors[left] * (1.0 - weight) + anchors[right] * weight
    return np.clip(colors, 0, 255).astype(np.uint8)


def gt_boundary_overlay(image: np.ndarray, instance_mask: np.ndarray) -> np.ndarray:
    """把 GT 实例边界用绿色叠加到原始图像上。"""
    overlay = image.copy()
    boundary = dilate_binary_mask(label_boundary_map(instance_mask), radius=1)
    overlay[boundary] = np.asarray([0, 255, 120], dtype=np.uint8)
    return overlay


def save_prediction_panel(
    image: Image.Image,
    instance_mask: np.ndarray,
    boundary_probability: np.ndarray,
    foreground_probability: np.ndarray,
    output_path: Path,
) -> None:
    """保存原图、GT 边界、预测边界概率和预测田块概率的对比图。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_array = np.asarray(image.convert("RGB"))
    image_size = image.size
    boundary_map = resize_probability_map(boundary_probability, image_size)
    foreground_map = resize_probability_map(foreground_probability, image_size)
    panels = [
        Image.fromarray(image_array),
        Image.fromarray(gt_boundary_overlay(image_array, instance_mask)),
        Image.fromarray(colorize_probability_map(boundary_map)),
        Image.fromarray(colorize_probability_map(foreground_map)),
    ]
    width, height = panels[0].size
    canvas = Image.new("RGB", (width * len(panels), height))
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * width, 0))
    canvas.save(output_path)


def save_validation_visualizations(
    *,
    epoch: int,
    loader: DataLoader,
    dinov3: DINOv3Wrapper,
    head: torch.nn.Module,
    cache_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """在验证集上保存少量预测可视化，帮助判断检测头学到的空间模式。"""
    if args.visualize_every is None or args.visualize_every <= 0:
        return
    if args.visualize_samples is None or args.visualize_samples <= 0:
        return
    if epoch % args.visualize_every != 0 and epoch != args.epochs:
        return

    head.eval()
    saved = 0
    visual_dir = output_dir / "visualizations" / f"epoch_{epoch:04d}"
    with torch.no_grad():
        for batch in loader:
            images = batch["images"]
            names = batch["names"]
            instance_masks = batch["instance_masks"]
            dinov3_inputs = batch.get("dinov3_inputs")
            features = load_or_extract_features(
                dinov3=dinov3,
                images=images,
                names=names,
                split="val",
                cache_dir=cache_dir,
                cache_features=args.cache_features,
                device=device,
                amp_enabled=args.amp,
                amp_dtype=resolve_amp_dtype(args.amp_dtype),
                dinov3_inputs=dinov3_inputs,
            )
            probabilities = torch.sigmoid(head(features)).detach().float().cpu().numpy()
            for index, name in enumerate(names):
                output_path = visual_dir / f"{Path(name).stem}.png"
                save_prediction_panel(
                    image=images[index],
                    instance_mask=instance_masks[index],
                    boundary_probability=probabilities[index, 0],
                    foreground_probability=probabilities[index, 1],
                    output_path=output_path,
                )
                saved += 1
                if saved >= args.visualize_samples:
                    print(f"Saved visualizations: {visual_dir}")
                    return


def run_epoch(
    *,
    epoch: int,
    epochs: int,
    split: str,
    loader: DataLoader,
    dinov3: DINOv3Wrapper,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float | int | None]:
    """运行一个训练或验证 epoch，并返回 YOLO 风格的聚合指标。"""
    training = optimizer is not None
    head.train(training)

    total_images = 0
    total_loss = 0.0
    total_boundary_loss = 0.0
    total_foreground_loss = 0.0
    boundary_metrics = BinaryMetricAccumulator(
        target_threshold=args.target_threshold,
        metric_threshold=args.metric_threshold,
    )
    foreground_metrics = BinaryMetricAccumulator(
        target_threshold=args.target_threshold,
        metric_threshold=args.metric_threshold,
    )

    bar = tqdm(
        loader,
        desc=f"{split} {epoch}/{epochs}",
        ncols=88,
        leave=False,
        ascii=True,
        bar_format="{l_bar}{bar:14}{r_bar}",
    )
    batch_cache_features = args.cache_features and not (training and args.augment_train and args.augment_factor > 1)
    for batch in bar:
        images = batch.get("images")
        names = batch["names"]
        instance_masks = batch["instance_masks"]
        dinov3_inputs = batch.get("dinov3_inputs")
        features = load_or_extract_features(
            dinov3=dinov3,
            images=images,
            names=names,
            split=split,
            cache_dir=cache_dir,
            cache_features=batch_cache_features,
            device=device,
            amp_enabled=args.amp,
            amp_dtype=resolve_amp_dtype(args.amp_dtype),
            dinov3_inputs=dinov3_inputs,
        )
        targets = build_patch_targets(
            instance_masks,
            patch_shape=tuple(features.shape[-2:]),
            device=device,
            boundary_dilation=args.gt_boundary_dilation,
        )

        amp_dtype = resolve_amp_dtype(args.amp_dtype)
        amp_enabled = bool(args.amp and device.type == "cuda")
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = head(features)
            boundary_loss = balanced_bce_with_logits(
                logits[:, 0],
                targets[:, 0],
                target_threshold=args.target_threshold,
                max_pos_weight=args.max_pos_weight,
            )
            foreground_loss = balanced_bce_with_logits(
                logits[:, 1],
                targets[:, 1],
                target_threshold=args.target_threshold,
                max_pos_weight=args.max_pos_weight,
            )
            loss = args.boundary_loss_weight * boundary_loss + args.foreground_loss_weight * foreground_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        batch_size = len(names)
        total_images += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_boundary_loss += float(boundary_loss.detach()) * batch_size
        total_foreground_loss += float(foreground_loss.detach()) * batch_size
        boundary_metrics.update(logits[:, 0], targets[:, 0])
        foreground_metrics.update(logits[:, 1], targets[:, 1])

        fast_boundary = boundary_metrics.compute_fast()
        fast_foreground = foreground_metrics.compute_fast()
        bar.set_postfix(
            {
                "loss": format_metric(total_loss / max(total_images, 1), digits=3),
                "bF1.5": format_metric(fast_boundary["f1"], digits=3),
                "fgIoU.5": format_metric(fast_foreground["iou"], digits=3),
            }
        )

    boundary = metric_prefix("boundary", boundary_metrics.compute())
    foreground = metric_prefix("foreground", foreground_metrics.compute())
    metrics: dict[str, float | int | None] = {
        "images": total_images,
        "loss": total_loss / max(total_images, 1),
        "boundary_loss": total_boundary_loss / max(total_images, 1),
        "foreground_loss": total_foreground_loss / max(total_images, 1),
    }
    metrics.update(boundary)
    metrics.update(foreground)
    return metrics


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    """把一个 epoch 的指标追加写入 CSV，方便后续用表格或画图查看。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
        if existing_fieldnames and existing_fieldnames != fieldnames:
            merged_fieldnames = existing_fieldnames + [
                key for key in fieldnames if key not in existing_fieldnames
            ]
            rows.append({key: row.get(key, "") for key in merged_fieldnames})
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists or path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    """把一个 epoch 的完整指标追加写入 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """保存检测头权重、优化器状态和当前 epoch 指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def checkpoint_score(metrics: Mapping[str, Any] | None) -> float:
    """从 checkpoint 指标中取出用于选择 best.pt 的验证分数。"""
    if not metrics:
        return -1.0
    value = metrics.get("val_boundary_best_f1")
    if value is None:
        value = metrics.get("val_boundary_f1")
    return float(value or -1.0)


def load_checkpoint_score(path: Path) -> float:
    """读取 checkpoint 中的 best 评价分数，失败时返回 -1。"""
    if not path.exists():
        return -1.0
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        return -1.0
    return checkpoint_score(checkpoint.get("metrics"))


def resolve_resume_checkpoint(args: argparse.Namespace, output_dir: Path) -> Path | None:
    """根据配置决定是否从 last.pt 继续训练。"""
    if not args.resume:
        return None
    if args.resume_checkpoint:
        path = resolve_project_path(args.resume_checkpoint)
    else:
        path = output_dir / "checkpoints" / "last.pt"
    return path if path.exists() else None


def resume_from_checkpoint(
    checkpoint_path: Path,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float]:
    """加载检测头和优化器状态，并返回下一轮 epoch 与历史验证分数。"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    head.load_state_dict(checkpoint["head_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    completed_epoch = int(checkpoint.get("epoch", 0))
    best_score = checkpoint_score(checkpoint.get("metrics"))
    print(f"Resumed checkpoint: {checkpoint_path}")
    print(f"Completed epoch: {completed_epoch}; next epoch: {completed_epoch + 1}")
    return completed_epoch + 1, best_score


def prefixed_metrics(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    """给 train 或 val 指标添加统一前缀。"""
    return {f"{prefix}_{key}": value for key, value in values.items()}


def print_metric_header() -> None:
    """打印类似 YOLO 的紧凑指标表头。"""
    print(" epoch   loss(train/val)   bP      bR      bF1     bAP     fgIoU   thr     save")
    print("         b*=boundary best-threshold metrics, fgIoU=foreground selected-threshold")


def print_epoch_summary(
    *,
    epoch: int,
    epochs: int,
    train_metrics: Mapping[str, Any],
    val_metrics: Mapping[str, Any],
    is_best: bool,
) -> None:
    """按固定列宽打印一行核心验证指标，避免终端输出过长。"""
    print(
        f"{epoch:>4}/{epochs:<4} "
        f"{train_metrics['loss']:.4f}/{val_metrics['loss']:.4f}     "
        f"{format_metric(val_metrics.get('boundary_best_precision'), digits=3):>6} "
        f"{format_metric(val_metrics.get('boundary_best_recall'), digits=3):>6} "
        f"{format_metric(val_metrics.get('boundary_best_f1'), digits=3):>6} "
        f"{format_metric(val_metrics.get('boundary_ap'), digits=3):>6} "
        f"{format_metric(val_metrics.get('foreground_selected_iou'), digits=3):>7} "
        f"{format_metric(val_metrics.get('boundary_best_threshold'), digits=3):>6} "
        f"{'*' if is_best else ''}"
    )


def main() -> int:
    """训练 DINOv3 patch 检测头，并保存进度指标和最佳权重。"""
    args = parse_training_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = resolve_project_path(args.output_dir)
    cache_dir = resolve_project_path(args.feature_cache_dir) if args.feature_cache_dir else output_dir / "feature_cache"
    metrics_csv = output_dir / "metrics.csv"
    metrics_jsonl = output_dir / "metrics.jsonl"

    model_config = load_config(resolve_project_path(args.model_config))
    dinov3_config = build_dinov3_config(model_config)
    dinov3 = DINOv3Wrapper.from_config(dinov3_config)
    input_channels = int(getattr(getattr(dinov3.model, "config", None), "hidden_size", 1024))
    head = PatchDetectionHead.build(
        PatchDetectionHeadConfig(
            input_channels=input_channels,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            output_channels=2,
            head_type=args.head_type,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = make_grad_scaler(device, amp_enabled=amp_enabled, amp_dtype=amp_dtype)
    train_augment_config = build_train_augment_config(args)
    model_image_size = int(getattr(getattr(dinov3.model, "config", None), "image_size", 224))
    dinov3_input_size = int(args.dinov3_input_size or model_image_size)
    preprocess_config = (
        DINOv3PreprocessConfig(enabled=True, image_size=dinov3_input_size)
        if bool(args.preprocess_in_workers)
        else None
    )
    keep_train_images = preprocess_config is None

    train_loader = make_loader(
        args.train_root,
        limit=args.limit_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        augment_config=train_augment_config,
        preprocess_config=preprocess_config,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        keep_image=keep_train_images,
    )
    val_loader = make_loader(
        args.val_root,
        limit=args.limit_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        augment_config=None,
        preprocess_config=preprocess_config,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        keep_image=True,
    )

    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Train tiles: {train_loader.dataset.base_size}")
    print(f"Train samples/epoch: {len(train_loader.dataset)}")
    print(f"Val images: {len(val_loader.dataset)}")
    if train_augment_config is not None:
        print(f"Train augmentation: on, factor={train_augment_config.factor}; train feature cache disabled")
    else:
        print("Train augmentation: off")
    print(f"Head type: {args.head_type}")
    print(f"Metric threshold: {args.metric_threshold}")
    print(f"AMP: {'on' if amp_enabled else 'off'} ({args.amp_dtype})")
    print(
        "DataLoader: "
        f"batch={args.batch_size}, workers={args.num_workers}, "
        f"prefetch_factor={args.prefetch_factor}, persistent_workers={args.persistent_workers}"
    )
    print(
        "DINOv3 preprocessing: "
        f"{'workers' if preprocess_config is not None else 'main process/HF processor'}, "
        f"input_size={dinov3_input_size}"
    )
    print(f"Train image transfer: {'on' if keep_train_images else 'off'}")
    print(f"Feature cache: {'on' if args.cache_features else 'off'} -> {cache_dir}")
    print(f"Output dir: {output_dir}")

    best_metric = load_checkpoint_score(output_dir / "checkpoints" / "best.pt")
    start_epoch = 1
    resume_checkpoint = resolve_resume_checkpoint(args, output_dir)
    if resume_checkpoint is not None:
        start_epoch, resume_score = resume_from_checkpoint(
            checkpoint_path=resume_checkpoint,
            head=head,
            optimizer=optimizer,
            device=device,
        )
        best_metric = max(best_metric, resume_score)

    if start_epoch > args.epochs:
        print(f"Training is already complete: last epoch >= configured epochs ({args.epochs}).")
        return 0

    start_time = time.time()
    print_metric_header()
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = run_epoch(
                epoch=epoch,
                epochs=args.epochs,
                split="train",
                loader=train_loader,
                dinov3=dinov3,
                head=head,
                optimizer=optimizer,
                cache_dir=cache_dir,
                args=args,
                device=device,
                scaler=scaler,
            )
            val_metrics = run_epoch(
                epoch=epoch,
                epochs=args.epochs,
                split="val",
                loader=val_loader,
                dinov3=dinov3,
                head=head,
                optimizer=None,
                cache_dir=cache_dir,
                args=args,
                device=device,
                scaler=None,
            )
            save_validation_visualizations(
                epoch=epoch,
                loader=val_loader,
                dinov3=dinov3,
                head=head,
                cache_dir=cache_dir,
                output_dir=output_dir,
                args=args,
                device=device,
            )

            epoch_row: dict[str, Any] = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "elapsed_min": (time.time() - start_time) / 60.0,
            }
            epoch_row.update(prefixed_metrics("train", train_metrics))
            epoch_row.update(prefixed_metrics("val", val_metrics))

            write_csv_row(metrics_csv, epoch_row)
            write_jsonl_row(metrics_jsonl, epoch_row)
            save_checkpoint(
                output_dir / "checkpoints" / "last.pt",
                epoch=epoch,
                head=head,
                optimizer=optimizer,
                metrics=epoch_row,
                args=args,
            )

            score = float(val_metrics.get("boundary_best_f1") or val_metrics.get("boundary_f1") or 0.0)
            is_best = score > best_metric
            if is_best:
                best_metric = score
                save_checkpoint(
                    output_dir / "checkpoints" / "best.pt",
                    epoch=epoch,
                    head=head,
                    optimizer=optimizer,
                    metrics=epoch_row,
                    args=args,
                )

            print_epoch_summary(
                epoch=epoch,
                epochs=args.epochs,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_best=is_best,
            )
    except KeyboardInterrupt:
        print()
        print("Training interrupted. Resume later with the same command; last completed epoch is saved in last.pt.")
        return 130

    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved best checkpoint: {output_dir / 'checkpoints' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
