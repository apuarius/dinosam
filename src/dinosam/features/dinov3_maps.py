import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from PIL import Image


def image_to_sat493m_tensor(image: np.ndarray, device: str) -> Any:
    """按 SAT-493M 归一化参数把 RGB 图像转换成 torch 输入张量。"""
    import torch

    mean = torch.tensor([0.430, 0.411, 0.296], device=device).view(3, 1, 1)
    std = torch.tensor([0.213, 0.156, 0.143], device=device).view(3, 1, 1)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(device) / 255.0
    return ((tensor - mean) / std).unsqueeze(0)


def _tensor_to_numpy(value: Any) -> np.ndarray:
    """把 torch 张量或类数组对象转换成 CPU numpy 数组。"""
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _iter_feature_candidates(raw_features: Any) -> list[Any]:
    """从常见 DINOv3 输出结构中收集可能的 patch token 张量。"""
    candidates: list[Any] = []

    if hasattr(raw_features, "last_hidden_state"):
        candidates.append(raw_features.last_hidden_state)

    if isinstance(raw_features, Mapping):
        for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state"):
            if key in raw_features:
                candidates.append(raw_features[key])

    if hasattr(raw_features, "to_tuple"):
        candidates.extend(raw_features.to_tuple())
    elif isinstance(raw_features, (list, tuple)):
        candidates.extend(raw_features)

    if not candidates:
        candidates.append(raw_features)
    return candidates


def _square_grid_size(token_count: int) -> int | None:
    """如果 token 数量可以组成正方形网格，则返回边长。"""
    size = int(math.sqrt(token_count))
    if size * size == token_count:
        return size
    return None


def _reshape_token_array(tokens: np.ndarray) -> np.ndarray | None:
    """把 NCH 或 NHWC 等常见 token 数组整理成 HWC 特征网格。"""
    array = tokens
    if array.ndim == 4:
        array = array[0]
        if array.shape[0] > array.shape[1] and array.shape[0] > array.shape[-1]:
            array = np.moveaxis(array, 0, -1)
        return array.astype(np.float32, copy=False)

    if array.ndim == 3:
        array = array[0]

    if array.ndim != 2:
        return None

    token_count = array.shape[0]
    grid_size = _square_grid_size(token_count)
    if grid_size is not None:
        return array.reshape(grid_size, grid_size, array.shape[-1]).astype(np.float32, copy=False)

    grid_size = _square_grid_size(token_count - 1)
    if grid_size is not None:
        patch_tokens = array[1:]
        return patch_tokens.reshape(grid_size, grid_size, array.shape[-1]).astype(np.float32, copy=False)

    return None


def extract_patch_feature_grid(raw_features: Any) -> np.ndarray:
    """从 DINOv3 原始输出中抽取 HWC 形式的 patch feature 网格。"""
    for candidate in _iter_feature_candidates(raw_features):
        array = _tensor_to_numpy(candidate)
        grid = _reshape_token_array(array)
        if grid is not None:
            return grid

    raise ValueError("Cannot infer a square patch feature grid from DINOv3 output.")


def boundary_map_from_feature_grid(feature_grid: np.ndarray) -> np.ndarray:
    """用相邻 patch 的 cosine distance 生成归一化边界热图。"""
    if feature_grid.ndim != 3:
        raise ValueError(f"Feature grid must be HWC, got shape: {feature_grid.shape}")

    features = feature_grid.astype(np.float32, copy=False)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    normalized = features / np.maximum(norms, 1e-6)

    height, width, _ = normalized.shape
    boundary = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)

    if width > 1:
        right = 1.0 - np.sum(normalized[:, :-1] * normalized[:, 1:], axis=-1)
        boundary[:, :-1] += right
        boundary[:, 1:] += right
        counts[:, :-1] += 1.0
        counts[:, 1:] += 1.0

    if height > 1:
        down = 1.0 - np.sum(normalized[:-1, :] * normalized[1:, :], axis=-1)
        boundary[:-1, :] += down
        boundary[1:, :] += down
        counts[:-1, :] += 1.0
        counts[1:, :] += 1.0

    boundary = boundary / np.maximum(counts, 1.0)
    low = float(boundary.min())
    high = float(boundary.max())
    if high <= low:
        return np.zeros_like(boundary, dtype=np.float32)
    return ((boundary - low) / (high - low)).astype(np.float32)


def resize_float_map(value: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """把二维 float 热图缩放到指定宽高，并保持 0 到 1 的范围。"""
    value = np.clip(value, 0.0, 1.0)
    image = Image.fromarray((value * 255).astype(np.uint8), mode="L")
    resized = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0
