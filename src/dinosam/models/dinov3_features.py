import math
from collections.abc import Mapping
from typing import Any

import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _feature_candidates(raw_features: Any) -> list[Any]:
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

    return candidates or [raw_features]


def _square_size(token_count: int) -> int | None:
    size = int(math.sqrt(token_count))
    return size if size * size == token_count else None


def _reshape_tokens(tokens: np.ndarray) -> np.ndarray | None:
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

    for special_tokens in (0, 1, 5, 4, 2, 3, 6, 7, 8):
        patch_count = array.shape[0] - special_tokens
        grid_size = _square_size(patch_count)
        if grid_size is None:
            continue
        patch_tokens = array[special_tokens:]
        return patch_tokens.reshape(grid_size, grid_size, array.shape[-1]).astype(
            np.float32,
            copy=False,
        )
    return None


def extract_patch_feature_grid(raw_features: Any) -> np.ndarray:
    """Extract an HWC patch-feature grid from a DINOv3 model output."""
    for candidate in _feature_candidates(raw_features):
        grid = _reshape_tokens(_to_numpy(candidate))
        if grid is not None:
            return grid
    raise ValueError("Cannot infer a square patch feature grid from DINOv3 output.")
