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
