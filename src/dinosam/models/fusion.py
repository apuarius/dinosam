from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureBundle:
    """存放进入融合模块的 DINOv3 特征和可选 SAM2 特征。"""

    dinov3: Any
    sam2: Any | None = None


@dataclass(frozen=True)
class FusedFeatures:
    """统一承载融合模块输出的特征对象。"""

    raw: Any


class BaseFusion:
    """所有特征融合实现需要遵循的最小接口。"""

    def forward(self, features: FeatureBundle) -> FusedFeatures:
        """执行具体的特征融合逻辑，子类必须实现。"""
        raise NotImplementedError

    def __call__(self, features: FeatureBundle) -> FusedFeatures:
        """让融合模块可以像函数一样被调用。"""
        return self.forward(features)
