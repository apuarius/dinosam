from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureBundle:
    dinov3: Any
    sam2: Any | None = None


@dataclass(frozen=True)
class FusedFeatures:
    raw: Any


class BaseFusion:
    def forward(self, features: FeatureBundle) -> FusedFeatures:
        raise NotImplementedError

    def __call__(self, features: FeatureBundle) -> FusedFeatures:
        return self.forward(features)
