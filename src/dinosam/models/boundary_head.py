from dataclasses import dataclass


@dataclass(frozen=True)
class PatchDetectionHeadConfig:
    """保存 DINOv3 patch 检测头的结构参数。"""

    input_channels: int = 1024
    hidden_channels: int = 256
    dropout: float = 0.1
    output_channels: int = 2


class PatchDetectionHead:
    """轻量 patch 检测头，用 DINOv3 特征预测边界和田块区域。"""

    @staticmethod
    def build(config: PatchDetectionHeadConfig):
        """按配置构建一个小型卷积检测头。"""
        import torch.nn as nn

        return nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(config.dropout),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout2d(config.dropout),
            nn.Conv2d(config.hidden_channels, config.output_channels, kernel_size=1),
        )
