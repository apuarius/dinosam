from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PatchDetectionHeadConfig:
    """保存 DINOv3 patch 检测头的结构参数。"""

    input_channels: int = 1024
    hidden_channels: int = 256
    dropout: float = 0.1
    output_channels: int = 2
    head_type: str = "basic"


def _group_count(channels: int) -> int:
    """为 GroupNorm 选择能整除通道数的小组数，避免小 batch 下 BatchNorm 不稳定。"""
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SqueezeExcitation(nn.Module):
    """轻量通道注意力，用于重标定 DINOv3 高维特征通道。"""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced_channels = max(channels // reduction, 16)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """按通道权重增强与农田边界更相关的特征。"""
        return features * self.layers(features)


class ConvNormAct(nn.Sequential):
    """卷积、GroupNorm、GELU 和 Dropout 的小模块。"""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int,
        padding: int = 0,
        dilation: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )


class ResidualConvBlock(nn.Module):
    """残差卷积块，用于增强 patch 网格上的局部边界结构表达。"""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvNormAct(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                dropout=dropout,
            ),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """保留输入特征并学习局部修正，降低加深 head 后的训练风险。"""
        return self.activation(features + self.layers(features))


class ASPPLite(nn.Module):
    """轻量多尺度空洞卷积，适配 14x14 patch 网格上的农田长边界。"""

    def __init__(
        self,
        channels: int,
        *,
        branch_channels: int,
        dropout: float,
        dilations: tuple[int, ...] = (1, 2, 3),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvNormAct(
                    channels,
                    branch_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    dropout=dropout,
                )
                for dilation in dilations
            ]
        )
        self.fuse = ConvNormAct(
            branch_channels * len(dilations),
            channels,
            kernel_size=1,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """融合局部、中尺度和较大范围上下文。"""
        return self.fuse(torch.cat([branch(features) for branch in self.branches], dim=1))


class ASPPBoundaryHead(nn.Module):
    """V2 检测头：通道注意力、多尺度上下文和 boundary/foreground 双分支。"""

    def __init__(self, config: PatchDetectionHeadConfig) -> None:
        super().__init__()
        if config.output_channels != 2:
            raise ValueError("ASPPBoundaryHead expects exactly 2 output channels.")

        branch_channels = max(config.hidden_channels // 4, 32)
        self.reduce = nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(config.dropout),
        )
        self.channel_attention = SqueezeExcitation(config.hidden_channels)
        self.context = ASPPLite(
            config.hidden_channels,
            branch_channels=branch_channels,
            dropout=config.dropout,
        )
        self.shared = ResidualConvBlock(config.hidden_channels, dropout=config.dropout)
        self.boundary_branch = nn.Sequential(
            ResidualConvBlock(config.hidden_channels, dropout=config.dropout),
            nn.Conv2d(config.hidden_channels, 1, kernel_size=1),
        )
        self.foreground_branch = nn.Sequential(
            ResidualConvBlock(config.hidden_channels, dropout=config.dropout),
            nn.Conv2d(config.hidden_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回 boundary 与 foreground 两个 logits 通道。"""
        hidden = self.reduce(features)
        hidden = self.channel_attention(hidden)
        hidden = self.context(hidden)
        hidden = self.shared(hidden)
        boundary = self.boundary_branch(hidden)
        foreground = self.foreground_branch(hidden)
        return torch.cat([boundary, foreground], dim=1)

def _attention_heads(channels: int) -> int:
    """为 patch self-attention 选择可整除通道数的注意力头数。"""
    for heads in (8, 4, 2, 1):
        if channels % heads == 0:
            return heads
    return 1


class PatchSelfAttention(nn.Module):
    """在 14x14 patch 网格上建模全局田块关系的轻量自注意力模块。"""

    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        heads = _attention_heads(channels)
        self.norm_attention = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_ffn = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """把 BCHW 特征展平成 token 序列做注意力，再恢复成 patch 网格。"""
        batch, channels, height, width = features.shape
        tokens = features.flatten(2).transpose(1, 2)
        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class AttentionBoundaryHead(nn.Module):
    """T1 检测头：ASPP 多尺度上下文加 patch self-attention。"""

    def __init__(self, config: PatchDetectionHeadConfig) -> None:
        super().__init__()
        if config.output_channels != 2:
            raise ValueError("AttentionBoundaryHead expects exactly 2 output channels.")

        branch_channels = max(config.hidden_channels // 4, 32)
        self.reduce = nn.Sequential(
            nn.Conv2d(config.input_channels, config.hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout2d(config.dropout),
        )
        self.channel_attention = SqueezeExcitation(config.hidden_channels)
        self.context = ASPPLite(
            config.hidden_channels,
            branch_channels=branch_channels,
            dropout=config.dropout,
        )
        self.patch_attention = PatchSelfAttention(config.hidden_channels, dropout=config.dropout)
        self.shared = ResidualConvBlock(config.hidden_channels, dropout=config.dropout)
        self.boundary_branch = nn.Sequential(
            ResidualConvBlock(config.hidden_channels, dropout=config.dropout),
            nn.Conv2d(config.hidden_channels, 1, kernel_size=1),
        )
        self.foreground_branch = nn.Sequential(
            ResidualConvBlock(config.hidden_channels, dropout=config.dropout),
            nn.Conv2d(config.hidden_channels, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回 boundary 与 foreground 两个 logits 通道。"""
        hidden = self.reduce(features)
        hidden = self.channel_attention(hidden)
        hidden = self.context(hidden)
        hidden = self.patch_attention(hidden)
        hidden = self.shared(hidden)
        boundary = self.boundary_branch(hidden)
        foreground = self.foreground_branch(hidden)
        return torch.cat([boundary, foreground], dim=1)


class PatchDetectionHead:
    """轻量 patch 检测头工厂，支持历史 head 和当前 T1 head。"""

    @staticmethod
    def build(config: PatchDetectionHeadConfig) -> nn.Module:
        """按配置构建检测头；默认 basic 保持旧实验兼容。"""
        head_type = config.head_type.lower()
        if head_type == "basic":
            return nn.Sequential(
                nn.Conv2d(config.input_channels, config.hidden_channels, kernel_size=1),
                nn.GELU(),
                nn.Dropout2d(config.dropout),
                nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout2d(config.dropout),
                nn.Conv2d(config.hidden_channels, config.output_channels, kernel_size=1),
            )
        if head_type in {"aspp", "v2"}:
            return ASPPBoundaryHead(config)
        if head_type in {"attention", "v3", "t1"}:
            return AttentionBoundaryHead(config)
        raise ValueError(f"Unsupported patch detection head type: {config.head_type}")
