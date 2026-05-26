from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dinosam.project import resolve_project_path


@dataclass(frozen=True)
class DINOv3Config:
    """描述 DINOv3 模型加载所需的最小配置。"""

    repo_dir: str | Path = "third_party/dinov3"
    model_name: str = "dinov3_vitl16"
    pretrained: bool = True
    weights: str | Path | None = None
    device: str = "cuda"
    freeze: bool = True
    eval_mode: bool = True


@dataclass(frozen=True)
class DINOv3Features:
    """统一承载 DINOv3 编码器输出的特征对象。"""

    raw: Any


def _resolve_weight_value(value: str | Path | None) -> str | None:
    """把 DINOv3 权重路径或 URL 转换成 torch.hub 可接受的字符串。"""
    if value is None:
        return None

    text = str(value)
    if "://" in text:
        return text

    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str(resolve_project_path(path))


class DINOv3Wrapper:
    """封装 DINOv3 官方 torch.hub 加载入口和特征提取调用。"""

    def __init__(self, model: Any, config: DINOv3Config) -> None:
        """保存已加载的 DINOv3 模型和对应配置。"""
        self.model = model
        self.config = config

    @classmethod
    def from_config(cls, config: DINOv3Config) -> "DINOv3Wrapper":
        """根据配置从本地 DINOv3 submodule 加载模型。"""
        import torch

        repo_dir = resolve_project_path(config.repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repository not found: {repo_dir}")

        kwargs: dict[str, Any] = {}
        kwargs["pretrained"] = config.pretrained
        weights = _resolve_weight_value(config.weights)
        if weights is not None:
            kwargs["weights"] = weights

        model = torch.hub.load(
            str(repo_dir),
            config.model_name,
            source="local",
            **kwargs,
        )
        model.to(config.device)

        if config.eval_mode:
            model.eval()

        if config.freeze:
            for parameter in model.parameters():
                parameter.requires_grad_(False)

        return cls(model=model, config=config)

    def encode(self, images: Any) -> DINOv3Features:
        """对输入图像进行 DINOv3 编码，并返回统一的特征包装对象。"""
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(images)
        else:
            features = self.model(images)
        return DINOv3Features(raw=features)

    def __call__(self, images: Any) -> DINOv3Features:
        """让 wrapper 可以像函数一样直接执行特征提取。"""
        return self.encode(images)
