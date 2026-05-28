from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dinosam.project import resolve_project_path


@dataclass(frozen=True)
class DINOv3Config:
    """描述 DINOv3 模型加载所需的最小配置。"""

    load_from: str = "torch_hub"
    hf_model_dir: str | Path | None = None
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


def _resolve_hf_model_dir(value: str | Path | None) -> Path:
    """把 Hugging Face 本地模型目录解析成仓库内的绝对路径。"""
    if value is None:
        raise ValueError("DINOv3 hf_model_dir is not configured.")

    path = Path(value)
    if path.is_absolute():
        return path
    return resolve_project_path(path)


def _require_hf_model_files(model_dir: Path) -> None:
    """检查 Hugging Face 本地模型目录中是否包含最关键的三个文件。"""
    required_names = ("config.json", "model.safetensors", "preprocessor_config.json")
    missing = [name for name in required_names if not (model_dir / name).exists()]
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"DINOv3 Hugging Face files are missing: {missing_text}")


def _configure_model_runtime(model: Any, config: DINOv3Config) -> None:
    """把模型移动到目标设备，并按配置切换冻结和推理模式。"""
    model.to(config.device)

    if config.eval_mode:
        model.eval()

    if config.freeze:
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def _move_batch_to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    """把 processor 生成的输入批次移动到和模型一致的设备上。"""
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


class DINOv3Wrapper:
    """封装 DINOv3 的两种加载入口和统一特征提取调用。"""

    def __init__(self, model: Any, config: DINOv3Config, processor: Any | None = None) -> None:
        """保存已加载的 DINOv3 模型、配置和可选的图像预处理器。"""
        self.model = model
        self.config = config
        self.processor = processor

    @classmethod
    def from_config(cls, config: DINOv3Config) -> "DINOv3Wrapper":
        """根据配置选择 torch.hub 或 Hugging Face 本地目录加载 DINOv3。"""
        load_from = config.load_from.lower()
        if load_from == "torch_hub":
            return cls._from_torch_hub_config(config)
        if load_from == "huggingface":
            return cls._from_huggingface_config(config)

        raise ValueError(f"Unsupported DINOv3 load_from value: {config.load_from}")

    @classmethod
    def _from_torch_hub_config(cls, config: DINOv3Config) -> "DINOv3Wrapper":
        """从本地 DINOv3 submodule 和 .pth 权重加载官方 torch.hub 模型。"""
        import torch

        repo_dir = resolve_project_path(config.repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(f"DINOv3 repository not found: {repo_dir}")

        kwargs: dict[str, Any] = {"pretrained": config.pretrained}
        weights = _resolve_weight_value(config.weights)
        if weights is not None:
            kwargs["weights"] = weights

        model = torch.hub.load(
            str(repo_dir),
            config.model_name,
            source="local",
            **kwargs,
        )
        _configure_model_runtime(model, config)
        return cls(model=model, config=config)

    @classmethod
    def _from_huggingface_config(cls, config: DINOv3Config) -> "DINOv3Wrapper":
        """从 Hugging Face 本地模型目录加载 config、processor 和 safetensors 权重。"""
        from transformers import AutoImageProcessor, AutoModel

        model_dir = _resolve_hf_model_dir(config.hf_model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"DINOv3 Hugging Face model directory not found: {model_dir}")
        _require_hf_model_files(model_dir)

        processor = AutoImageProcessor.from_pretrained(str(model_dir), local_files_only=True)
        model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
        _configure_model_runtime(model, config)
        return cls(model=model, config=config, processor=processor)

    def prepare_inputs(self, images: Any, return_tensors: str = "pt") -> Mapping[str, Any]:
        """使用 Hugging Face processor 把图像转换成 DINOv3 可直接接收的输入。"""
        if self.processor is None:
            raise RuntimeError("This DINOv3 wrapper was not loaded with a Hugging Face processor.")

        batch = self.processor(images=images, return_tensors=return_tensors)
        if hasattr(batch, "to"):
            return batch.to(self.config.device)
        return _move_batch_to_device(batch, self.config.device)

    def encode(self, images: Any) -> DINOv3Features:
        """对输入图像执行一次 DINOv3 编码，并返回统一的特征包装对象。"""
        if self.processor is not None and not isinstance(images, Mapping):
            images = self.prepare_inputs(images)

        if isinstance(images, Mapping):
            features = self.model(**images)
        elif hasattr(self.model, "forward_features"):
            features = self.model.forward_features(images)
        else:
            features = self.model(images)
        return DINOv3Features(raw=features)

    def __call__(self, images: Any) -> DINOv3Features:
        """让 wrapper 可以像函数一样直接执行特征提取。"""
        return self.encode(images)
