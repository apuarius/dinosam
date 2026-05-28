from dataclasses import dataclass
from typing import Any, Mapping

from dinosam.models.dinov3_wrapper import DINOv3Config
from dinosam.models.sam2_wrapper import SAM2Config


@dataclass(frozen=True)
class ModelConfigs:
    """统一保存一次实验中 DINOv3 和 SAM2 的模型配置。"""

    dinov3: DINOv3Config
    sam2: SAM2Config


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """从模型配置字典中取出指定 section，并检查它确实是字典结构。"""
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise TypeError(f"Model config section must be a mapping: {name}")
    return value


def build_dinov3_config(config: Mapping[str, Any]) -> DINOv3Config:
    """把 YAML 中的 dinov3 section 转换成 DINOv3Config。"""
    section = _section(config, "dinov3")
    return DINOv3Config(
        load_from=section.get("load_from", "torch_hub"),
        hf_model_dir=section.get("hf_model_dir"),
        repo_dir=section.get("repo_dir", "third_party/dinov3"),
        model_name=section.get("model_name", "dinov3_vitl16"),
        pretrained=section.get("pretrained", True),
        weights=section.get("weights"),
        device=section.get("device", "cuda"),
        freeze=section.get("freeze", True),
        eval_mode=section.get("eval_mode", True),
    )


def build_sam2_config(config: Mapping[str, Any]) -> SAM2Config:
    """把 YAML 中的 sam2 section 转换成 SAM2Config。"""
    section = _section(config, "sam2")
    return SAM2Config(
        repo_dir=section.get("repo_dir", "third_party/sam2"),
        model_cfg=section.get("model_cfg", "configs/sam2.1/sam2.1_hiera_l.yaml"),
        checkpoint=section.get("checkpoint"),
        device=section.get("device", "cuda"),
        mode=section.get("mode", "eval"),
    )


def build_model_configs(config: Mapping[str, Any]) -> ModelConfigs:
    """把完整模型配置转换成 wrapper 可以直接使用的配置对象。"""
    return ModelConfigs(
        dinov3=build_dinov3_config(config),
        sam2=build_sam2_config(config),
    )
