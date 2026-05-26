import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dinosam.project import resolve_project_path


@dataclass(frozen=True)
class SAM2Config:
    """描述 SAM2 图像预测器加载所需的最小配置。"""

    repo_dir: str | Path = "third_party/sam2"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint: str | Path | None = None
    device: str = "cuda"
    mode: str = "eval"


@dataclass(frozen=True)
class SAM2Prediction:
    """统一承载 SAM2 predict 返回的 masks、scores 和 logits。"""

    masks: Any
    scores: Any
    logits: Any


def _ensure_repo_on_path(repo_dir: Path) -> None:
    """把 SAM2 仓库根目录加入 sys.path，便于导入官方 sam2 包。"""
    repo_dir_text = str(repo_dir)
    if repo_dir_text not in sys.path:
        sys.path.insert(0, repo_dir_text)


def _resolve_checkpoint(value: str | Path | None) -> str | None:
    """把 SAM2 checkpoint 的相对路径转换成绝对路径字符串。"""
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(resolve_project_path(path))


class SAM2ImageWrapper:
    """封装 SAM2 官方图像预测器的加载、设图和预测流程。"""

    def __init__(self, predictor: Any, config: SAM2Config) -> None:
        """保存已加载的 SAM2ImagePredictor 和对应配置。"""
        self.predictor = predictor
        self.config = config

    @classmethod
    def from_config(cls, config: SAM2Config) -> "SAM2ImageWrapper":
        """根据配置从本地 SAM2 submodule 加载图像预测器。"""
        repo_dir = resolve_project_path(config.repo_dir)
        if not repo_dir.exists():
            raise FileNotFoundError(f"SAM2 repository not found: {repo_dir}")

        _ensure_repo_on_path(repo_dir)

        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        model = build_sam2(
            config_file=config.model_cfg,
            ckpt_path=_resolve_checkpoint(config.checkpoint),
            device=config.device,
            mode=config.mode,
        )
        predictor = SAM2ImagePredictor(model)
        return cls(predictor=predictor, config=config)

    def set_image(self, image: Any) -> None:
        """把待分割图像送入 SAM2，并缓存图像特征供后续 prompt 使用。"""
        self.predictor.set_image(image)

    def predict(self, **prompt_kwargs: Any) -> SAM2Prediction:
        """根据点、框或 mask 等 prompt 参数调用 SAM2 生成预测结果。"""
        masks, scores, logits = self.predictor.predict(**prompt_kwargs)
        return SAM2Prediction(masks=masks, scores=scores, logits=logits)

    def reset(self) -> None:
        """清空 SAM2 预测器中缓存的当前图像状态。"""
        if hasattr(self.predictor, "reset_predictor"):
            self.predictor.reset_predictor()
