import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dinosam.project import resolve_project_path


@dataclass(frozen=True)
class SAM2Config:
    repo_dir: str | Path = "third_party/sam2"
    model_cfg: str = "configs/sam2.1/sam2.1_hiera_l.yaml"
    checkpoint: str | Path | None = None
    device: str = "cuda"
    mode: str = "eval"


@dataclass(frozen=True)
class SAM2Prediction:
    masks: Any
    scores: Any
    logits: Any


def _ensure_repo_on_path(repo_dir: Path) -> None:
    repo_dir_text = str(repo_dir)
    if repo_dir_text not in sys.path:
        sys.path.insert(0, repo_dir_text)


def _resolve_checkpoint(value: str | Path | None) -> str | None:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(resolve_project_path(path))


class SAM2ImageWrapper:
    def __init__(self, predictor: Any, config: SAM2Config) -> None:
        self.predictor = predictor
        self.config = config

    @classmethod
    def from_config(cls, config: SAM2Config) -> "SAM2ImageWrapper":
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
        self.predictor.set_image(image)

    def predict(self, **prompt_kwargs: Any) -> SAM2Prediction:
        masks, scores, logits = self.predictor.predict(**prompt_kwargs)
        return SAM2Prediction(masks=masks, scores=scores, logits=logits)

    def reset(self) -> None:
        if hasattr(self.predictor, "reset_predictor"):
            self.predictor.reset_predictor()
