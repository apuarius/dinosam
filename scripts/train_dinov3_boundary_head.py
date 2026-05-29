import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from dinosam.data.instance_tiles import (  # noqa: E402
    InstanceTilePair,
    list_instance_tile_pairs,
    load_instance_mask,
)
from dinosam.evaluation import (  # noqa: E402
    dilate_binary_mask,
    label_boundary_map,
    score_map_average_precision,
    score_map_auc,
)
from dinosam.features import extract_patch_feature_grid  # noqa: E402
from dinosam.models import (  # noqa: E402
    DINOv3Wrapper,
    PatchDetectionHead,
    PatchDetectionHeadConfig,
    build_dinov3_config,
)
from dinosam.project import resolve_project_path  # noqa: E402
from dinosam.train import load_config  # noqa: E402


class InstanceTileDataset(Dataset):
    """读取遥感切片和同名实例 mask，供检测头训练使用。"""

    def __init__(self, pairs: list[InstanceTilePair]) -> None:
        """保存已经配对好的 Image/Instance 文件路径列表。"""
        self.pairs = pairs

    def __len__(self) -> int:
        """返回数据集中的切片数量。"""
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """读取单个样本，并复制数组以避免只读 numpy 警告。"""
        pair = self.pairs[index]
        image = Image.open(pair.image_path).convert("RGB")
        instance_mask = load_instance_mask(pair.instance_path).copy()
        return {
            "image": image,
            "instance_mask": instance_mask,
            "name": pair.name,
        }


class BinaryMetricAccumulator:
    """累积 patch 级二分类指标，支持进度条实时显示。"""

    def __init__(self, target_threshold: float) -> None:
        """初始化计数器和用于 AP/AUC 的分数缓存。"""
        self.target_threshold = target_threshold
        self.true_positive = 0.0
        self.false_positive = 0.0
        self.false_negative = 0.0
        self.intersection = 0.0
        self.union = 0.0
        self.scores: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """用一个 batch 的 logits 和 soft target 更新指标。"""
        probabilities = torch.sigmoid(logits).detach().float().cpu().numpy()
        target_values = targets.detach().float().cpu().numpy()
        target_binary = target_values > self.target_threshold
        prediction_binary = probabilities >= 0.5

        self.true_positive += float(np.logical_and(prediction_binary, target_binary).sum())
        self.false_positive += float(np.logical_and(prediction_binary, ~target_binary).sum())
        self.false_negative += float(np.logical_and(~prediction_binary, target_binary).sum())
        self.intersection += float(np.logical_and(prediction_binary, target_binary).sum())
        self.union += float(np.logical_or(prediction_binary, target_binary).sum())
        self.scores.append(probabilities.reshape(-1))
        self.targets.append(target_binary.reshape(-1))

    def compute_fast(self) -> dict[str, float]:
        """计算无需排序的实时 precision、recall、F1 和 IoU。"""
        precision = self.true_positive / max(self.true_positive + self.false_positive, 1.0)
        recall = self.true_positive / max(self.true_positive + self.false_negative, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        iou = self.intersection / max(self.union, 1.0)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    def compute(self) -> dict[str, float | None]:
        """计算完整指标，包括 AP 和 ROC-AUC。"""
        metrics: dict[str, float | None] = dict(self.compute_fast())
        if not self.scores:
            metrics.update({"ap": None, "auc": None})
            return metrics

        scores = np.concatenate(self.scores)
        targets = np.concatenate(self.targets)
        metrics.update(
            {
                "ap": score_map_average_precision(scores, targets),
                "auc": score_map_auc(scores, targets),
            }
        )
        return metrics


def build_parser() -> argparse.ArgumentParser:
    """构建 DINOv3 patch 检测头训练脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a small boundary head on frozen DINOv3 features.")
    parser.add_argument("--train-root", default="data/SAM2_dataset_1024_s512/Train")
    parser.add_argument("--val-root", default="data/SAM2_dataset_1024_s512/Val")
    parser.add_argument("--model-config", default="configs/model/dinov3_sam2.yaml")
    parser.add_argument("--output-dir", default="outputs/dinov3_boundary_head")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--boundary-loss-weight", type=float, default=1.0)
    parser.add_argument("--foreground-loss-weight", type=float, default=0.5)
    parser.add_argument("--max-pos-weight", type=float, default=20.0)
    parser.add_argument("--target-threshold", type=float, default=0.05)
    parser.add_argument("--gt-boundary-dilation", type=int, default=4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feature-cache-dir", default=None)
    parser.add_argument("--device", default=None)
    return parser


def collate_tiles(items: list[dict[str, Any]]) -> dict[str, Any]:
    """把 DataLoader 读出的样本整理成图片列表、mask 列表和文件名列表。"""
    return {
        "images": [item["image"] for item in items],
        "instance_masks": [item["instance_mask"] for item in items],
        "names": [item["name"] for item in items],
    }


def limit_pairs(pairs: list[InstanceTilePair], limit: int | None) -> list[InstanceTilePair]:
    """根据命令行 limit 截断样本列表，便于先做短跑调试。"""
    if limit is None:
        return pairs
    return pairs[:limit]


def make_loader(
    dataset_root: str,
    limit: int | None,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    """根据数据集根目录构建 PyTorch DataLoader。"""
    pairs = limit_pairs(list_instance_tile_pairs(dataset_root), limit)
    dataset = InstanceTileDataset(pairs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tiles,
        pin_memory=torch.cuda.is_available(),
    )


def resize_soft_mask(mask: np.ndarray, patch_shape: tuple[int, int]) -> np.ndarray:
    """把像素级二值图平均池化到 DINOv3 patch 网格大小。"""
    image = Image.fromarray(mask.astype(np.float32), mode="F")
    resized = image.resize((patch_shape[1], patch_shape[0]), Image.Resampling.BOX)
    return np.asarray(resized, dtype=np.float32)


def build_patch_targets(
    instance_masks: list[np.ndarray],
    patch_shape: tuple[int, int],
    device: torch.device,
    boundary_dilation: int,
) -> torch.Tensor:
    """从实例 mask 构建 boundary 和 foreground 两个 patch 级训练目标。"""
    targets: list[np.ndarray] = []
    for instance_mask in instance_masks:
        foreground = instance_mask > 0
        boundary = label_boundary_map(instance_mask)
        boundary = dilate_binary_mask(boundary, radius=boundary_dilation)
        target = np.stack(
            [
                resize_soft_mask(boundary, patch_shape),
                resize_soft_mask(foreground, patch_shape),
            ],
            axis=0,
        )
        targets.append(target)
    array = np.stack(targets, axis=0)
    return torch.from_numpy(array).float().to(device)


def balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_threshold: float,
    max_pos_weight: float,
) -> torch.Tensor:
    """计算带正样本平衡权重的 BCE loss，缓解边界 patch 稀疏问题。"""
    raw_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    positive = targets > target_threshold
    positive_count = positive.sum().float()
    negative_count = positive.numel() - positive_count
    pos_weight = torch.clamp(negative_count / torch.clamp_min(positive_count, 1.0), max=max_pos_weight)
    weights = torch.where(positive, torch.full_like(targets, pos_weight), torch.ones_like(targets))
    return (raw_loss * weights).sum() / torch.clamp_min(weights.sum(), 1.0)


def candidate_token_tensor(raw_features: Any) -> torch.Tensor:
    """从 DINOv3 输出对象中取出最可能的 patch token 张量。"""
    if hasattr(raw_features, "last_hidden_state"):
        return raw_features.last_hidden_state
    if isinstance(raw_features, dict):
        for key in ("x_norm_patchtokens", "patch_tokens", "last_hidden_state"):
            value = raw_features.get(key)
            if torch.is_tensor(value):
                return value
    if torch.is_tensor(raw_features):
        return raw_features
    raise TypeError(f"Cannot find token tensor in DINOv3 output: {type(raw_features).__name__}")


def infer_special_token_count(token_count: int) -> int:
    """根据 token 数推断开头需要跳过的 CLS/register token 数量。"""
    for special_tokens in (5, 1, 0, 4, 2, 3, 6, 7, 8):
        patch_count = token_count - special_tokens
        grid_size = int(math.sqrt(patch_count))
        if patch_count > 0 and grid_size * grid_size == patch_count:
            return special_tokens
    raise ValueError(f"Cannot infer square patch grid from token count: {token_count}")


def extract_patch_feature_tensor(raw_features: Any) -> torch.Tensor:
    """把 DINOv3 输出转换成 BCHW 格式的 patch 特征张量。"""
    tokens = candidate_token_tensor(raw_features)
    if tokens.ndim == 4:
        if tokens.shape[-1] > tokens.shape[1]:
            return tokens.permute(0, 3, 1, 2).contiguous().float()
        return tokens.float()

    if tokens.ndim != 3:
        grid = extract_patch_feature_grid(raw_features)
        array = np.moveaxis(grid, -1, 0)[None, ...]
        return torch.from_numpy(array).float()

    special_tokens = infer_special_token_count(tokens.shape[1])
    patch_tokens = tokens[:, special_tokens:, :]
    grid_size = int(math.sqrt(patch_tokens.shape[1]))
    return patch_tokens.reshape(tokens.shape[0], grid_size, grid_size, tokens.shape[-1]).permute(0, 3, 1, 2).contiguous().float()


def feature_cache_paths(cache_dir: Path, split: str, names: list[str]) -> list[Path]:
    """为一个 batch 的切片生成对应的特征缓存路径。"""
    split_dir = cache_dir / split
    return [split_dir / f"{Path(name).stem}.pt" for name in names]


def load_or_extract_features(
    dinov3: DINOv3Wrapper,
    images: list[Image.Image],
    names: list[str],
    split: str,
    cache_dir: Path,
    cache_features: bool,
    device: torch.device,
) -> torch.Tensor:
    """优先读取 DINOv3 特征缓存，不存在时运行冻结编码器并写入缓存。"""
    paths = feature_cache_paths(cache_dir, split, names)
    if cache_features and all(path.exists() for path in paths):
        features = [torch.load(path, map_location=device).float() for path in paths]
        return torch.stack(features, dim=0)

    if dinov3.processor is None:
        raise RuntimeError("Boundary head training currently expects a Hugging Face DINOv3 model.")

    with torch.no_grad():
        inputs = dinov3.prepare_inputs(images)
        raw_features = dinov3(inputs).raw
        features = extract_patch_feature_tensor(raw_features).to(device)

    if cache_features:
        for path, feature in zip(paths, features, strict=True):
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(feature.detach().cpu().half(), path)
    return features.float()


def metric_prefix(prefix: str, metrics: dict[str, float | None]) -> dict[str, float | None]:
    """给一组指标统一添加前缀，便于区分 boundary 和 foreground。"""
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def format_metric(value: float | None, digits: int = 4) -> str:
    """把可能为空的指标格式化成适合终端显示的字符串。"""
    if value is None:
        return "nan"
    return f"{value:.{digits}f}"


def run_epoch(
    *,
    epoch: int,
    epochs: int,
    split: str,
    loader: DataLoader,
    dinov3: DINOv3Wrapper,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    cache_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int | None]:
    """运行一个训练或验证 epoch，并返回 YOLO 风格的聚合指标。"""
    training = optimizer is not None
    head.train(training)

    total_images = 0
    total_loss = 0.0
    total_boundary_loss = 0.0
    total_foreground_loss = 0.0
    boundary_metrics = BinaryMetricAccumulator(target_threshold=args.target_threshold)
    foreground_metrics = BinaryMetricAccumulator(target_threshold=args.target_threshold)

    bar = tqdm(loader, desc=f"{split} {epoch}/{epochs}", dynamic_ncols=True)
    for batch in bar:
        images = batch["images"]
        names = batch["names"]
        instance_masks = batch["instance_masks"]
        features = load_or_extract_features(
            dinov3=dinov3,
            images=images,
            names=names,
            split=split,
            cache_dir=cache_dir,
            cache_features=args.cache_features,
            device=device,
        )
        targets = build_patch_targets(
            instance_masks,
            patch_shape=tuple(features.shape[-2:]),
            device=device,
            boundary_dilation=args.gt_boundary_dilation,
        )

        logits = head(features)
        boundary_loss = balanced_bce_with_logits(
            logits[:, 0],
            targets[:, 0],
            target_threshold=args.target_threshold,
            max_pos_weight=args.max_pos_weight,
        )
        foreground_loss = balanced_bce_with_logits(
            logits[:, 1],
            targets[:, 1],
            target_threshold=args.target_threshold,
            max_pos_weight=args.max_pos_weight,
        )
        loss = args.boundary_loss_weight * boundary_loss + args.foreground_loss_weight * foreground_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        batch_size = len(images)
        total_images += batch_size
        total_loss += float(loss.detach()) * batch_size
        total_boundary_loss += float(boundary_loss.detach()) * batch_size
        total_foreground_loss += float(foreground_loss.detach()) * batch_size
        boundary_metrics.update(logits[:, 0], targets[:, 0])
        foreground_metrics.update(logits[:, 1], targets[:, 1])

        fast_boundary = boundary_metrics.compute_fast()
        fast_foreground = foreground_metrics.compute_fast()
        bar.set_postfix(
            loss=format_metric(total_loss / max(total_images, 1)),
            b_f1=format_metric(fast_boundary["f1"], digits=3),
            fg_iou=format_metric(fast_foreground["iou"], digits=3),
        )

    boundary = metric_prefix("boundary", boundary_metrics.compute())
    foreground = metric_prefix("foreground", foreground_metrics.compute())
    metrics: dict[str, float | int | None] = {
        "images": total_images,
        "loss": total_loss / max(total_images, 1),
        "boundary_loss": total_boundary_loss / max(total_images, 1),
        "foreground_loss": total_foreground_loss / max(total_images, 1),
    }
    metrics.update(boundary)
    metrics.update(foreground)
    return metrics


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    """把一个 epoch 的指标追加写入 CSV，方便后续用表格或画图查看。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    """把一个 epoch 的完整指标追加写入 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """保存检测头权重、优化器状态和当前 epoch 指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def prefixed_metrics(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    """给 train 或 val 指标添加统一前缀。"""
    return {f"{prefix}_{key}": value for key, value in values.items()}


def main() -> int:
    """训练 DINOv3 patch 检测头，并保存进度指标和最佳权重。"""
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = resolve_project_path(args.output_dir)
    cache_dir = resolve_project_path(args.feature_cache_dir) if args.feature_cache_dir else output_dir / "feature_cache"
    metrics_csv = output_dir / "metrics.csv"
    metrics_jsonl = output_dir / "metrics.jsonl"

    model_config = load_config(resolve_project_path(args.model_config))
    dinov3_config = build_dinov3_config(model_config)
    dinov3 = DINOv3Wrapper.from_config(dinov3_config)
    input_channels = int(getattr(getattr(dinov3.model, "config", None), "hidden_size", 1024))
    head = PatchDetectionHead.build(
        PatchDetectionHeadConfig(
            input_channels=input_channels,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            output_channels=2,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = make_loader(
        args.train_root,
        limit=args.limit_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = make_loader(
        args.val_root,
        limit=args.limit_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    print(f"Device: {device}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Val images: {len(val_loader.dataset)}")
    print(f"Feature cache: {'on' if args.cache_features else 'off'} -> {cache_dir}")
    print(f"Output dir: {output_dir}")

    best_metric = -1.0
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            epoch=epoch,
            epochs=args.epochs,
            split="train",
            loader=train_loader,
            dinov3=dinov3,
            head=head,
            optimizer=optimizer,
            cache_dir=cache_dir,
            args=args,
            device=device,
        )
        val_metrics = run_epoch(
            epoch=epoch,
            epochs=args.epochs,
            split="val",
            loader=val_loader,
            dinov3=dinov3,
            head=head,
            optimizer=None,
            cache_dir=cache_dir,
            args=args,
            device=device,
        )

        epoch_row: dict[str, Any] = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_min": (time.time() - start_time) / 60.0,
        }
        epoch_row.update(prefixed_metrics("train", train_metrics))
        epoch_row.update(prefixed_metrics("val", val_metrics))

        write_csv_row(metrics_csv, epoch_row)
        write_jsonl_row(metrics_jsonl, epoch_row)
        save_checkpoint(
            output_dir / "checkpoints" / "last.pt",
            epoch=epoch,
            head=head,
            optimizer=optimizer,
            metrics=epoch_row,
            args=args,
        )

        score = float(val_metrics.get("boundary_f1") or 0.0)
        if score > best_metric:
            best_metric = score
            save_checkpoint(
                output_dir / "checkpoints" / "best.pt",
                epoch=epoch,
                head=head,
                optimizer=optimizer,
                metrics=epoch_row,
                args=args,
            )

        print(
            "Epoch "
            f"{epoch}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_boundary_f1={format_metric(val_metrics.get('boundary_f1'))} | "
            f"val_boundary_ap={format_metric(val_metrics.get('boundary_ap'))} | "
            f"val_foreground_iou={format_metric(val_metrics.get('foreground_iou'))}"
        )

    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved best checkpoint: {output_dir / 'checkpoints' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
