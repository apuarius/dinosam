# dinosam-lab

DINOv3 + SAM2 experiment workspace.

## Structure

- `src/`: project code
- `configs/`: experiment configs
- `scripts/`: helper scripts
- `third_party/dinov3`: DINOv3 submodule
- `third_party/sam2`: SAM2 submodule
- `data/`: local datasets, ignored by Git
- `weights/`: local checkpoints, ignored by Git
- `outputs/`: experiment outputs, ignored by Git

## Clone

```bash
git clone --recurse-submodules <repo-url>
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Setup

```bash
pip install -e .
```

## Checks

```bash
python scripts/check_submodules.py
```

## Dataset Layout

```text
data/
  images/
    train/
    val/
    test/
  masks/
    train/
    val/
    test/
```

Images and masks are paired by file stem. For example, `data/images/train/tile_001.png`
can pair with `data/masks/train/tile_001.png`.

## T1 Experiment

T1 is the current experiment line. It uses frozen DINOv3 satellite features, the
attention boundary/foreground head, train-only online augmentation, and SAM2
box/positive/negative prompts. Earlier V1/V2 configs are kept for reference but
are not the active training target.

## Train T1

```bash
python scripts/train_dinov3_boundary_head.py --config configs/train/dinov3_boundary_head_t1.yaml
```

Training uses online augmentation only for `train`. With the default `augment_factor: 4`,
1161 train tiles become 4644 train samples per epoch. Validation and test images are
not augmented. Train feature caching is disabled while online augmentation is enabled.

T1 defaults target higher GPU utilization on large GPUs:

```text
batch_size: 256
num_workers: 16
amp_dtype: bfloat16
preprocess_in_workers: true
prefetch_factor: 6
```

If the process runs out of memory, reduce to `--batch-size 128` or `64`. If
DataLoader workers are unstable on the server, reduce to `--num-workers 8`.

## Current External Versions

```bash
git submodule status
```
