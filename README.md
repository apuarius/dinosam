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

## Train

```bash
python scripts/train_dinov3_boundary_head.py --config configs/train/dinov3_boundary_head_v2.yaml
```

Training uses online augmentation only for `train`. With the default `augment_factor: 4`,
1161 train tiles become 4644 train samples per epoch. Validation and test images are
not augmented. Train feature caching is disabled while online augmentation is enabled.

## Current External Versions

```bash
git submodule status
```
