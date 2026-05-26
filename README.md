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

## Smoke Checks

```bash
python scripts/check_submodules.py
python -m dinosam.train --config configs/train/smoke.yaml
```

## Current External Versions

```bash
git submodule status
```
