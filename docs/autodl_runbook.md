# AutoDL Runbook

这份手册记录在 AutoDL 上启动 `dinosam-lab` 的最小流程。目标是先跑通环境、路径和配置，不在第一步就加载大模型。

## 1. Clone 仓库

```bash
cd /root/autodl-tmp
git clone --recurse-submodules https://github.com/apuarius/dinosam.git
cd dinosam
```

如果已经普通 clone 了，再补 submodule：

```bash
git submodule update --init --recursive
```

## 2. 安装项目

```bash
conda create -n dinosam python=3.10 -y
conda activate dinosam
pip install -U pip
pip install -e .
```

这一步会让 Python 识别 `src/dinosam` 包，并安装 `pyyaml`。

## 3. 准备工作目录

```bash
python scripts/prepare_workspace.py
```

这个脚本会创建：

```text
data/
weights/
weights/dinov3/
weights/sam2/
outputs/
outputs/runs/
outputs/predictions/
outputs/visualizations/
```

它还会根据 `configs/model/dinov3_sam2.yaml` 打印当前期望的权重路径。

## 4. 放置权重

当前模型配置期望：

```text
weights/dinov3/dinov3_vitl16.pth
weights/sam2/sam2.1_hiera_large.pt
```

SAM2 checkpoint 可以按官方仓库说明下载 `sam2.1_hiera_large.pt`，然后放到 `weights/sam2/`。

DINOv3 权重先按实际获取方式放到 `weights/dinov3/`。如果后续改用 `torch.hub` 默认下载或 Hugging Face 权重，需要同步修改 `configs/model/dinov3_sam2.yaml`。

## 5. 路径和配置检查

```bash
python scripts/check_submodules.py
python -m dinosam.train --config configs/train/smoke.yaml
```

这一步只检查配置和路径，不会真正加载 DINOv3/SAM2 权重。

## 6. 每次实验前记录

```bash
git rev-parse HEAD
git submodule status
nvidia-smi
python -V
pip freeze > outputs/requirements.lock.txt
```

这些信息之后用来复现实验。
