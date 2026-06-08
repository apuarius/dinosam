# AutoDL Runbook

这份手册记录在 AutoDL 上启动 `dinosam-lab` 的最小流程。当前实验使用按区域划分后的新数据集，主线命名为 T1。

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
conda create -n dinosam python=3.11 -y
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
weights/dinov3/dinov3-vitl16-pretrain-sat493m/
weights/sam2/sam2.1_hiera_large.pt
```

SAM2 checkpoint 可以按官方仓库说明下载 `sam2.1_hiera_large.pt`，然后放到 `weights/sam2/`。

DINOv3 SAT-493M Hugging Face 本地目录应包含：

```text
config.json
model.safetensors
preprocessor_config.json
```

## 5. 放置新数据集

正式数据集目录：

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

图像和 mask 按文件 stem 配对，例如：

```text
data/images/train/tile_001.png
data/masks/train/tile_001.png
```

## 6. 路径和配置检查

```bash
python scripts/check_submodules.py
```

这一步只检查 submodule 是否存在。

## 7. 训练 T1

```bash
python scripts/train_dinov3_boundary_head.py --config configs/train/dinov3_boundary_head_t1.yaml
```

T1 使用原 V3 attention head 作为当前最新迭代：DINOv3 frozen features + attention boundary/foreground head + train-only augmentation + SAM2 box/positive/negative prompts。V1/V2 暂时搁置，只保留为历史配置。

当前配置只对 train 做在线增强，`augment_factor: 4` 会把 1161 张 train tile 扩展为每轮 4644 个训练样本。Val/Test 不增强。由于增强后的图像每次不同，train 特征缓存会自动关闭；Val 特征缓存仍可使用。

T1 默认使用较高吞吐配置：

```text
batch_size: 256
num_workers: 8
amp: true
amp_dtype: bfloat16
preprocess_in_workers: true
prefetch_factor: 2
```

这版会在 DataLoader worker 内完成 DINOv3 的 224 resize 和 SAT-493M normalize，避免主进程 Hugging Face processor 卡住 GPU。

如果出现 OOM，先降 batch：

```bash
python scripts/train_dinov3_boundary_head.py \
  --config configs/train/dinov3_boundary_head_t1.yaml \
  --batch-size 128 \
  --num-workers 4 \
  --prefetch-factor 2
```

如果启动稳定，想进一步压缩每轮训练时间，优先只提高 batch，不再提高 prefetch：

```bash
python scripts/train_dinov3_boundary_head.py \
  --config configs/train/dinov3_boundary_head_t1.yaml \
  --batch-size 384 \
  --num-workers 8 \
  --prefetch-factor 2
```

如果需要回退到 Hugging Face processor 主进程预处理：

```bash
python scripts/train_dinov3_boundary_head.py \
  --config configs/train/dinov3_boundary_head_t1.yaml \
  --no-preprocess-in-workers
```

## 8. 每次实验前记录

```bash
git rev-parse HEAD
git submodule status
nvidia-smi
python -V
pip freeze > outputs/requirements.lock.txt
```

这些信息之后用来复现实验。
