# DINOv3 + SAM2 桥接实验

这组实验只用于打通 DINOv3 特征与 SAM2 prompt/分割的数据流，不把当前
`sam2-dataset-1024` 的结果当作最终精度结论。当前数据集存在 tile 重叠率高、
田块漏标注等问题，适合先做链路验证和可视化诊断。

## 数据目录

服务器上建议保持如下结构：

```text
data/sam2-dataset-1024/
  All/
    Image/
    Instance/
```

脚本的 `--dataset-root` 可以指向 `data/sam2-dataset-1024`，也可以直接指向
包含 `Image/` 和 `Instance/` 的目录。

## 1. Oracle Prompt + SAM2

这个实验从 GT 实例 mask 自动生成 box 和正点 prompt，再调用 SAM2 预测 mask。
它的作用是确认：如果 prompt 足够准，SAM2 在这批遥感切片上的上限大概如何。

先跑 10 张：

```bash
python scripts/run_oracle_sam2_prompts.py \
  --dataset-root data/sam2-dataset-1024 \
  --prompt-mode box_point \
  --limit 10 \
  --min-area 256 \
  --save-overlays 5
```

输出目录：

```text
outputs/oracle_sam2_prompts/
  oracle_box_point.jsonl
  oracle_box_point_summary.json
  overlays/
```

可以对比三种 prompt：

```bash
python scripts/run_oracle_sam2_prompts.py --dataset-root data/sam2-dataset-1024 --prompt-mode box --limit 50
python scripts/run_oracle_sam2_prompts.py --dataset-root data/sam2-dataset-1024 --prompt-mode point --limit 50
python scripts/run_oracle_sam2_prompts.py --dataset-root data/sam2-dataset-1024 --prompt-mode box_point --limit 50
```

## 2. DINOv3 特征边界热图

这个实验读取 DINOv3 patch features，计算相邻 patch 的 cosine distance，并保存
边界热图。它的作用是判断 DINOv3 卫星权重是否能显出田埂、道路、沟渠等边界。

先跑 20 张：

```bash
python scripts/export_dinov3_boundary_maps.py \
  --dataset-root data/sam2-dataset-1024 \
  --limit 20
```

输出目录：

```text
outputs/dinov3_boundary_maps/
  boundary_maps.jsonl
  gray/
  overlay/
```

如果 overlay 中红色高响应大致贴合田块边界，再继续接 watershed 或连通域生成
SAM2 prompt；如果高响应主要来自纹理噪声，就先不要把它接入 SAM2。

## 注意事项

- 先小批量跑通，再跑全量 1170 张。
- 当前数据集有漏标注，IoU 只看链路和相对变化，不作为最终模型指标。
- 50% overlap 会导致随机划分泄漏，后续训练或验证要按 tile 文件名中的 r/c 坐标做空间划分。
- `outputs/`、`data/`、`weights/` 都不进入 Git。
