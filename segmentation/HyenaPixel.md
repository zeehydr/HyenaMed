# HyenaPixel

**HyenaPixel** is a lightweight vision architecture designed for efficient medical image analysis. This repository contains the implementation and experimental configurations for **skin lesion segmentation** using HyenaPixel with the **SegFormer** decoder and MMSegmentation framework.

The experiments are primarily conducted on the **ISIC 2016** and **ISIC 2017** skin lesion segmentation datasets.

---

## 📌 Repository Structure

```text
HyenaPixel-main/
│
├── mmsegmentation/
│   ├── configs/
│   │   └── hpx_former_s18/
│   │       ├── segformer_hpx-former-s18-90k_isic2016-224x224.py
│   │       ├── segformer_hpx-former-s18-90k_isic2016-256x256.py
│   │       ├── segformer_hpx-former-s18-90k_isic2017-224x224.py
│   │       └── lraspphead_hpx-former-s18-90k_isic2016-224x224.py
│   │
│   ├── tools/
│   │   ├── train.py
│   │   ├── test.py
│   │   └── analysis_tools/
│   │
│   └── mmseg/
│
├── work_dirs/
├── results/
├── setup.py
└── README.md
```

---

# ⚙️ Installation

## 1. Create Conda Environment

```bash
conda create -n hyenapixel python=3.10
conda activate hyenapixel
```

## 2. Install PyTorch

The experiments use **PyTorch 2.1.0 with CUDA 11.8**.

```bash
pip install torch==2.1.0 torchvision==0.16.0 \
--index-url https://download.pytorch.org/whl/cu118
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

Expected configuration:

```text
PyTorch: 2.1.0
CUDA: 11.8
CUDA available: True
```

## 3. Install MMCV

Install the compatible MMCV version:

```bash
pip install mmcv==2.1.0 \
-f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
```

## 4. Install the Repository

From the project root:

```bash
pip install -e .
```

---

# 🧩 Segmentation Framework

The segmentation experiments are implemented using **MMSegmentation**.

The main model configuration combines:

```text
HyenaPixel / HPX-Former
          │
          ▼
      Backbone
          │
          ▼
     SegFormer Head
          │
          ▼
   Binary Segmentation
```

The main configuration files are located at:

```text
mmsegmentation/configs/hpx_former_s18/
```

---

# 🗂️ Dataset

The experiments use skin lesion segmentation datasets including:

- ISIC 2016
- ISIC 2017

The dataset paths should be configured inside the corresponding MMSegmentation configuration files.

For example:

```text
mmsegmentation/configs/hpx_former_s18/
```

Before training, make sure that the dataset paths in the configuration point to your local dataset directories.

---

# 🚀 Training

## ISIC 2016 — 224 × 224

```bash
python mmsegmentation/tools/train.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-224x224.py
```

---

## ISIC 2016 — 256 × 256

```bash
python mmsegmentation/tools/train.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-256x256.py
```

---

## Multi-GPU Training

For two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash mmsegmentation/tools/dist_train.sh \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-256x256.py
```

Alternatively:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
bash mmsegmentation/tools/dist_train.sh \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-256x256.py 2
```

where `2` denotes the number of GPUs.

---

## ISIC 2017 — 224 × 224

```bash
python mmsegmentation/tools/train.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2017-224x224.py
```

---

## LRASPP Head

An alternative lightweight decoder can also be trained using:

```bash
python mmsegmentation/tools/train.py \
mmsegmentation/configs/hpx_former_s18/lraspphead_hpx-former-s18-90k_isic2016-224x224.py
```

---

# 🧪 Testing

## ISIC 2016 — 256 × 256

After training, evaluate the checkpoint using:

```bash
python mmsegmentation/tools/test.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-256x256.py \
work_dirs/segformer_hpx-former-s18-90k_isic2016-256x256/iter_90000.pth
```

To save segmentation results:

```bash
python mmsegmentation/tools/test.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-256x256.py \
work_dirs/segformer_hpx-former-s18-256x256/iter_90000.pth \
--show-dir results/segformer_hpx-former-s18-256x256/test/ \
--work-dir results/segformer_hpx-former-s18-256x256/test/
```

> **Note:** Make sure the configuration filename matches the configuration actually used for training. Checkpoints and configuration files must use compatible image resolutions.

---

## ISIC 2016 — 224 × 224

```bash
python mmsegmentation/tools/test.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-224x224.py \
work_dirs/segformer_hpx-former-s18-224x224/iter_40000.pth
```

For the final checkpoint:

```bash
python mmsegmentation/tools/test.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-224x224.py \
work_dirs/segformer_hpx-former-s18-224x224/iter_90000.pth
```

---

# 📊 FLOPs and Parameters

MMSegmentation provides an analysis tool for calculating computational complexity.

## ISIC 2016

```bash
python mmsegmentation/tools/analysis_tools/get_flops.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-224x224.py \
--shape 256
```

---

## ISIC 2017

```bash
python mmsegmentation/tools/analysis_tools/get_flops.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2017-224x224.py \
--shape 256
```

The tool reports:

- FLOPs
- Number of parameters
- Input/output information

> **Important:** The `--shape` argument determines the actual input resolution used for complexity analysis. Therefore, use the same resolution when comparing models.

---

# ⚡ Inference Benchmark

To measure inference performance:

## ISIC 2016

```bash
python mmsegmentation/tools/analysis_tools/benchmark.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-224x224.py \
work_dirs/segformer_hpx-former-s18-90k_isic2016-224x224/iter_90000.pth
```

---

## ISIC 2017

```bash
python mmsegmentation/tools/analysis_tools/benchmark.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2017-224x224.py \
work_dirs/segformer_hpx-former-s18-90k_isic2017-224x224/iter_90000.pth
```

---

## Benchmarking an Intermediate Checkpoint

For example:

```bash
python mmsegmentation/tools/analysis_tools/benchmark.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-224x224.py \
work_dirs/segformer_hpx-former-s18-90k_isic2016-224x224/iter_9000.pth
```

The benchmark can be used to estimate inference speed and latency.

---

# 🏷️ ISIC Ground-Truth Label Handling

For the ISIC segmentation experiments, the ground-truth masks are converted so that the lesion region is represented by label `1`.

The relevant file is:

```text
mmsegmentation/mmseg/datasets/transforms/loading.py
```

The following modification was made around line 106:

```python
gt_semantic_seg[gt_semantic_seg == 255] = 1
```

This converts pixels with the ignore/background value `255` into the foreground class label `1`.

> **Important:** This modification is dataset/task-specific. If you use other segmentation datasets, verify the label encoding before applying this change.

---

# ⚠️ Common MMSegmentation FLOPs Error

When running:

```bash
python mmsegmentation/tools/analysis_tools/get_flops.py ...
```

you may encounter:

```text
ValueError: "input_shape" and "inputs" cannot be both set.
```

This is related to the FLOPs analysis utility and the arguments passed to the model analysis function.

In the affected `get_flops.py` version, comment out **line 89 or line 90**, depending on the exact version of the file.

After making the change, run the FLOPs command again.

> The exact line number can differ between MMSegmentation versions, so check the surrounding code rather than relying only on the line number.

---

# 🎯 Grad-CAM Visualization

Grad-CAM can be used to visualize the regions of the input image that contribute to the segmentation prediction.

During Grad-CAM execution, if CUDA handling causes an issue, the following line in the visualization script can be disabled:

```python
# use_cuda=torch.cuda.is_available()
```

In the current implementation this corresponds approximately to **line 118**, although the line number may change depending on the repository version.

After commenting the line, run the Grad-CAM visualization script normally.

---

# 🔬 Fast-SCNN Baseline

Fast-SCNN can be evaluated as a lightweight baseline.

For example:

```bash
python mmsegmentation/tools/test.py \
mmsegmentation/configs/fastscnn/fast_scnn_8xb4-90k_isic2016-256x256.py \
work_dirs/fast_scnn_8xb4-90k_isic2016-256x256/iter_40000.pth
```

This allows HyenaPixel to be compared against another lightweight semantic segmentation architecture.

---

# 📁 Output Directories

Training checkpoints are normally stored under:

```text
work_dirs/
```

For example:

```text
work_dirs/
└── segformer_hpx-former-s18-90k_isic2016-256x256/
    ├── iter_1000.pth
    ├── iter_10000.pth
    ├── iter_40000.pth
    ├── iter_90000.pth
    └── ...
```

Visualization and testing results can be stored under:

```text
results/
```

Example:

```text
results/
└── segformer_hpx-former-s18-90k_isic2016-256x256/
    └── test/
```

---

# 🛠️ Recommended Environment

The main environment used for the experiments is:

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.1.0 |
| TorchVision | 0.16.0 |
| CUDA | 11.8 |
| MMCV | 2.1.0 |
| Framework | MMSegmentation |

The PyTorch and MMCV versions should be kept compatible.

---

# 🔄 Complete Experimental Workflow

A typical experiment follows this workflow:

```text
1. Create Conda environment
          ↓
2. Install PyTorch + CUDA 11.8
          ↓
3. Install compatible MMCV
          ↓
4. Install HyenaPixel
          ↓
5. Configure ISIC dataset paths
          ↓
6. Train model
          ↓
7. Evaluate checkpoint
          ↓
8. Calculate FLOPs / Parameters
          ↓
9. Benchmark inference speed
          ↓
10. Generate segmentation visualizations
          ↓
11. Compare against lightweight baselines
```

---

# 📌 Example Complete Setup

```bash
conda create -n hyenapixel python=3.10
conda activate hyenapixel

pip install torch==2.1.0 torchvision==0.16.0 \
--index-url https://download.pytorch.org/whl/cu118

pip install mmcv==2.1.0 \
-f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install -e .
```

Then train:

```bash
python mmsegmentation/tools/train.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-256x256.py
```

Evaluate:

```bash
python mmsegmentation/tools/test.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-256x256.py \
work_dirs/segformer_hpx-former-s18-90k_isic2016-256x256/iter_90000.pth
```

Calculate complexity:

```bash
python mmsegmentation/tools/analysis_tools/get_flops.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-256x256.py \
--shape 256
```

Benchmark:

```bash
python mmsegmentation/tools/analysis_tools/benchmark.py \
mmsegmentation/configs/hpx_former_s18/segformer_hpx-former-s18-90k_isic2016-256x256.py \
work_dirs/segformer_hpx-former-s18-90k_isic2016-256x256/iter_90000.pth
```

---

# 📖 Citation

If you find this repository useful for your research, please cite the corresponding HyenaPixel/HyenaMed publication.

```bibtex
@inproceedings{hyenamed2026,
  title     = {HyenaMed: Lightweight Skin Lesion Segmentation via Convolutional Global Context Modeling},
  year      = {2026},
  booktitle = {Future Engineering and Technology Conference (FET)}
}
```

Please update the BibTeX entry with the final author list, DOI, and publication metadata once the official IEEE Xplore record is available.

---

# 🙏 Acknowledgements

This project builds upon the excellent open-source work from the computer vision community, particularly:

- MMSegmentation
- SegFormer
- PyTorch
- MMCV

We thank the authors and developers of these projects for making their implementations publicly available.

---

# 📜 License

Please refer to the repository license and the licenses of the third-party components used in this project.

---

# 👤 Author

**Zeeshan Haider**

Research interests:

- Computer Vision
- Medical Image Analysis
- Lightweight Deep Learning
- Skin Lesion Segmentation
- Efficient Neural Networks
- Semantic Segmentation

For questions, issues, or collaboration, please open an issue in this repository.