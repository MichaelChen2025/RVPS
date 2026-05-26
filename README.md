# RVPS

Road Vision Perception System

Real-time road scene understanding using YOLO object detection, lane detection, and drivable area segmentation.

<img width="2013" height="1181" alt="ScreenShot_2026-05-26_202712_009" src="https://github.com/user-attachments/assets/f7d7bc82-9af8-4896-a9b4-d828f85bc674" />
---

[English](#english) | [中文](#中文)

---

<a id="english"></a>

# English

## Introduction

RVPS (Road Vision Perception System) is a real-time computer vision project for driving scene understanding.

This project combines:

* YOLO object detection
* Lane detection
* Drivable area segmentation
* Real-time visualization

RVPS was developed as a learning project for computer vision and autonomous driving perception systems.

---

## Features

✅ Real-time object detection

✅ Lane segmentation

✅ Drivable area segmentation

✅ Camera input

✅ Video file input

✅ GPU acceleration

✅ CUDA support

✅ Apple Silicon (MPS) support

---

## Tested Environment

Recommended:

* Python 3.10
* PyCharm (optional)
* Windows / macOS

Tested hardware:

* NVIDIA RTX 4070
* Apple Silicon (M-series)

---

## Project Structure

```text
RVPS/
├── RVPS_Main.py
├── requirements.txt
├── README.md
├── YOLOP/
│   ├── lib/
│   ├── tools/
│   ├── weights/
│   │   └── End-to-end.pth
│   └── ...
```

---

## Installation

Clone repository:

```bash
git clone https://github.com/MichaelChen2025/RVPS.git

cd RVPS
```

Create virtual environment:

Windows:

```bash
python -m venv .venv

.venv\Scripts\activate
```

Mac/Linux:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Windows CPU

Install CPU PyTorch:

```bash
pip install torch torchvision torchaudio
```

---

## Windows NVIDIA CUDA

Recommended for NVIDIA GPUs:

CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch;print(torch.cuda.is_available())"
```

Expected:

```text
True
```

---

## macOS Apple Silicon

For M1/M2/M3/M4 devices:

```bash
pip install torch torchvision torchaudio
```

Verify:

```bash
python -c "import torch;print(torch.backends.mps.is_available())"
```

Expected:

```text
True
```

---

## Run

Camera:

```python
SOURCE = 0
```

Video:

```python
SOURCE = "./videos/test.mp4"
```

Run:

```bash
python RVPS_Main.py
```

---

## Notes

YOLOv8 weights are downloaded automatically on first run.

The project already includes:

* YOLOP source code
* YOLOP pretrained weights

---

## Future Work

* Bird Eye View (BEV)
* Depth estimation
* Trajectory prediction
* Multi-camera support

---

<a id="中文"></a>

# 中文

## 项目简介

RVPS（Road Vision Perception System）是一个实时道路场景视觉感知项目。

项目结合：

* YOLO目标检测
* 车道线检测
* 可行驶区域分割
* 实时可视化

本项目主要用于学习计算机视觉和自动驾驶感知系统。

---

## 功能

✅ 实时目标检测

✅ 车道线分割

✅ 可行驶区域分割

✅ 摄像头输入

✅ 视频输入

✅ GPU加速

✅ CUDA支持

✅ Apple Silicon支持

---

## 测试环境

推荐环境：

* Python 3.10
* PyCharm（可选）
* Windows / macOS

测试硬件：

* NVIDIA RTX 4070
* Apple Silicon M系列

---

## 项目结构

```text
RVPS/
├── RVPS_Main.py
├── requirements.txt
├── README.md
├── YOLOP/
│   ├── lib/
│   ├── tools/
│   ├── weights/
│   │   └── End-to-end.pth
│   └── ...
```

---

## 安装

克隆仓库：

```bash
git clone https://github.com/MichaelChen2025/RVPS.git

cd RVPS
```

创建虚拟环境：

Windows：

```bash
python -m venv .venv

.venv\Scripts\activate
```

Mac/Linux：

```bash
python3 -m venv .venv

source .venv/bin/activate
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

## Windows CPU版本

安装CPU版PyTorch：

```bash
pip install torch torchvision torchaudio
```

---

## Windows NVIDIA CUDA版本

适用于RTX显卡：

CUDA12.1：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证：

```bash
python -c "import torch;print(torch.cuda.is_available())"
```

输出：

```text
True
```

---

## macOS Apple Silicon

适用于M1/M2/M3/M4：

```bash
pip install torch torchvision torchaudio
```

验证：

```bash
python -c "import torch;print(torch.backends.mps.is_available())"
```

输出：

```text
True
```

---

## 运行

摄像头：

```python
SOURCE = 0
```

视频：

```python
SOURCE="./videos/test.mp4"
```

运行：

```bash
python RVPS_Main.py
```

---

## 注意事项

YOLOv8模型将在首次运行时自动下载。

项目已包含：

* YOLOP源码
* YOLOP预训练权重

---

## 后续计划

* BEV鸟瞰图
* 深度估计
* 轨迹预测
* 多摄像头支持
