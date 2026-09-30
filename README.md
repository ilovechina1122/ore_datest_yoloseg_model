# 🪨 Ore Dataset & YOLO11 Instance Segmentation & Deployment
基于 YOLO11-seg 的多类别矿石/矿物实例分割与端到端高性能部署工程

[![YOLO11](https://img.shields.io/badge/YOLO-v11-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![TensorRT](https://img.shields.io/badge/TensorRT-8.6+-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![ONNXRuntime](https://img.shields.io/badge/ONNXRuntime-1.16+-005CED.svg)](https://onnxruntime.ai/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 📌 项目概述

本项目是一个完整的**矿石/矿物视觉感知与端到端边缘部署工程**，主要针对地质勘探、选矿流程与矿物分拣等场景。

项目基于最新的 **YOLO11-seg** 实例分割架构，实现了复杂背景下 5 类关键矿物的高精度轮廓分割与目标定位。同时，工程内集成了从**数据集标注与转换**、**模型训练与优化**，到 **ONNXRuntime (Python) / TensorRT (C++) 双端高性能部署**的完整链路。

---

## 💎 支持矿物类别

模型当前支持以下 5 类常见矿石/矿物的实例分割识别：

| 类别 ID | 矿物英文名 | 中文名 | 常见特征与场景 |
| :---: | :---: | :---: | :--- |
| **0** | `quartz` | **石英** | 半透明至白色块状、玻璃光泽、高硬度脉石 |
| **1** | `pyrite` | **黄铁矿** | 金属光泽、浅黄铜色、立方体或粒状集合体（俗称“愚人金”） |
| **2** | `malachite` | **孔雀石** | 翠绿色或深绿色条带状、同心环带结构 |
| **3** | `bornite` | **斑铜矿** | 铜红至暗铜色、新鲜断口暗褐色、氧化后具蓝紫色斑驳锖色 |
| **4** | `basalt` | **玄武岩** | 暗黑色/暗灰色基性喷出岩、致密块状或气孔结构 |

---

## 📊 模型性能指标

采用 **YOLO11n-seg** 预训练模型，输入分辨率 `640x640`，在自建与开源矿石数据集上进行训练与验证：

| 评估项目 | 类别数 | 目标框 Precision (B) | 目标框 Recall (B) | 目标框 mAP@50 (B) | 分割 Mask Precision (M) | 分割 Mask Recall (M) | 分割 Mask mAP@50 (M) | 分割 Mask mAP@50-95 (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **YOLO11n-seg** | 5 | **86.9%** | **74.1%** | **81.1%** | **86.2%** | **73.7%** | **80.4%** | **66.1%** |

* 权重体积：ONNX 仅 **11.1 MB**，FP32 PT 权重约 **22 MB**，极为轻量，非常适合工业嵌入式与边缘端落地。

---

## 📁 目录结构导览

```plaintext
ore_datest_yoloseg_model/
│
├── annotated_dataset/                 # 自标标注矿物分割数据集
│   ├── classes.txt                    # 类别清单
│   ├── ore_data.yaml                  # YOLO 数据集配置文件
│   ├── images/                        # 图像数据 (train / val / test)
│   ├── labels/                        # 对应 YOLO 多边形分割标注 (txt 格式)
│   └── 格式转换/                      # 数据标注转换工具集 (VIA JSON -> YOLO txt)
│       └── convert_via_to_yolo.py
│
├── ore_dataset_open/                  # 开源及多源融合矿石数据集 (VOC/JSON格式标注)
│   ├── train/ / valid/ / test/
│   └── *.json                         # 各类别切分标注文件
│
├── yolo_code/                         # 模型训练与导出工程
│   ├── ore_train_003.py               # 模型训练主脚本 (AdamW 优化器、自动保存策略)
│   ├── ore_test_003.py                # 离线验证与评估脚本
│   ├── ore_test_oo3_online.py         # 在线推理脚本
│   ├── export_onnx.py                 # ONNX 导出工具 (支持动态 batch、简化与算子指定)
│   └── ore_seg_project/               # 训练日志、曲线与各阶段权重 (best.pt / epoch*.pt)
│
└── ore_deploy/                        # 端到端高性能推理部署
    ├── best.onnx                      # 优化导出的 ONNX 模型
    ├── onnx_seg_infer.py              # Python ONNXRuntime 极速推理实现 (含 Mask 后处理)
    ├── official_onnx_infer.py         # 官方推理比对脚本
    ├── test_img/                      # 测试样本图片与推理输出目录
    ├── onnx_result/                   # ONNX 推理可视化结果保存目录
    └── trt_seg_infer/                 # TensorRT C++ 现代 API 极速部署工程
        ├── CMakeLists.txt             # 跨平台构建脚本
        ├── main.cpp                   # C++17 TensorRT 8.6 前后处理与 Mask 解算
        └── build/                     # 编译生成目标
```

---

## 🚀 快速上手

### 1. 环境准备

建议使用 Python 3.8+ 及 PyTorch 环境：

```bash
# 安装基础依赖与 Ultralytics
pip install ultralytics torch torchvision opencv-python onnx onnxruntime
```

### 2. 模型训练

进入 `yolo_code` 目录执行训练：

```bash
cd yolo_code
python ore_train_003.py
```
> [!TIP]
> 可通过修改 `ore_train_003.py` 中的 `data` 参数指定数据集配置，支持自适应早停（`patience=20`）以避免过拟合。

### 3. 模型导出为 ONNX

训练完成后，使用以下脚本将最优权重导出为高效的 ONNX 模型：

```bash
python export_onnx.py
```

---

## ⚡ 高性能部署 (Deployment)

### 方案 A：Python ONNXRuntime 推理

适用于快速验证、跨平台 PC 部署或开发板端（如树莓派/RK3588）：

```bash
cd ore_deploy
python onnx_seg_infer.py
```

### 方案 B：TensorRT C++ 极速推理

针对 NVIDIA Jetson 嵌入式平台或边缘工控机（GTX/RTX 系列 GPU），提供基于 **TensorRT 8.6+ 现代 API** 的 C++ 实现：

* **输入规格**：`1x3x640x640`
* **网络输出**：
  * 检测头 `output0`：`1x41x8400`（4 边界框坐标 + 5 类别置信度 + 32 原型 Mask 系数）
  * 分割头 `output1`：`1x32x160x160`（Mask 原型特征图）

#### 编译与执行：
```bash
cd ore_deploy/trt_seg_infer
mkdir build && cd build
cmake ..
make -j$(nproc)

# 执行推理：./trt_seg_infer <engine_path> <input_image_folder> [output_folder]
./trt_seg_infer ore_seg_best.engine ../test_img/ ../test_img/results
```

---

## 📜 许可证

本项目遵循 [MIT License](LICENSE) 开源协议。
