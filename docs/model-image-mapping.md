# Tianshu 项目镜像与模型对应关系

Tianshu（天枢）项目包含三个核心AI模型，每个模型在不同的Docker镜像中运行，具有特定的功能和依赖。

## 1. 镜像与模型概览

| 镜像名称 | 模型 | 功能 | 运行环境 |
|---------|------|------|---------|
| tianshu-backend:latest | MinerU + PaddleOCR-VL + SenseVoice | 多模态文档处理 | 生产环境 |
| tianshu-backend:dev | MinerU + PaddleOCR-VL + SenseVoice | 多模态文档处理（开发模式） | 开发环境 |
| tianshu-frontend:latest | Vue.js 前端应用 | 用户界面 | 生产环境 |

tianshu-backend基础镜像：
```
docker pull nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04
https://hub.docker.com/r/nvidia/cuda/tags?name=12.6.2-cudnn-runtime-ubuntu24.04
```

## 2. 详细对应关系

### 2.1 MinerU 模型

**功能**: 文档处理引擎
- PDF、图片完整解析
- 表格、公式识别
- GPU加速处理

**镜像中的位置**:
- **生产镜像**: `tianshu-backend:latest`
- **开发镜像**: `tianshu-backend:dev`

**依赖关系**:
- Python 3.12 环境
- PyTorch (CUDA 12.6)
- PaddlePaddle (CUDA 12.6)
- MinerU 核心库

**模型存储**:
- 自动管理，缓存在 `~/.cache/mineru/` 目录
- 首次使用时自动下载

### 2.2 PaddleOCR-VL 模型

**功能**: 多语言OCR引擎
- 支持109+种语言自动识别
- 文档方向分类
- 文本图像矫正
- 版面区域检测

**镜像中的位置**:
- **生产镜像**: `tianshu-backend:latest`
- **开发镜像**: `tianshu-backend:dev`

**依赖关系**:
- Python 3.12 环境
- PaddlePaddle GPU 3.2.0
- PaddleOCR-VL 库

**模型存储**:
- 自动管理，缓存在 `~/.paddleocr/models/` 目录
- 首次使用时自动下载（约2GB）
- 支持ModelScope和HuggingFace下载源

### 2.3 SenseVoice 模型

**功能**: 音频处理引擎
- 多语言语音识别（中/英/日/韩/粤）
- 说话人识别（Speaker Diarization）
- 情感识别
- 时间戳对齐

**镜像中的位置**:
- **生产镜像**: `tianshu-backend:latest`
- **开发镜像**: `tianshu-backend:dev`

**依赖关系**:
- Python 3.12 环境
- PyTorch (CUDA 12.6) + torchaudio
- FunASR 框架
- SenseVoice 模型

**模型存储**:
- 默认缓存目录：`/app/models/sensevoice`（Docker环境中）
- 首次使用时自动从ModelScope下载
- 支持自定义缓存目录

## 3. 镜像构建与模型集成

### 3.1 生产镜像 (tianshu-backend:latest)

**构建基础**:
- 基于 `nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04`
- 包含完整的Python 3.12环境和所有依赖

**模型集成方式**:
- 模型依赖在Dockerfile中通过pip安装
- 运行时自动下载模型到指定缓存目录
- 支持通过环境变量配置下载源

**挂载目录**:
- `/app/models`: 模型存储目录（读写）
- `/app/data/uploads`: 上传文件目录（只读）
- `/app/data/output`: 输出结果目录（读写）

### 3.2 开发镜像 (tianshu-backend:dev)

**构建基础**:
- 基于 `nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04`
- 仅安装依赖，不包含应用代码

**模型集成方式**:
- 依赖在构建时安装
- 代码通过卷挂载提供（支持热重载）
- 模型下载和缓存机制与生产环境一致

**挂载目录**:
- `/app/backend`: 源代码目录（读写）
- `/app/models`: 模型存储目录（读写）
- `/app/data/uploads`: 上传文件目录（只读）
- `/app/data/output`: 输出结果目录（读写）

### 3.3 前端镜像 (tianshu-frontend:latest)

**功能**: Vue.js前端应用
- 用户界面展示
- 任务提交和管理
- 结果预览和下载

**模型关系**: 
- 不直接包含AI模型
- 通过API与后端模型交互

## 4. 模型下载与配置

### 4.1 下载源配置

通过环境变量控制模型下载源：

```bash
# 使用ModelScope（国内推荐）
MODEL_DOWNLOAD_SOURCE=modescope

# 使用HuggingFace
MODEL_DOWNLOAD_SOURCE=huggingface

# 自动选择（默认）
MODEL_DOWNLOAD_SOURCE=auto
```

### 4.2 国内镜像配置

```bash
# HuggingFace镜像
HF_ENDPOINT=https://hf-mirror.com
```

### 4.3 模型缓存目录

```bash
# MinerU模型缓存
MINERU_CACHE_DIR=~/.cache/mineru/

# PaddleOCR模型缓存
PADDLEOCR_CACHE_DIR=~/.paddleocr/models/

# SenseVoice模型缓存
SENSEVOICE_CACHE_DIR=/app/models/sensevoice
```

## 5. GPU 支持与要求

### 5.1 硬件要求

- **MinerU**: 支持GPU加速
- **PaddleOCR-VL**: 仅支持GPU推理，要求Compute Capability ≥ 8.5
- **SenseVoice**: 支持GPU加速

### 5.2 Docker GPU配置

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

## 6. 故障排查

### 6.1 模型下载失败

1. 检查网络连接
2. 配置国内镜像源
3. 手动下载模型到挂载目录

### 6.2 GPU不可用

1. 检查NVIDIA驱动版本
2. 验证CUDA支持
3. 确认Docker GPU配置

### 6.3 模型加载错误

1. 检查模型缓存目录权限
2. 验证依赖安装
3. 查看日志获取详细错误信息