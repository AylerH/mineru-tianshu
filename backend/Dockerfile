# 企业级 AI 数据预处理平台 - 支持 GPU 加速的多模态文档处理服务
#
# 构建方式:
#   docker build -t tianshu-backend:latest -f backend/Dockerfile .
#
# 运行方式:
#   docker run --gpus all -p 8000:8000 tianshu-backend:latest

# ============================================================================
# Stage 1: Base Image with CUDA Support
# ============================================================================
FROM m.daocloud.io/docker.io/nvidia/cuda:12.6.2-cudnn-runtime-ubuntu24.04 AS base

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 更新软件包索引并安装证书，同时配置USTC镜像源作为备用
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get update && \
    apt-get install -y apt-transport-https ca-certificates || true

# 尝试使用USTC镜像源，如果失败则回退到官方源
RUN echo "deb https://mirrors.ustc.edu.cn/debian/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release) main contrib non-free" > /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/debian/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release)-updates main contrib non-free" >> /etc/apt/sources.list && \
    echo "deb https://mirrors.ustc.edu.cn/debian-security/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release)-security main contrib non-free" >> /etc/apt/sources.list || \
    (echo "Failed to configure USTC mirror, using default sources" && \
     echo "deb http://archive.ubuntu.com/ubuntu/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release) main restricted universe multiverse" > /etc/apt/sources.list && \
     echo "deb http://archive.ubuntu.com/ubuntu/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release)-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
     echo "deb http://archive.ubuntu.com/ubuntu/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release)-security main restricted universe multiverse" >> /etc/apt/sources.list)

# 更新软件包索引
RUN apt-get update || (echo "Retrying with default sources..." && \
    echo "deb http://archive.ubuntu.com/ubuntu/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release) main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release)-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://archive.ubuntu.com/ubuntu/ $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release)-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update)

# 分批安装系统依赖以避免超时（减少一次性安装的包数量）
RUN apt-get install -y --no-install-recommends \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgl1 \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgstreamer1.0-0 \
        ffmpeg \
        && rm -rf /var/lib/apt/lists/*

# 确保 python3 指向正确的版本
RUN ln -sf /usr/bin/python3 /usr/bin/python

# 创建虚拟环境并激活
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级 pip（使用国内镜像源，增加超时时间）
# 在虚拟环境中升级包，避免系统包管理冲突
RUN pip install --upgrade pip setuptools wheel \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --default-timeout=300

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base AS dependencies

WORKDIR /tmp

# 复制 requirements.txt
COPY backend/requirements.txt /tmp/requirements.txt

# 分步安装依赖（解决 PyTorch + PaddlePaddle CUDA 冲突）
# Step 1: 安装 PaddlePaddle GPU 版本（~1.8GB，耐心等待）
RUN pip install paddlepaddle-gpu==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    --break-system-packages \
    --default-timeout=600 \
    --retries 5

# Step 2: 安装 PyTorch (CUDA 12.6, ~2GB, 包含 torchaudio for SenseVoice)
RUN pip install torch==2.6.0+cu126 torchvision==0.21.0+cu126 torchaudio==2.6.0+cu126 \
    --index-url https://download.pytorch.org/whl/cu126 \
    --break-system-packages \
    --default-timeout=600 \
    --retries 5

# Step 3: 预先安装 Python 3.12 关键依赖
RUN pip install "kiwisolver>=1.4.5" "Pillow>=11.0.0" \
    "numpy>=1.26.0,<2.0.0" "setuptools>=75.0.0" "lxml>=5.3.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=300 \
    --retries 5

# Step 4: 安装其他依赖（跳过已安装的 torch 和 paddle）
RUN pip install -r /tmp/requirements.txt \
    --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=300 \
    --retries 5 || true

RUN pip install --break-system-packages \
    --default-timeout=300 \
    --retries 5 \
    https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl


# Step 5: 补充安装缺失的依赖
RUN pip install -r /tmp/requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --break-system-packages \
    --default-timeout=300 \
    --retries 5

# 验证安装（仅检查导入，不验证 CUDA，因为构建时无 GPU）
RUN python -c "import sys; print('Python:', sys.version)" && \
    python -c "import torch; print('✅ PyTorch:', torch.__version__)" && \
    python -c "import transformers; print('✅ Transformers:', transformers.__version__)" && \
    echo "⚠️  Paddle CUDA 验证将在运行时进行（构建时无 GPU 访问）"

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

# 创建应用目录
WORKDIR /app

# 复制后端代码
COPY backend/ /app/backend/
COPY pyproject.toml /app/

# 复制启动脚本
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# 创建必要的目录
RUN mkdir -p /app/models /app/data /app/uploads /app/output /app/logs

# 设置工作目录为后端
WORKDIR /app/backend

# 暴露端口
# 8000: API Server
# 8001: LitServe Worker
# 8002: MCP Server
EXPOSE 8000 8001 8002

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 使用 entrypoint 脚本
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# 默认启动 API Server
# 第一个参数是服务类型（api/worker/mcp），用于 entrypoint 判断
CMD ["api", "python", "api_server.py"]
