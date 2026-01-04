# 多阶段构建
# 构建阶段
FROM python:3.10-slim as builder

WORKDIR /app

# 安装构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制并安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 最终阶段
FROM python:3.10-slim

WORKDIR /app

# 复制构建阶段的虚拟环境
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 创建非root用户
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app:${PATH}"

# 复制应用代码
COPY . /app/

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["sh", "-c", "gunicorn -c deploy/gunicorn_conf.py app.main:app"]

# 健康检查
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1 