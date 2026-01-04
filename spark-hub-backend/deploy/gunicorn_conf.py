#!/usr/bin/env python3
# Gunicorn配置文件

import multiprocessing
import os

# 工作进程数 - 通常设置为CPU核心数的2-4倍
workers_per_core_str = os.getenv("WORKERS_PER_CORE", "2")
workers_per_core = float(workers_per_core_str)
cores = multiprocessing.cpu_count()
default_web_concurrency = workers_per_core * cores
web_concurrency = int(os.getenv("WEB_CONCURRENCY", default_web_concurrency))

# 确保至少有一个worker
workers = max(web_concurrency, 1)

# 设置worker类为uvicorn的worker
worker_class = "uvicorn.workers.UvicornWorker"

# 绑定地址和端口
host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "8000")
bind = f"{host}:{port}"

# 其他Gunicorn配置
keepalive = 120
timeout = 120
graceful_timeout = 30
worker_connections = 1000

# 日志配置
accesslog = os.getenv("ACCESS_LOG", "logs/access.log")
errorlog = os.getenv("ERROR_LOG", "logs/error.log")
loglevel = os.getenv("LOG_LEVEL", "info")

# 确保日志目录存在
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir) 