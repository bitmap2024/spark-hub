#!/usr/bin/env python3
import os
import uvicorn
import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler

# 配置日志
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 创建日志处理器
file_handler = RotatingFileHandler(
    f"{log_dir}/app.log", 
    maxBytes=10485760,  # 10MB
    backupCount=5
)
console_handler = logging.StreamHandler()

# 配置日志格式
log_format = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(log_format)
console_handler.setFormatter(log_format)

# 获取根日志记录器并添加处理器
logger = logging.getLogger("uvicorn")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

if __name__ == "__main__":
    try:
        # 预加载应用程序以捕获导入错误
        from app.main import app
        
        # 获取环境变量，如果没有则使用默认值
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", "8000"))
        env = os.getenv("ENVIRONMENT", "development")
        workers = int(os.getenv("WORKERS", "1"))
        
        # 只在开发环境启用reload
        reload = True  # 根据需求保留reload=True
        
        # 启动服务
        uvicorn.run(
            "app.main:app", 
            host=host, 
            port=port, 
            reload=reload,
            workers=workers,
            log_level="info",
            access_log=True
        )
    except Exception as e:
        logger.error(f"启动服务时发生错误: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
