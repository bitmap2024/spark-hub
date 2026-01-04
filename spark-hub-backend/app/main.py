from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.api_v1.api import api_router

app = FastAPI()

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(api_router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to SparkHub API"}

# 添加健康检查端点
@app.get("/health", tags=["Health"])
async def health_check():
    """
    健康检查端点，用于Kubernetes的活性和就绪探针
    """
    return {"status": "ok"}