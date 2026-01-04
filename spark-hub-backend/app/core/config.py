import os
import secrets
from typing import Any, Dict, List, Optional, Union
from pydantic import AnyHttpUrl, EmailStr, PostgresDsn, field_validator
from pydantic_settings import BaseSettings
from urllib.parse import quote_plus

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Spark Hub API"
    
    # CORS配置
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # 数据库配置 - 使用PostgreSQL
    SQLALCHEMY_DATABASE_URI: Optional[str] = os.getenv("DATABASE_URL")
    
    # JWT配置
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    
    # 超级用户配置
    FIRST_SUPERUSER_EMAIL: EmailStr = "admin@sparkhub.com"
    FIRST_SUPERUSER_PASSWORD: str = "admin"
    FIRST_SUPERUSER_USERNAME: str = "admin"
    
    @field_validator("SQLALCHEMY_DATABASE_URI", mode="before")
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        # 如果有提供数据库URL且是PostgreSQL连接，直接使用
        if v and v.startswith("postgresql"):
            return v
            
        # 否则使用环境变量构建PostgreSQL连接
        postgres_user = os.getenv("POSTGRES_USER", "bitmap")
        postgres_password = os.getenv("POSTGRES_PASSWORD", "bitmap@666")
        postgres_server = os.getenv("POSTGRES_SERVER", "localhost")
        postgres_db = os.getenv("POSTGRES_DB", "spark-hub")
        
        # 对密码进行URL编码，处理特殊字符
        encoded_password = quote_plus(postgres_password)
        
        # 构建 PostgreSQL 连接 URL
        return f"postgresql://{postgres_user}:{encoded_password}@{postgres_server}/{postgres_db}"

    # 使用 model_config 代替 Config 内部类
    model_config = {
        "case_sensitive": True,
        "env_file": ".env",
        "extra": "allow",  # 允许额外的字段
    }

settings = Settings() 