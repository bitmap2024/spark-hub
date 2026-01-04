from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel


# 论文基础模型
class PaperBase(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    publish_date: str
    doi: Optional[str] = None
    url: Optional[str] = None


# 用于创建论文的模型
class PaperCreate(PaperBase):
    knowledge_base_id: int


# 用于更新论文的模型
class PaperUpdate(BaseModel):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    publish_date: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None


# 数据库内论文模型
class PaperInDBBase(PaperBase):
    id: int
    knowledge_base_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True


# 返回给API客户端的论文模型
class Paper(PaperInDBBase):
    pass 