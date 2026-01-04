from typing import List, Optional
from pydantic import BaseModel
from datetime import datetime

from app.schemas.paper import Paper, PaperCreate

# 标签模型
class TagBase(BaseModel):
    name: str

class TagCreate(TagBase):
    pass

class TagUpdate(TagBase):
    pass

class Tag(TagBase):
    id: int

    class Config:
        orm_mode = True

# 论文模型
class PaperBase(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    publish_date: str
    doi: Optional[str] = None
    url: Optional[str] = None

class PaperCreate(PaperBase):
    tags: Optional[List[str]] = None

class PaperUpdate(PaperBase):
    title: Optional[str] = None
    authors: Optional[List[str]] = None
    abstract: Optional[str] = None
    publish_date: Optional[str] = None
    tags: Optional[List[str]] = None

class PaperInDBBase(PaperBase):
    id: int
    knowledge_base_id: int
    created_at: datetime

    class Config:
        orm_mode = True

class Paper(PaperInDBBase):
    tags: List[Tag] = []

# 知识库模型
class KnowledgeBaseBase(BaseModel):
    title: str
    description: str
    user_id: int
    tags: Optional[List[str]] = []

class KnowledgeBaseCreate(KnowledgeBaseBase):
    pass

class KnowledgeBaseUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

# 数据库中知识库的属性
class KnowledgeBaseInDBBase(KnowledgeBaseBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    stars: int = 0
    forks: int = 0

    class Config:
        orm_mode = True

# 返回给API的知识库模型（不包含论文）
class KnowledgeBase(KnowledgeBaseInDBBase):
    pass

# 带有论文的知识库模型
class KnowledgeBaseWithPapers(KnowledgeBase):
    papers: List[Paper] = []

# 创建知识库时包含论文
class KnowledgeBaseCreateWithPapers(KnowledgeBaseCreate):
    papers: Optional[List[PaperCreate]] = None

# 用户简要信息
class User(BaseModel):
    id: int
    username: str
    avatar: str

    class Config:
        orm_mode = True

# 带有用户信息的知识库
class KnowledgeBaseWithUser(KnowledgeBase):
    owner: Optional[User] = None

# 知识库列表项
class KnowledgeBaseListItem(BaseModel):
    id: int
    title: str
    description: str
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    stars: int = 0
    forks: int = 0
    papers_count: int = 0
    tags: List[Tag] = []
    is_starred: bool = False

    class Config:
        orm_mode = True 