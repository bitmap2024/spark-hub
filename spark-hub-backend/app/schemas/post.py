from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

# 评论基础模型
class CommentBase(BaseModel):
    content: str

# 创建评论
class CommentCreate(CommentBase):
    pass

# 更新评论
class CommentUpdate(BaseModel):
    content: Optional[str] = None
    is_approved: Optional[bool] = None
    is_hidden: Optional[bool] = None

# 显示评论的用户信息
class CommentAuthor(BaseModel):
    id: int
    username: str
    avatar: Optional[str] = None

    class Config:
        from_attributes = True

# 评论详情
class Comment(CommentBase):
    id: int
    author_id: int
    author: CommentAuthor
    post_id: int
    is_approved: bool
    is_hidden: bool
    likes: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# 帖子基础模型
class PostBase(BaseModel):
    title: str
    content: str

# 创建帖子
class PostCreate(PostBase):
    pass

# 更新帖子
class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    is_approved: Optional[bool] = None
    is_hidden: Optional[bool] = None

# 显示帖子的用户信息
class PostAuthor(BaseModel):
    id: int
    username: str
    avatar: Optional[str] = None

    class Config:
        from_attributes = True

# 简化评论用于列表
class CommentBrief(BaseModel):
    id: int
    content: str
    author_id: int
    author: CommentAuthor
    likes: int
    created_at: datetime

    class Config:
        from_attributes = True

# 帖子详情响应
class Post(PostBase):
    id: int
    author_id: int
    author: PostAuthor
    is_approved: bool
    is_hidden: bool
    likes: int
    views: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    comments: List[CommentBrief] = []

    class Config:
        from_attributes = True

# 帖子列表简化响应
class PostBrief(BaseModel):
    id: int
    title: str
    author_id: int
    author: PostAuthor
    likes: int
    views: int
    comments_count: int = Field(..., alias="comments_count")
    created_at: datetime

    class Config:
        from_attributes = True
        populate_by_name = True

# 分页响应
class PostPage(BaseModel):
    items: List[PostBrief]
    total: int
    page: int
    per_page: int
    total_pages: int 