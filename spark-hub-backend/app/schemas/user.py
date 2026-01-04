from typing import List, Optional
from pydantic import BaseModel, EmailStr
from datetime import datetime

# 用户基础模型
class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    is_active: Optional[bool] = True
    is_superuser: bool = False
    avatar: Optional[str] = None
    location: Optional[str] = None
    experience: Optional[str] = None
    gender: Optional[str] = None
    age: Optional[int] = None
    school: Optional[str] = None

# 用于创建用户的模型
class UserCreate(UserBase):
    email: EmailStr
    username: str
    password: str

# 用于更新用户的模型
class UserUpdate(UserBase):
    password: Optional[str] = None

# 数据库内用户模型
class UserInDBBase(UserBase):
    id: int
    email: EmailStr
    username: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

# 返回给API客户端的用户模型
class User(UserInDBBase):
    followers: int = 0
    following: int = 0

# 带有更多详细信息的用户模型
class UserDetail(User):
    followingList: Optional[List[int]] = None
    likedKnowledgeBases: Optional[List[int]] = None

# 存储在数据库中的用户模型（含密码）
class UserInDB(UserInDBBase):
    hashed_password: str

# 返回给API的用户列表项
class UserListItem(UserInDBBase):
    followers_count: int = 0
    following_count: int = 0
    is_following: bool = False

# 用户登录
class UserLogin(BaseModel):
    email: EmailStr
    password: str

# 令牌
class Token(BaseModel):
    access_token: str
    token_type: str

# 令牌数据
class TokenData(BaseModel):
    username: Optional[str] = None 