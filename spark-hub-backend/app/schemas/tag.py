from typing import List, Optional
from pydantic import BaseModel


# 标签基础模型
class TagBase(BaseModel):
    name: str


# 用于创建标签的模型
class TagCreate(TagBase):
    pass


# 用于更新标签的模型
class TagUpdate(TagBase):
    pass


# 数据库内标签模型
class TagInDBBase(TagBase):
    id: int

    class Config:
        orm_mode = True


# 返回给API客户端的标签模型
class Tag(TagInDBBase):
    pass 