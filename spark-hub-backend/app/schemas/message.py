from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

# 消息基础模型
class MessageBase(BaseModel):
    content: str

# 用于创建消息的模型
class MessageCreate(MessageBase):
    receiver_id: int

# 更新消息时的输入数据
class MessageUpdate(MessageBase):
    is_read: Optional[bool] = None

# 数据库内消息模型
class MessageInDBBase(MessageBase):
    id: int
    sender_id: int
    receiver_id: int
    is_read: bool
    created_at: datetime

    class Config:
        orm_mode = True

# 返回给API客户端的消息模型
class Message(MessageInDBBase):
    pass

# 用户基本信息
class UserBasic(BaseModel):
    id: int
    username: str
    avatar: str

    class Config:
        orm_mode = True

# 会话模型
class Conversation(BaseModel):
    id: int  # 会话对象的ID
    participants: List[int]  # 参与者ID列表
    last_message: Message
    unread_count: int
    other_user: Optional[UserBasic] = None

    class Config:
        orm_mode = True 