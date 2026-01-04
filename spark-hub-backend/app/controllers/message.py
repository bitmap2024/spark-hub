from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.message import Message
from app.models.user import User
from app.schemas.message import MessageCreate, MessageResponse
from app.auth import get_current_user

router = APIRouter()

@router.get("/conversations", response_model=List[dict])
def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 获取所有与当前用户相关的消息
    messages = db.query(Message).filter(
        (Message.sender_id == current_user.id) | 
        (Message.receiver_id == current_user.id)
    ).order_by(Message.created_at.desc()).all()
    
    # 整理对话列表
    conversations = {}
    for msg in messages:
        other_id = msg.receiver_id if msg.sender_id == current_user.id else msg.sender_id
        if other_id not in conversations:
            other_user = db.query(User).filter(User.id == other_id).first()
            if not other_user:
                continue
                
            conversations[other_id] = {
                "user": {
                    "id": other_user.id,
                    "username": other_user.username,
                    "avatar": other_user.avatar
                },
                "last_message": msg.to_dict(),
                "unread_count": 0
            }
        
        # 更新未读消息数
        if not msg.is_read and msg.sender_id != current_user.id:
            conversations[other_id]["unread_count"] += 1
    
    return list(conversations.values())

@router.get("/messages/{user_id}", response_model=List[MessageResponse])
def get_messages(
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 获取与特定用户的对话消息
    messages = db.query(Message).filter(
        ((Message.sender_id == current_user.id) & (Message.receiver_id == user_id)) |
        ((Message.sender_id == user_id) & (Message.receiver_id == current_user.id))
    ).order_by(Message.created_at.asc()).all()
    
    # 标记消息为已读
    for msg in messages:
        if not msg.is_read and msg.sender_id == user_id:
            msg.is_read = True
    db.commit()
    
    return messages

@router.post("/messages", response_model=MessageResponse)
def send_message(
    message: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 检查接收者是否存在
    receiver = db.query(User).filter(User.id == message.receiver_id).first()
    if not receiver:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="接收者不存在"
        )
    
    # 创建新消息
    db_message = Message(
        sender_id=current_user.id,
        receiver_id=message.receiver_id,
        content=message.content
    )
    db.add(db_message)
    db.commit()
    db.refresh(db_message)
    
    return db_message 