from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy import or_, and_
from sqlalchemy.orm import Session
from datetime import datetime

from app.crud.base import CRUDBase
from app.models.message import Message
from app.models.user import User
from app.schemas.message import MessageCreate, MessageUpdate, Conversation

class CRUDMessage(CRUDBase[Message, MessageCreate, MessageCreate]):
    def create_with_sender(
        self, db: Session, *, obj_in: MessageCreate, sender_id: int
    ) -> Message:
        """
        创建消息，指定发送者
        """
        db_obj = Message(
            content=obj_in.content,
            sender_id=sender_id,
            receiver_id=obj_in.receiver_id,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get_conversation(
        self, db: Session, *, user_id1: int, user_id2: int, skip: int = 0, limit: int = 100
    ) -> List[Message]:
        """
        获取两个用户之间的会话
        """
        return (
            db.query(self.model)
            .filter(
                or_(
                    and_(
                        Message.sender_id == user_id1,
                        Message.receiver_id == user_id2
                    ),
                    and_(
                        Message.sender_id == user_id2,
                        Message.receiver_id == user_id1
                    )
                )
            )
            .order_by(Message.created_at)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_conversations(
        self, db: Session, *, user_id: int
    ) -> List[Tuple[int, Message, int]]:
        """
        获取用户的所有会话
        返回：(对话用户ID, 最新消息, 未读消息数)
        """
        # 获取用户参与的所有消息的用户ID（发送者或接收者）
        user_ids = (
            db.query(Message.sender_id)
            .filter(Message.receiver_id == user_id)
            .distinct()
            .all()
        )
        user_ids += (
            db.query(Message.receiver_id)
            .filter(Message.sender_id == user_id)
            .distinct()
            .all()
        )
        # 去重
        user_ids = list(set([uid[0] for uid in user_ids]))
        
        conversations = []
        for uid in user_ids:
            # 获取与该用户的最新一条消息
            latest_message = (
                db.query(self.model)
                .filter(
                    or_(
                        and_(
                            Message.sender_id == user_id,
                            Message.receiver_id == uid
                        ),
                        and_(
                            Message.sender_id == uid,
                            Message.receiver_id == user_id
                        )
                    )
                )
                .order_by(Message.created_at.desc())
                .first()
            )
            
            # 获取未读消息数
            unread_count = (
                db.query(self.model)
                .filter(
                    Message.sender_id == uid,
                    Message.receiver_id == user_id,
                    Message.is_read == False
                )
                .count()
            )
            
            if latest_message:
                conversations.append((uid, latest_message, unread_count))
        
        # 按最新消息时间排序
        conversations.sort(key=lambda x: x[1].created_at, reverse=True)
        return conversations
    
    def mark_as_read(
        self, db: Session, *, user_id1: int, user_id2: int
    ) -> int:
        """
        标记用户2发送给用户1的所有消息为已读
        返回修改的消息数
        """
        unread_messages = (
            db.query(self.model)
            .filter(
                Message.sender_id == user_id2,
                Message.receiver_id == user_id1,
                Message.is_read == False
            )
            .all()
        )
        
        for message in unread_messages:
            message.is_read = True
            db.add(message)
        
        db.commit()
        return len(unread_messages)

message = CRUDMessage(Message) 