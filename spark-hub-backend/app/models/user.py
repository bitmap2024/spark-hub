from sqlalchemy import Boolean, Column, Integer, String, DateTime, Table, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base

# 创建用户关注关系表
user_following = Table(
    "user_following",
    Base.metadata,
    Column("follower_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("followed_id", Integer, ForeignKey("users.id"), primary_key=True),
)

# 用户喜欢的知识库
user_liked_knowledge_bases = Table(
    "user_liked_knowledge_bases",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("knowledge_base_id", Integer, ForeignKey("knowledge_bases.id"), primary_key=True),
)

# 不重复定义user_likes_papers表，因为已经在knowledge_base.py中定义
# 只使用引用

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    avatar = Column(String, nullable=True)
    location = Column(String, nullable=True)
    experience = Column(String, nullable=True)
    gender = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    school = Column(String, nullable=True)
    is_active = Column(Boolean(), default=True)
    is_superuser = Column(Boolean(), default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 用户创建的知识库
    knowledge_bases = relationship("KnowledgeBase", back_populates="owner")
    
    # 关注关系
    followers = relationship(
        "User",
        secondary=user_following,
        primaryjoin=id==user_following.c.followed_id,
        secondaryjoin=id==user_following.c.follower_id,
        backref="following"
    )
    
    # 用户喜欢的知识库
    liked_knowledge_bases = relationship(
        "KnowledgeBase",
        secondary=user_liked_knowledge_bases,
        back_populates="liked_by_users"
    )
    
    # 发送的消息
    sent_messages = relationship(
        "Message",
        foreign_keys="Message.sender_id",
        back_populates="sender"
    )
    
    # 接收的消息
    received_messages = relationship(
        "Message",
        foreign_keys="Message.receiver_id",
        back_populates="receiver"
    )
    
    # 点赞的论文
    liked_papers = relationship(
        "Paper", 
        secondary="user_likes_papers", 
        back_populates="liked_by_users"
    )
    
    # 用户发布的帖子
    posts = relationship("Post", back_populates="author")
    
    # 用户发表的评论
    comments = relationship("Comment", back_populates="author")
    
    # 用户点赞的帖子
    liked_posts = relationship(
        "Post",
        secondary="user_liked_posts",
        back_populates="liked_by_users"
    ) 