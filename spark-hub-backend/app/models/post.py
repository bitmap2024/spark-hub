from sqlalchemy import Boolean, Column, Integer, String, DateTime, Table, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base

# 用户喜欢的帖子关联表
user_liked_posts = Table(
    "user_liked_posts",
    Base.metadata,
    Column("user_id", Integer, ForeignKey("users.id"), primary_key=True),
    Column("post_id", Integer, ForeignKey("posts.id"), primary_key=True),
)

class Post(Base):
    __tablename__ = "posts"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    is_approved = Column(Boolean, default=False)
    is_hidden = Column(Boolean, default=False)
    likes = Column(Integer, default=0)
    views = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联到用户
    author_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", back_populates="posts")
    
    # 帖子的评论
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    
    # 喜欢这个帖子的用户
    liked_by_users = relationship(
        "User",
        secondary=user_liked_posts,
        back_populates="liked_posts"
    )


class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    is_approved = Column(Boolean, default=False)
    is_hidden = Column(Boolean, default=False)
    likes = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关联到用户
    author_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", back_populates="comments")
    
    # 关联到帖子
    post_id = Column(Integer, ForeignKey("posts.id"))
    post = relationship("Post", back_populates="comments") 