from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Text, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.tag import paper_tags

# 这里不需要重复定义user_liked_papers，只需要在关系中引用

class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    authors = Column(String)  # 存储为逗号分隔的字符串
    abstract = Column(Text)
    publish_date = Column(String)
    doi = Column(String, nullable=True)
    url = Column(String, nullable=True)
    knowledge_base_id = Column(Integer, ForeignKey("knowledge_bases.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关系
    knowledge_base = relationship("KnowledgeBase", back_populates="papers")
    
    # 标签关系
    tags = relationship(
        "Tag",
        secondary="paper_tags",
        back_populates="papers"
    )
    
    # 点赞用户
    liked_by_users = relationship(
        "User",
        secondary="user_likes_papers",
        back_populates="liked_papers"
    ) 