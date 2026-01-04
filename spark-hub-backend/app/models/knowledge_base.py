from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Table, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db.base_class import Base
from app.models.tag import Tag, knowledge_base_tags, paper_tags

# 用户点赞论文关系表
user_likes_papers = Table(
    'user_likes_papers',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('paper_id', Integer, ForeignKey('papers.id'), primary_key=True)
)

# 知识库标签和论文标签关系表已移至tag.py

class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    description = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # 关系
    owner = relationship("User", back_populates="knowledge_bases")
    papers = relationship("Paper", back_populates="knowledge_base", cascade="all, delete-orphan")
    tags = relationship("Tag", secondary=knowledge_base_tags, back_populates="knowledge_bases")
    
    # 统计信息
    stars_count = Column(Integer, default=0)
    forks_count = Column(Integer, default=0)
    
    # 用户喜欢关系
    liked_by_users = relationship(
        "User",
        secondary="user_liked_knowledge_bases",
        back_populates="liked_knowledge_bases"
    ) 