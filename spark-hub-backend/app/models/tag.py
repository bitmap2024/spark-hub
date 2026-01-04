from sqlalchemy import Column, Integer, String, ForeignKey, Table
from sqlalchemy.orm import relationship

from app.db.base_class import Base

# 知识库标签关系表
knowledge_base_tags = Table(
    'knowledge_base_tags',
    Base.metadata,
    Column('knowledge_base_id', Integer, ForeignKey('knowledge_bases.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

# 论文标签关系表
paper_tags = Table(
    'paper_tags',
    Base.metadata,
    Column('paper_id', Integer, ForeignKey('papers.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)

class Tag(Base):
    __tablename__ = "tags"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    
    # 关系
    knowledge_bases = relationship("KnowledgeBase", secondary=knowledge_base_tags, back_populates="tags")
    papers = relationship("Paper", secondary=paper_tags, back_populates="tags") 