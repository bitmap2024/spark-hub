from typing import List, Optional
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.knowledge_base import Tag, KnowledgeBase
from app.schemas.tag import TagCreate, TagUpdate


class CRUDTag(CRUDBase[Tag, TagCreate, TagUpdate]):
    def get_by_name(self, db: Session, *, name: str) -> Optional[Tag]:
        """
        通过名称获取标签
        """
        return db.query(self.model).filter(Tag.name == name).first()
    
    def get_or_create(self, db: Session, *, name: str) -> Tag:
        """
        获取标签，不存在则创建
        """
        tag = self.get_by_name(db, name=name)
        if not tag:
            tag_in = TagCreate(name=name)
            tag = self.create(db, obj_in=tag_in)
        return tag
    
    def get_multi_by_knowledge_base(
        self, db: Session, *, knowledge_base_id: int, skip: int = 0, limit: int = 100
    ) -> List[Tag]:
        """
        获取知识库的所有标签
        """
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if not kb:
            return []
        
        return kb.tags[skip:skip+limit]
    
    def get_popular(self, db: Session, *, limit: int = 10) -> List[Tag]:
        """
        获取最常用的标签
        """
        return (
            db.query(self.model)
            .join(self.model.knowledge_bases)
            .group_by(self.model.id)
            .order_by(db.func.count(self.model.knowledge_bases).desc())
            .limit(limit)
            .all()
        )

    def create_with_knowledge_base(
        self, db: Session, *, obj_in: TagCreate, knowledge_base_id: int
    ) -> Tag:
        tag = db.query(Tag).filter(Tag.name == obj_in.name).first()
        if not tag:
            tag = Tag(name=obj_in.name)
            db.add(tag)
            db.commit()
            db.refresh(tag)
        
        # 添加到知识库
        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).first()
        if kb and tag not in kb.tags:
            kb.tags.append(tag)
            db.commit()
            
        return tag


tag = CRUDTag(Tag) 