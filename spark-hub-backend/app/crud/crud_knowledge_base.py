from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy import func

from fastapi.encoders import jsonable_encoder

from app.crud.base import CRUDBase
from app.models.knowledge_base import KnowledgeBase, Tag
from app.models.paper import Paper
from app.schemas.knowledge_base import KnowledgeBaseCreate, KnowledgeBaseUpdate, PaperCreate, TagCreate

class CRUDKnowledgeBase(CRUDBase[KnowledgeBase, KnowledgeBaseCreate, KnowledgeBaseUpdate]):
    def get_by_user_id(self, db: Session, *, user_id: int, skip: int = 0, limit: int = 100) -> List[KnowledgeBase]:
        return db.query(KnowledgeBase).filter(KnowledgeBase.user_id == user_id).offset(skip).limit(limit).all()
    
    def get_by_username(self, db: Session, *, username: str, skip: int = 0, limit: int = 100) -> List[KnowledgeBase]:
        from app.models.user import User
        user = db.query(User).filter(User.username == username).first()
        if not user:
            return []
        return self.get_by_user_id(db, user_id=user.id, skip=skip, limit=limit)
    
    def create_with_owner(
        self, db: Session, *, obj_in: KnowledgeBaseCreate, owner_id: int
    ) -> KnowledgeBase:
        """
        创建知识库，指定所有者
        """
        db_obj = KnowledgeBase(
            title=obj_in.title,
            description=obj_in.description,
            user_id=owner_id,
        )
        
        # 处理标签
        if obj_in.tags:
            for tag_name in obj_in.tags:
                # 检查标签是否存在，不存在则创建
                tag = db.query(Tag).filter(Tag.name == tag_name).first()
                if not tag:
                    tag = Tag(name=tag_name)
                    db.add(tag)
                    db.flush()
                db_obj.tags.append(tag)
                
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get_multi_by_owner(
        self, db: Session, *, owner_id: int, skip: int = 0, limit: int = 100
    ) -> List[KnowledgeBase]:
        """
        获取用户的所有知识库
        """
        return (
            db.query(self.model)
            .filter(KnowledgeBase.user_id == owner_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_title(self, db: Session, *, title: str) -> Optional[KnowledgeBase]:
        """
        通过标题获取知识库
        """
        return db.query(self.model).filter(KnowledgeBase.title == title).first()
    
    def update_tags(
        self, db: Session, *, db_obj: KnowledgeBase, tags: List[str]
    ) -> KnowledgeBase:
        """
        更新知识库标签
        """
        # 清除现有标签
        db_obj.tags = []
        
        # 添加新标签
        for tag_name in tags:
            tag = db.query(Tag).filter(Tag.name == tag_name).first()
            if not tag:
                tag = Tag(name=tag_name)
                db.add(tag)
                db.flush()
            db_obj.tags.append(tag)
            
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def get_by_tag(
        self, db: Session, *, tag_name: str, skip: int = 0, limit: int = 100
    ) -> List[KnowledgeBase]:
        """
        通过标签获取知识库
        """
        return (
            db.query(self.model)
            .join(self.model.tags)
            .filter(Tag.name == tag_name)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def like(self, db: Session, *, kb_id: int, user_id: int) -> bool:
        """
        用户点赞知识库
        """
        from app.models.user import User
        
        kb = self.get(db, id=kb_id)
        user = db.query(User).filter(User.id == user_id).first()
        
        if not kb or not user or kb in user.liked_knowledge_bases:
            return False
        
        user.liked_knowledge_bases.append(kb)
        kb.stars_count += 1
        db.commit()
        return True
    
    def unlike(self, db: Session, *, kb_id: int, user_id: int) -> bool:
        """
        用户取消点赞知识库
        """
        from app.models.user import User
        
        kb = self.get(db, id=kb_id)
        user = db.query(User).filter(User.id == user_id).first()
        
        if not kb or not user or kb not in user.liked_knowledge_bases:
            return False
        
        user.liked_knowledge_bases.remove(kb)
        kb.stars_count -= 1
        db.commit()
        return True
    
    def get_liked_by_user(
        self, db: Session, *, user_id: int, skip: int = 0, limit: int = 100
    ) -> List[KnowledgeBase]:
        """
        获取用户点赞的知识库
        """
        from app.models.user import User
        
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return []
        
        return user.liked_knowledge_bases[skip:skip+limit]

class CRUDPaper(CRUDBase[Paper, PaperCreate, PaperCreate]):
    def get_by_knowledge_base_id(
        self, db: Session, *, knowledge_base_id: int, skip: int = 0, limit: int = 100
    ) -> List[Paper]:
        return db.query(Paper).filter(Paper.knowledge_base_id == knowledge_base_id).offset(skip).limit(limit).all()
    
    def create_with_knowledge_base(
        self, db: Session, *, obj_in: PaperCreate, knowledge_base_id: int
    ) -> Paper:
        db_obj = Paper(
            title=obj_in.title,
            authors=obj_in.authors,
            abstract=obj_in.abstract,
            publish_date=obj_in.publish_date,
            doi=obj_in.doi,
            url=obj_in.url,
            knowledge_base_id=knowledge_base_id,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def like(self, db: Session, *, db_obj: Paper, user_id: int) -> Paper:
        from app.models.user import User
        user = db.query(User).filter(User.id == user_id).first()
        if user and db_obj not in user.liked_papers:
            user.liked_papers.append(db_obj)
            db.commit()
            db.refresh(db_obj)
        return db_obj
    
    def unlike(self, db: Session, *, db_obj: Paper, user_id: int) -> Paper:
        from app.models.user import User
        user = db.query(User).filter(User.id == user_id).first()
        if user and db_obj in user.liked_papers:
            user.liked_papers.remove(db_obj)
            db.commit()
            db.refresh(db_obj)
        return db_obj

class CRUDTag(CRUDBase[Tag, TagCreate, TagCreate]):
    def get_by_name(self, db: Session, *, name: str) -> Optional[Tag]:
        return db.query(Tag).filter(Tag.name == name).first()
    
    def get_or_create(self, db: Session, *, name: str) -> Tag:
        tag = self.get_by_name(db, name=name)
        if not tag:
            tag = Tag(name=name)
            db.add(tag)
            db.commit()
            db.refresh(tag)
        return tag

knowledge_base = CRUDKnowledgeBase(KnowledgeBase)
paper = CRUDPaper(Paper)
tag = CRUDTag(Tag) 