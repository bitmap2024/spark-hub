from typing import List, Optional, Dict, Any, Union
from sqlalchemy.orm import Session

from app.crud.base import CRUDBase
from app.models.paper import Paper
from app.models.user import User
from app.schemas.paper import PaperCreate, PaperUpdate


class CRUDPaper(CRUDBase[Paper, PaperCreate, PaperUpdate]):
    def get_multi_by_knowledge_base(
        self, db: Session, *, knowledge_base_id: int, skip: int = 0, limit: int = 100
    ) -> List[Paper]:
        """
        获取知识库的所有论文
        """
        return (
            db.query(self.model)
            .filter(Paper.knowledge_base_id == knowledge_base_id)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def create_with_knowledge_base(
        self, db: Session, *, obj_in: PaperCreate, knowledge_base_id: int
    ) -> Paper:
        """
        创建论文，指定知识库
        """
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
    
    def search(
        self, db: Session, *, query: str, skip: int = 0, limit: int = 100
    ) -> List[Paper]:
        """
        搜索论文
        """
        return (
            db.query(self.model)
            .filter(
                Paper.title.ilike(f"%{query}%") | 
                Paper.abstract.ilike(f"%{query}%")
            )
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def like(self, db: Session, *, paper_id: int, user_id: int) -> bool:
        """
        用户点赞论文
        """
        paper = self.get(db, id=paper_id)
        user = db.query(User).filter(User.id == user_id).first()
        
        if not paper or not user or paper in user.liked_papers:
            return False
        
        user.liked_papers.append(paper)
        db.commit()
        return True
    
    def unlike(self, db: Session, *, paper_id: int, user_id: int) -> bool:
        """
        用户取消点赞论文
        """
        paper = self.get(db, id=paper_id)
        user = db.query(User).filter(User.id == user_id).first()
        
        if not paper or not user or paper not in user.liked_papers:
            return False
        
        user.liked_papers.remove(paper)
        db.commit()
        return True
    
    def get_liked_by_user(
        self, db: Session, *, user_id: int, skip: int = 0, limit: int = 100
    ) -> List[Paper]:
        """
        获取用户点赞的论文
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return []
        
        return user.liked_papers[skip:skip+limit]


paper = CRUDPaper(Paper) 