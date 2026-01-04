from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps
from app.crud import crud_knowledge_base, crud_paper, crud_tag
from app.schemas.knowledge_base import (
    KnowledgeBase, 
    KnowledgeBaseCreate, 
    KnowledgeBaseUpdate,
    PaperCreate
)
from app.schemas.user import User

router = APIRouter()

@router.get("/", response_model=List[schemas.KnowledgeBase])
def read_knowledge_bases(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取所有知识库列表
    """
    knowledge_bases = crud.knowledge_base.get_multi(db, skip=skip, limit=limit)
    return knowledge_bases

@router.post("/", response_model=schemas.KnowledgeBase)
def create_knowledge_base(
    *,
    db: Session = Depends(deps.get_db),
    knowledge_base_in: schemas.KnowledgeBaseCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    创建新知识库
    """
    knowledge_base = crud.knowledge_base.create_with_owner(
        db=db, obj_in=knowledge_base_in, owner_id=current_user.id
    )
    return knowledge_base

@router.get("/{kb_id}", response_model=schemas.KnowledgeBaseWithPapers)
def read_knowledge_base(
    *,
    db: Session = Depends(deps.get_db),
    kb_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取知识库详情（包含论文）
    """
    knowledge_base = crud.knowledge_base.get(db=db, id=kb_id)
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    return knowledge_base

@router.put("/{kb_id}", response_model=schemas.KnowledgeBase)
def update_knowledge_base(
    *,
    db: Session = Depends(deps.get_db),
    kb_id: int,
    knowledge_base_in: schemas.KnowledgeBaseUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    更新知识库
    """
    knowledge_base = crud.knowledge_base.get(db=db, id=kb_id)
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if knowledge_base.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    # 更新基本信息
    knowledge_base = crud.knowledge_base.update(
        db=db, db_obj=knowledge_base, obj_in=knowledge_base_in
    )
    
    # 更新标签（如果提供）
    if knowledge_base_in.tags is not None:
        knowledge_base = crud.knowledge_base.update_tags(
            db=db, db_obj=knowledge_base, tags=knowledge_base_in.tags
        )
    
    return knowledge_base

@router.delete("/{kb_id}", response_model=schemas.KnowledgeBase)
def delete_knowledge_base(
    *,
    db: Session = Depends(deps.get_db),
    kb_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    删除知识库
    """
    knowledge_base = crud.knowledge_base.get(db=db, id=kb_id)
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if knowledge_base.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    knowledge_base = crud.knowledge_base.remove(db=db, id=kb_id)
    return knowledge_base

@router.post("/{kb_id}/papers", response_model=schemas.Paper)
def add_paper_to_knowledge_base(
    *,
    db: Session = Depends(deps.get_db),
    kb_id: int,
    paper_in: schemas.PaperCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    添加论文到知识库
    """
    knowledge_base = crud.knowledge_base.get(db=db, id=kb_id)
    if not knowledge_base:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    if knowledge_base.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    paper = crud.paper.create_with_knowledge_base(
        db=db, obj_in=paper_in, knowledge_base_id=kb_id
    )
    return paper

@router.get("/user/{user_id}", response_model=List[schemas.KnowledgeBase])
def read_user_knowledge_bases(
    *,
    db: Session = Depends(deps.get_db),
    user_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取用户的知识库
    """
    knowledge_bases = crud.knowledge_base.get_multi_by_owner(
        db=db, owner_id=user_id, skip=skip, limit=limit
    )
    return knowledge_bases

@router.get("/username/{username}", response_model=List[schemas.KnowledgeBase])
def read_user_knowledge_bases_by_username(
    *,
    db: Session = Depends(deps.get_db),
    username: str,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    通过用户名获取用户的知识库
    """
    user = crud.user.get_by_username(db=db, username=username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    knowledge_bases = crud.knowledge_base.get_multi_by_owner(
        db=db, owner_id=user.id, skip=skip, limit=limit
    )
    return knowledge_bases

@router.post("/{kb_id}/like", response_model=bool)
def like_knowledge_base(
    *,
    db: Session = Depends(deps.get_db),
    kb_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    喜欢知识库
    """
    result = crud.knowledge_base.like(db=db, kb_id=kb_id, user_id=current_user.id)
    return result

@router.post("/{kb_id}/unlike", response_model=bool)
def unlike_knowledge_base(
    *,
    db: Session = Depends(deps.get_db),
    kb_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    取消喜欢知识库
    """
    result = crud.knowledge_base.unlike(db=db, kb_id=kb_id, user_id=current_user.id)
    return result 