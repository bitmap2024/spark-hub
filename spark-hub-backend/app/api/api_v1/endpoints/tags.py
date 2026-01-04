from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api import deps
from app.crud import crud_tag
from app.schemas.knowledge_base import Tag, TagCreate
from app.schemas.user import User

router = APIRouter()

@router.get("/", response_model=List[Tag])
def read_tags(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取所有标签
    """
    tags = crud_tag.tag.get_multi(db, skip=skip, limit=limit)
    return tags

@router.post("/", response_model=Tag)
def create_tag(
    *,
    db: Session = Depends(deps.get_db),
    tag_in: TagCreate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    创建新标签
    """
    # 检查标签是否已存在
    tag = crud_tag.tag.get_by_name(db, name=tag_in.name)
    if tag:
        return tag
    
    # 创建新标签
    tag = crud_tag.tag.create(db=db, obj_in=tag_in)
    return tag

@router.get("/{tag_id}", response_model=Tag)
def read_tag(
    *,
    db: Session = Depends(deps.get_db),
    tag_id: int,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取标签详情
    """
    tag = crud_tag.tag.get(db, id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="标签不存在",
        )
    return tag

@router.get("/{tag_id}/knowledge-bases", response_model=List[dict])
def read_tag_knowledge_bases(
    *,
    db: Session = Depends(deps.get_db),
    tag_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取标签下的知识库
    """
    tag = crud_tag.tag.get(db, id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="标签不存在",
        )
    
    knowledge_bases = crud_tag.tag.get_knowledge_bases(db, tag_id=tag_id, skip=skip, limit=limit)
    return knowledge_bases

@router.get("/{tag_id}/papers", response_model=List[dict])
def read_tag_papers(
    *,
    db: Session = Depends(deps.get_db),
    tag_id: int,
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取标签下的论文
    """
    tag = crud_tag.tag.get(db, id=tag_id)
    if not tag:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="标签不存在",
        )
    
    papers = crud_tag.tag.get_papers(db, tag_id=tag_id, skip=skip, limit=limit)
    return papers 