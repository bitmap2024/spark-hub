from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()

@router.get("/{paper_id}", response_model=schemas.Paper)
def read_paper(
    *,
    db: Session = Depends(deps.get_db),
    paper_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取论文详情
    """
    paper = crud.paper.get(db=db, id=paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return paper

@router.put("/{paper_id}", response_model=schemas.Paper)
def update_paper(
    *,
    db: Session = Depends(deps.get_db),
    paper_id: int,
    paper_in: schemas.PaperUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    更新论文
    """
    paper = crud.paper.get(db=db, id=paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # 检查是否有权限（知识库所有者）
    kb = crud.knowledge_base.get(db=db, id=paper.knowledge_base_id)
    if not kb or kb.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    paper = crud.paper.update(db=db, db_obj=paper, obj_in=paper_in)
    return paper

@router.delete("/{paper_id}", response_model=schemas.Paper)
def delete_paper(
    *,
    db: Session = Depends(deps.get_db),
    paper_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    删除论文
    """
    paper = crud.paper.get(db=db, id=paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    
    # 检查是否有权限（知识库所有者）
    kb = crud.knowledge_base.get(db=db, id=paper.knowledge_base_id)
    if not kb or kb.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    
    paper = crud.paper.remove(db=db, id=paper_id)
    return paper

@router.post("/{paper_id}/like", response_model=bool)
def like_paper(
    *,
    db: Session = Depends(deps.get_db),
    paper_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    点赞论文
    """
    result = crud.paper.like(db=db, paper_id=paper_id, user_id=current_user.id)
    return result

@router.post("/{paper_id}/unlike", response_model=bool)
def unlike_paper(
    *,
    db: Session = Depends(deps.get_db),
    paper_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    取消点赞论文
    """
    result = crud.paper.unlike(db=db, paper_id=paper_id, user_id=current_user.id)
    return result

@router.get("/liked", response_model=List[schemas.Paper])
def get_liked_papers(
    *,
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取当前用户点赞的论文
    """
    papers = crud.paper.get_liked_by_user(
        db=db, user_id=current_user.id, skip=skip, limit=limit
    )
    return papers

@router.get("/search", response_model=List[schemas.Paper])
def search_papers(
    *,
    db: Session = Depends(deps.get_db),
    q: str,
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    搜索论文
    """
    papers = crud.paper.search(db=db, query=q, skip=skip, limit=limit)
    return papers 