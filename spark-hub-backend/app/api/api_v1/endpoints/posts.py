from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()


@router.get("/", response_model=schemas.post.PostPage)
def get_posts(
    db: Session = Depends(deps.get_db),
    current_user: Optional[models.User] = Depends(deps.get_current_user_optional),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    approved_only: bool = Query(True),
) -> Any:
    """
    获取帖子列表
    """
    filters = {}
    
    # 非管理员只能看已审核且未隐藏的帖子
    if not (current_user and current_user.is_superuser):
        filters["is_approved"] = True
        filters["is_hidden"] = False
    # 管理员可以通过approved_only参数过滤
    elif current_user.is_superuser and approved_only:
        filters["is_approved"] = True
    
    posts, total = crud.post.get_multi_with_pagination(
        db, skip=skip, limit=limit, filters=filters
    )
    
    # 计算总页数
    total_pages = (total + limit - 1) // limit if total > 0 else 1
    
    return {
        "items": posts,
        "total": total,
        "page": (skip // limit) + 1,
        "per_page": limit,
        "total_pages": total_pages
    }


@router.get("/pending", response_model=schemas.post.PostPage)
def get_pending_posts(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_superuser),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
) -> Any:
    """
    获取待审核的帖子列表（仅管理员）
    """
    filters = {
        "is_approved": False,
        "is_hidden": False
    }
    
    posts, total = crud.post.get_multi_with_pagination(
        db, skip=skip, limit=limit, filters=filters
    )
    
    # 计算总页数
    total_pages = (total + limit - 1) // limit if total > 0 else 1
    
    return {
        "items": posts,
        "total": total,
        "page": (skip // limit) + 1,
        "per_page": limit,
        "total_pages": total_pages
    }


@router.get("/hidden", response_model=schemas.post.PostPage)
def get_hidden_posts(
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_superuser),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
) -> Any:
    """
    获取被隐藏的帖子列表（仅管理员）
    """
    filters = {
        "is_hidden": True
    }
    
    posts, total = crud.post.get_multi_with_pagination(
        db, skip=skip, limit=limit, filters=filters
    )
    
    # 计算总页数
    total_pages = (total + limit - 1) // limit if total > 0 else 1
    
    return {
        "items": posts,
        "total": total,
        "page": (skip // limit) + 1,
        "per_page": limit,
        "total_pages": total_pages
    }


@router.get("/{post_id}", response_model=schemas.post.Post)
def get_post(
    *,
    db: Session = Depends(deps.get_db),
    post_id: int,
    current_user: Optional[models.User] = Depends(deps.get_current_user_optional),
) -> Any:
    """
    获取帖子详情
    """
    post = crud.post.get_with_comments(db, post_id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    
    # 非管理员只能看已审核且未隐藏的帖子
    if not (current_user and current_user.is_superuser):
        if not post.is_approved or post.is_hidden:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="帖子不存在或已被隐藏"
            )
    
    # 增加浏览量
    crud.post.increment_views(db, post_id=post_id)
    
    return post


@router.post("/", response_model=schemas.post.Post)
def create_post(
    *,
    db: Session = Depends(deps.get_db),
    post_in: schemas.post.PostCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    创建新帖子
    """
    # 自动审核通过管理员发布的帖子
    is_approved = current_user.is_superuser
    
    post = crud.post.create(
        db,
        obj_in={
            **post_in.model_dump(),
            "author_id": current_user.id,
            "is_approved": is_approved
        }
    )
    return post


@router.put("/{post_id}", response_model=schemas.post.Post)
def update_post(
    *,
    db: Session = Depends(deps.get_db),
    post_id: int,
    post_in: schemas.post.PostUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    更新帖子
    """
    post = crud.post.get(db, id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    
    # 只有作者或管理员可以更新帖子
    if post.author_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限更新此帖子"
        )
    
    # 如果是普通用户更新，需要重新审核
    if not current_user.is_superuser and (post_in.title or post_in.content):
        post_in = schemas.post.PostUpdate(**post_in.model_dump(), is_approved=False)
    
    post = crud.post.update(db, db_obj=post, obj_in=post_in)
    return post


@router.delete("/{post_id}", response_model=schemas.post.Post)
def delete_post(
    *,
    db: Session = Depends(deps.get_db),
    post_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    删除帖子
    """
    post = crud.post.get(db, id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    
    # 只有作者或管理员可以删除帖子
    if post.author_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此帖子"
        )
    
    post = crud.post.remove(db, id=post_id)
    return post


@router.post("/{post_id}/approve", response_model=schemas.post.Post)
def approve_post(
    *,
    db: Session = Depends(deps.get_db),
    post_id: int,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    审核通过/取消通过帖子（仅管理员）
    """
    post = crud.post.toggle_approve(db, post_id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    return post


@router.post("/{post_id}/hide", response_model=schemas.post.Post)
def hide_post(
    *,
    db: Session = Depends(deps.get_db),
    post_id: int,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    隐藏/显示帖子（仅管理员）
    """
    post = crud.post.toggle_hide(db, post_id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    return post


@router.post("/{post_id}/like", response_model=schemas.post.Post)
def like_post(
    *,
    db: Session = Depends(deps.get_db),
    post_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    点赞/取消点赞帖子
    """
    post, _ = crud.post.like_post(db, post_id=post_id, user_id=current_user.id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    return post


# 评论相关API
@router.post("/{post_id}/comments", response_model=schemas.post.Comment)
def create_comment(
    *,
    db: Session = Depends(deps.get_db),
    post_id: int,
    comment_in: schemas.post.CommentCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    创建评论
    """
    post = crud.post.get(db, id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    
    # 非管理员用户只能在已审核且未隐藏的帖子下评论
    if not current_user.is_superuser and (not post.is_approved or post.is_hidden):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="该帖子不允许评论"
        )
    
    # 自动审核通过管理员发布的评论
    is_approved = current_user.is_superuser
    
    comment = crud.comment.create(
        db,
        obj_in={
            **comment_in.model_dump(),
            "author_id": current_user.id,
            "post_id": post_id,
            "is_approved": is_approved
        }
    )
    return comment


@router.get("/{post_id}/comments", response_model=List[schemas.post.Comment])
def get_comments(
    *,
    db: Session = Depends(deps.get_db),
    post_id: int,
    current_user: Optional[models.User] = Depends(deps.get_current_user_optional),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
) -> Any:
    """
    获取帖子的评论
    """
    post = crud.post.get(db, id=post_id)
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="帖子不存在"
        )
    
    # 非管理员只能看已审核且未隐藏的帖子的评论
    if not (current_user and current_user.is_superuser):
        if not post.is_approved or post.is_hidden:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="帖子不存在或已被隐藏"
            )
    
    comments = crud.comment.get_by_post(db, post_id=post_id, skip=skip, limit=limit)
    
    # 非管理员只能看到已审核且未隐藏的评论
    if not (current_user and current_user.is_superuser):
        comments = [c for c in comments if c.is_approved and not c.is_hidden]
    
    return comments


@router.put("/comments/{comment_id}", response_model=schemas.post.Comment)
def update_comment(
    *,
    db: Session = Depends(deps.get_db),
    comment_id: int,
    comment_in: schemas.post.CommentUpdate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    更新评论
    """
    comment = crud.comment.get(db, id=comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="评论不存在"
        )
    
    # 只有作者或管理员可以更新评论
    if comment.author_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限更新此评论"
        )
    
    # 如果是普通用户更新，需要重新审核
    if not current_user.is_superuser and comment_in.content:
        comment_in = schemas.post.CommentUpdate(**comment_in.model_dump(), is_approved=False)
    
    comment = crud.comment.update(db, db_obj=comment, obj_in=comment_in)
    return comment


@router.delete("/comments/{comment_id}", response_model=schemas.post.Comment)
def delete_comment(
    *,
    db: Session = Depends(deps.get_db),
    comment_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    删除评论
    """
    comment = crud.comment.get(db, id=comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="评论不存在"
        )
    
    # 只有作者或管理员可以删除评论
    if comment.author_id != current_user.id and not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有权限删除此评论"
        )
    
    comment = crud.comment.remove(db, id=comment_id)
    return comment


@router.post("/comments/{comment_id}/approve", response_model=schemas.post.Comment)
def approve_comment(
    *,
    db: Session = Depends(deps.get_db),
    comment_id: int,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    审核通过/取消通过评论（仅管理员）
    """
    comment = crud.comment.toggle_approve(db, comment_id=comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="评论不存在"
        )
    return comment


@router.post("/comments/{comment_id}/hide", response_model=schemas.post.Comment)
def hide_comment(
    *,
    db: Session = Depends(deps.get_db),
    comment_id: int,
    current_user: models.User = Depends(deps.get_current_active_superuser),
) -> Any:
    """
    隐藏/显示评论（仅管理员）
    """
    comment = crud.comment.toggle_hide(db, comment_id=comment_id)
    if not comment:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="评论不存在"
        )
    return comment 