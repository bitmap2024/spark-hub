from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api import deps
from app.crud import crud_user
from app.schemas.user import User, UserUpdate

router = APIRouter()

@router.get("/", response_model=List[User])
def read_users(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取用户列表
    """
    users = crud_user.user.get_multi(db, skip=skip, limit=limit)
    return users

@router.get("/{user_id}", response_model=User)
def read_user(
    user_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取用户信息
    """
    user = crud_user.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    
    # 添加关注状态
    user.is_following = crud_user.user.is_following(db, follower=current_user, followed=user)
    user.followers_count = crud_user.user.get_followers_count(db, user=user)
    user.following_count = crud_user.user.get_following_count(db, user=user)
    
    return user

@router.get("/username/{username}", response_model=User)
def read_user_by_username(
    username: str,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    通过用户名获取用户信息
    """
    user = crud_user.user.get_by_username(db, username=username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    
    # 添加关注状态
    user.is_following = crud_user.user.is_following(db, follower=current_user, followed=user)
    user.followers_count = crud_user.user.get_followers_count(db, user=user)
    user.following_count = crud_user.user.get_following_count(db, user=user)
    
    return user

@router.put("/me", response_model=User)
def update_user_me(
    *,
    db: Session = Depends(deps.get_db),
    user_in: UserUpdate,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    更新当前用户信息
    """
    user = crud_user.user.update(db, db_obj=current_user, obj_in=user_in)
    return user

@router.post("/follow/{user_id}", response_model=User)
def follow_user(
    user_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    关注用户
    """
    user = crud_user.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能关注自己",
        )
    
    if crud_user.user.is_following(db, follower=current_user, followed=user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="已经关注该用户",
        )
    
    crud_user.user.follow(db, follower=current_user, followed=user)
    
    # 更新用户信息
    user.is_following = True
    user.followers_count = crud_user.user.get_followers_count(db, user=user)
    user.following_count = crud_user.user.get_following_count(db, user=user)
    
    return user

@router.post("/unfollow/{user_id}", response_model=User)
def unfollow_user(
    user_id: int,
    db: Session = Depends(deps.get_db),
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    取消关注用户
    """
    user = crud_user.user.get(db, id=user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在",
        )
    
    if not crud_user.user.is_following(db, follower=current_user, followed=user):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="未关注该用户",
        )
    
    crud_user.user.unfollow(db, follower=current_user, followed=user)
    
    # 更新用户信息
    user.is_following = False
    user.followers_count = crud_user.user.get_followers_count(db, user=user)
    user.following_count = crud_user.user.get_following_count(db, user=user)
    
    return user

@router.get("/me/following", response_model=List[User])
def read_following(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取当前用户关注的用户列表
    """
    users = current_user.following[skip:skip+limit]
    for user in users:
        user.is_following = True
        user.followers_count = crud_user.user.get_followers_count(db, user=user)
        user.following_count = crud_user.user.get_following_count(db, user=user)
    return users

@router.get("/me/followers", response_model=List[User])
def read_followers(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取当前用户的粉丝列表
    """
    users = current_user.followers[skip:skip+limit]
    for user in users:
        user.is_following = crud_user.user.is_following(db, follower=current_user, followed=user)
        user.followers_count = crud_user.user.get_followers_count(db, user=user)
        user.following_count = crud_user.user.get_following_count(db, user=user)
    return users 