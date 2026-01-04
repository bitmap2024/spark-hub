from typing import Any, Dict, Optional, Union, List
from sqlalchemy.orm import Session

from app.core.security import get_password_hash, verify_password
from app.crud.base import CRUDBase
from app.models.user import User
from app.schemas.user import UserCreate, UserUpdate

class CRUDUser(CRUDBase[User, UserCreate, UserUpdate]):
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        """
        通过邮箱获取用户
        """
        return db.query(User).filter(User.email == email).first()
    
    def get_by_username(self, db: Session, *, username: str) -> Optional[User]:
        """
        通过用户名获取用户
        """
        return db.query(User).filter(User.username == username).first()
    
    def create(self, db: Session, *, obj_in: UserCreate) -> User:
        """
        创建用户
        """
        db_obj = User(
            email=obj_in.email,
            username=obj_in.username,
            hashed_password=get_password_hash(obj_in.password),
            avatar=obj_in.avatar,
            is_active=obj_in.is_active,
            is_superuser=obj_in.is_superuser,
            location=obj_in.location,
            experience=obj_in.experience,
            gender=obj_in.gender,
            age=obj_in.age,
            school=obj_in.school,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj
    
    def update(
        self, db: Session, *, db_obj: User, obj_in: Union[UserUpdate, Dict[str, Any]]
    ) -> User:
        """
        更新用户
        """
        if isinstance(obj_in, dict):
            update_data = obj_in
        else:
            update_data = obj_in.dict(exclude_unset=True)
        if update_data.get("password"):
            hashed_password = get_password_hash(update_data["password"])
            del update_data["password"]
            update_data["hashed_password"] = hashed_password
        return super().update(db, db_obj=db_obj, obj_in=update_data)
    
    def authenticate(self, db: Session, *, email: str, password: str) -> Optional[User]:
        """
        验证用户
        """
        user = self.get_by_email(db, email=email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user
    
    def is_active(self, user: User) -> bool:
        """
        检查用户是否激活
        """
        return user.is_active
    
    def is_superuser(self, user: User) -> bool:
        """
        检查用户是否是超级用户
        """
        return user.is_superuser
    
    def follow(self, db: Session, *, user_id: int, target_id: int) -> bool:
        """
        关注用户
        """
        user = self.get(db, id=user_id)
        target = self.get(db, id=target_id)
        
        if not user or not target or target in user.following:
            return False
        
        user.following.append(target)
        db.commit()
        return True
    
    def unfollow(self, db: Session, *, user_id: int, target_id: int) -> bool:
        """
        取消关注用户
        """
        user = self.get(db, id=user_id)
        target = self.get(db, id=target_id)
        
        if not user or not target or target not in user.following:
            return False
        
        user.following.remove(target)
        db.commit()
        return True
    
    def is_following(self, db: Session, *, user_id: int, target_id: int) -> bool:
        """
        检查是否关注了用户
        """
        user = self.get(db, id=user_id)
        target = self.get(db, id=target_id)
        
        if not user or not target:
            return False
        
        return target in user.following
    
    def get_following(self, db: Session, *, user_id: int) -> List[User]:
        """
        获取用户关注的用户列表
        """
        user = self.get(db, id=user_id)
        if not user:
            return []
        
        return user.following
    
    def get_followers(self, db: Session, *, user_id: int) -> List[User]:
        """
        获取用户的粉丝列表
        """
        user = self.get(db, id=user_id)
        if not user:
            return []
        
        return user.followers

user = CRUDUser(User) 