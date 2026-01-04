import logging

from sqlalchemy.orm import Session

from app import crud, schemas
from app.core.config import settings
from app.db import base  # noqa: F401
from app.db.base_class import Base
from app.db.session import engine
from app.models import user, knowledge_base, message

# 确保导入所有SQLAlchemy模型
# 导入同一目录下的base模块，其中会导入所有模型

logger = logging.getLogger(__name__)

def init_db(db: Session) -> None:
    """
    初始化数据库，创建第一个超级用户
    """
    # 创建表
    Base.metadata.create_all(bind=engine)
    
    # 创建超级用户
    user = crud.user.get_by_email(db, email=settings.FIRST_SUPERUSER_EMAIL)
    if not user:
        user_in = schemas.UserCreate(
            email=settings.FIRST_SUPERUSER_EMAIL,
            password=settings.FIRST_SUPERUSER_PASSWORD,
            username=settings.FIRST_SUPERUSER_USERNAME,
            is_superuser=True,
        )
        user = crud.user.create(db, obj_in=user_in)
        logger.info(f"超级用户已创建: {user.email}")
    else:
        logger.info(f"超级用户已存在: {user.email}") 