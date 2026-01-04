# 从此处导入所有SQLAlchemy模型
# 用于确保它们在初始化数据库时被注册

from app.db.base_class import Base  # noqa
from app.models.user import User  # noqa
from app.models.knowledge_base import KnowledgeBase, Tag, user_likes_papers, knowledge_base_tags, paper_tags  # noqa
from app.models.paper import Paper  # noqa
from app.models.message import Message  # noqa 