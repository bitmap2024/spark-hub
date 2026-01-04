from typing import List, Optional, Tuple, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from app.models.post import Post, Comment
from app.schemas.post import PostCreate, PostUpdate, CommentCreate, CommentUpdate
from app.crud.base import CRUDBase

class CRUDPost(CRUDBase[Post, PostCreate, PostUpdate]):
    def get_multi_with_pagination(
        self, 
        db: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Post], int]:
        """
        获取带有评论计数的帖子列表
        """
        query = db.query(
            Post, 
            func.count(Comment.id).label("comments_count")
        ).outerjoin(
            Comment, 
            Post.id == Comment.post_id
        ).group_by(
            Post.id
        )
        
        # 应用过滤条件
        if filters:
            for key, value in filters.items():
                if hasattr(Post, key) and value is not None:
                    query = query.filter(getattr(Post, key) == value)
        
        total = query.count()
        
        # 排序和分页
        posts_with_count = query.order_by(desc(Post.created_at)).offset(skip).limit(limit).all()
        
        # 处理结果，将评论计数添加到帖子对象中
        posts = []
        for post, comments_count in posts_with_count:
            setattr(post, "comments_count", comments_count)
            posts.append(post)
            
        return posts, total
    
    def get_with_comments(self, db: Session, *, post_id: int) -> Optional[Post]:
        """
        获取带有评论的帖子详情
        """
        return db.query(Post).filter(Post.id == post_id).first()
    
    def increment_views(self, db: Session, *, post_id: int) -> Optional[Post]:
        """
        增加帖子浏览量
        """
        post = db.query(Post).filter(Post.id == post_id).first()
        if post:
            post.views += 1
            db.commit()
            db.refresh(post)
        return post
    
    def toggle_approve(self, db: Session, *, post_id: int) -> Optional[Post]:
        """
        切换帖子审核状态
        """
        post = db.query(Post).filter(Post.id == post_id).first()
        if post:
            post.is_approved = not post.is_approved
            db.commit()
            db.refresh(post)
        return post
    
    def toggle_hide(self, db: Session, *, post_id: int) -> Optional[Post]:
        """
        切换帖子隐藏状态
        """
        post = db.query(Post).filter(Post.id == post_id).first()
        if post:
            post.is_hidden = not post.is_hidden
            db.commit()
            db.refresh(post)
        return post
    
    def like_post(self, db: Session, *, post_id: int, user_id: int) -> Tuple[Post, bool]:
        """
        用户点赞/取消点赞帖子
        返回帖子对象和是否是新点赞
        """
        post = db.query(Post).filter(Post.id == post_id).first()
        if not post:
            return None, False
            
        from app.models.user import User
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return None, False
            
        # 检查是否已经点赞
        is_liked = user in post.liked_by_users
        
        if is_liked:
            # 取消点赞
            post.liked_by_users.remove(user)
            post.likes = max(0, post.likes - 1)  # 确保不会变成负数
            result = False
        else:
            # 添加点赞
            post.liked_by_users.append(user)
            post.likes += 1
            result = True
            
        db.commit()
        db.refresh(post)
        return post, result


class CRUDComment(CRUDBase[Comment, CommentCreate, CommentUpdate]):
    def get_by_post(
        self, 
        db: Session, 
        *, 
        post_id: int,
        skip: int = 0,
        limit: int = 100
    ) -> List[Comment]:
        """
        获取帖子的所有评论
        """
        return db.query(Comment).filter(
            Comment.post_id == post_id
        ).order_by(
            desc(Comment.created_at)
        ).offset(skip).limit(limit).all()
    
    def toggle_approve(self, db: Session, *, comment_id: int) -> Optional[Comment]:
        """
        切换评论审核状态
        """
        comment = db.query(Comment).filter(Comment.id == comment_id).first()
        if comment:
            comment.is_approved = not comment.is_approved
            db.commit()
            db.refresh(comment)
        return comment
    
    def toggle_hide(self, db: Session, *, comment_id: int) -> Optional[Comment]:
        """
        切换评论隐藏状态
        """
        comment = db.query(Comment).filter(Comment.id == comment_id).first()
        if comment:
            comment.is_hidden = not comment.is_hidden
            db.commit()
            db.refresh(comment)
        return comment
    
    def like_comment(self, db: Session, *, comment_id: int) -> Optional[Comment]:
        """
        给评论点赞
        """
        comment = db.query(Comment).filter(Comment.id == comment_id).first()
        if comment:
            comment.likes += 1
            db.commit()
            db.refresh(comment)
        return comment


post = CRUDPost(Post)
comment = CRUDComment(Comment) 