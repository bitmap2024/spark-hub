from fastapi import APIRouter

from app.api.api_v1.endpoints import auth, users, knowledge_bases, papers, tags, messages, posts

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(knowledge_bases.router, prefix="/knowledge_bases", tags=["knowledge_bases"])
api_router.include_router(papers.router, prefix="/papers", tags=["papers"])
api_router.include_router(tags.router, prefix="/tags", tags=["tags"])
api_router.include_router(messages.router, prefix="/messages", tags=["messages"])
api_router.include_router(posts.router, prefix="/posts", tags=["posts"]) 