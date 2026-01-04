import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import json
from loguru import logger
import aioredis
from config.settings import REDIS_URI

class Task:
    """爬虫任务类"""
    def __init__(
        self,
        url: str,
        priority: int = 0,
        retry_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.url = url
        self.priority = priority
        self.retry_count = retry_count
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'priority': self.priority,
            'retry_count': self.retry_count,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat()
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)

class RedisScheduler:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        queue_key: str = "spark:queue",
        processing_key: str = "spark:processing",
        failed_key: str = "spark:failed",
        done_key: str = "spark:done",
        processing_timeout: int = 300,  # 5 minutes
    ):
        self.redis_url = redis_url
        self.queue_key = queue_key
        self.processing_key = processing_key
        self.failed_key = failed_key
        self.done_key = done_key
        self.processing_timeout = processing_timeout
        self.redis: Optional[aioredis.Redis] = None

    async def __aenter__(self):
        self.redis = await aioredis.from_url(self.redis_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.redis:
            await self.redis.close()

    async def add_urls(self, urls: List[str], metadata: Optional[Dict[str, Any]] = None) -> int:
        """添加URLs到队列"""
        if not urls:
            return 0

        pipeline = self.redis.pipeline()
        count = 0
        for url in urls:
            if metadata:
                task = {"url": url, "metadata": metadata, "added_at": datetime.now().isoformat()}
            else:
                task = {"url": url, "added_at": datetime.now().isoformat()}
            
            pipeline.sadd(self.queue_key, json.dumps(task))
            count += 1
        
        await pipeline.execute()
        logger.info(f"Added {count} URLs to queue")
        return count

    async def get_next_url(self) -> Optional[Dict[str, Any]]:
        """获取下一个要处理的URL"""
        # 将任务从队列移动到处理中
        task_json = await self.redis.spop(self.queue_key)
        if not task_json:
            return None

        task = json.loads(task_json)
        task["processing_started"] = datetime.now().isoformat()
        
        # 添加到处理中集合
        await self.redis.hset(
            self.processing_key,
            task["url"],
            json.dumps(task)
        )
        
        return task

    async def mark_done(self, url: str, result: Optional[Dict[str, Any]] = None):
        """标记URL为已完成"""
        # 从处理中移除
        task_json = await self.redis.hget(self.processing_key, url)
        if task_json:
            task = json.loads(task_json)
            task["completed_at"] = datetime.now().isoformat()
            if result:
                task["result"] = result
            
            # 添加到完成集合
            await self.redis.hset(self.done_key, url, json.dumps(task))
            await self.redis.hdel(self.processing_key, url)
            
            logger.info(f"Marked {url} as done")

    async def mark_failed(self, url: str, error: str):
        """标记URL为失败"""
        task_json = await self.redis.hget(self.processing_key, url)
        if task_json:
            task = json.loads(task_json)
            task["failed_at"] = datetime.now().isoformat()
            task["error"] = error
            
            # 添加到失败集合
            await self.redis.hset(self.failed_key, url, json.dumps(task))
            await self.redis.hdel(self.processing_key, url)
            
            logger.error(f"Marked {url} as failed: {error}")

    async def requeue_timeout_tasks(self):
        """重新入队超时的任务"""
        now = datetime.now()
        timeout_threshold = now - timedelta(seconds=self.processing_timeout)
        
        # 获取所有处理中的任务
        processing_tasks = await self.redis.hgetall(self.processing_key)
        
        for url, task_json in processing_tasks.items():
            task = json.loads(task_json)
            started_at = datetime.fromisoformat(task["processing_started"])
            
            if started_at < timeout_threshold:
                # 移回队列
                await self.redis.sadd(self.queue_key, json.dumps({
                    "url": task["url"],
                    "metadata": task.get("metadata"),
                    "added_at": datetime.now().isoformat(),
                    "requeued": True,
                    "previous_attempt": task
                }))
                await self.redis.hdel(self.processing_key, url)
                logger.warning(f"Requeued timeout task: {url}")

    async def get_stats(self) -> Dict[str, int]:
        """获取调度器统计信息"""
        return {
            "queued": await self.redis.scard(self.queue_key),
            "processing": await self.redis.hlen(self.processing_key),
            "failed": await self.redis.hlen(self.failed_key),
            "done": await self.redis.hlen(self.done_key)
        }

class Scheduler:
    """分布式调度器"""
    
    def __init__(self, name: str):
        self.name = name
        self.redis: Optional[aioredis.Redis] = None
        self.pending_key = f'{name}:pending'
        self.processing_key = f'{name}:processing'
        self.failed_key = f'{name}:failed'
        self.success_key = f'{name}:success'
        
    async def connect(self):
        """连接到Redis"""
        self.redis = await aioredis.from_url(REDIS_URI)
        
    async def close(self):
        """关闭Redis连接"""
        if self.redis:
            await self.redis.close()
            
    async def add_task(self, task: Task):
        """添加新任务到队列"""
        if not self.redis:
            raise RuntimeError("Scheduler not connected to Redis")
            
        task_data = json.dumps(task.to_dict())
        await self.redis.zadd(self.pending_key, {task_data: task.priority})
        logger.info(f"Added task: {task.url} with priority {task.priority}")
        
    async def get_next_task(self) -> Optional[Task]:
        """获取下一个要处理的任务"""
        if not self.redis:
            raise RuntimeError("Scheduler not connected to Redis")
            
        # 获取优先级最高的任务
        result = await self.redis.zpopmax(self.pending_key)
        if not result:
            return None
            
        task_data = json.loads(result[0][0])
        task = Task.from_dict(task_data)
        
        # 将任务移到处理中队列
        await self.redis.hset(self.processing_key, task.url, task_data)
        return task
        
    async def mark_task_complete(self, task: Task):
        """标记任务为完成"""
        if not self.redis:
            raise RuntimeError("Scheduler not connected to Redis")
            
        # 从处理中队列移除
        await self.redis.hdel(self.processing_key, task.url)
        
        # 添加到成功队列
        task_data = json.dumps(task.to_dict())
        await self.redis.hset(self.success_key, task.url, task_data)
        logger.success(f"Task completed: {task.url}")
        
    async def mark_task_failed(self, task: Task, error: str):
        """标记任务为失败"""
        if not self.redis:
            raise RuntimeError("Scheduler not connected to Redis")
            
        # 从处理中队列移除
        await self.redis.hdel(self.processing_key, task.url)
        
        # 更新重试次数和错误信息
        task.retry_count += 1
        task.metadata['last_error'] = error
        task.metadata['failed_at'] = datetime.now().isoformat()
        
        task_data = json.dumps(task.to_dict())
        
        # 如果还可以重试，放回待处理队列
        if task.retry_count < 3:
            await self.redis.zadd(self.pending_key, {task_data: task.priority - 1})
            logger.warning(f"Task failed, retrying: {task.url}")
        else:
            # 否则放入失败队列
            await self.redis.hset(self.failed_key, task.url, task_data)
            logger.error(f"Task failed permanently: {task.url}")
            
    async def get_stats(self) -> Dict[str, int]:
        """获取任务统计信息"""
        if not self.redis:
            raise RuntimeError("Scheduler not connected to Redis")
            
        pending = await self.redis.zcard(self.pending_key)
        processing = await self.redis.hlen(self.processing_key)
        failed = await self.redis.hlen(self.failed_key)
        success = await self.redis.hlen(self.success_key)
        
        return {
            'pending': pending,
            'processing': processing,
            'failed': failed,
            'success': success
        }
        
    async def clear_all(self):
        """清除所有队列数据（用于测试）"""
        if not self.redis:
            raise RuntimeError("Scheduler not connected to Redis")
            
        await self.redis.delete(
            self.pending_key,
            self.processing_key,
            self.failed_key,
            self.success_key
        ) 