import asyncio
from typing import List, Dict, Optional, Set
import aiohttp
from datetime import datetime, timedelta
import random
from loguru import logger
import json
import aioredis

class ProxyPool:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        proxy_key: str = "spark:proxies",
        min_score: float = 0.0,
        max_score: float = 10.0,
        initial_score: float = 5.0
    ):
        self.redis_url = redis_url
        self.proxy_key = proxy_key
        self.min_score = min_score
        self.max_score = max_score
        self.initial_score = initial_score
        self.redis: Optional[aioredis.Redis] = None

    async def __aenter__(self):
        self.redis = await aioredis.from_url(self.redis_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.redis:
            await self.redis.close()

    async def add_proxy(self, proxy: str):
        """添加新代理"""
        await self.redis.hset(
            self.proxy_key,
            proxy,
            json.dumps({
                "score": self.initial_score,
                "added_at": datetime.now().isoformat(),
                "last_used": None,
                "success_count": 0,
                "fail_count": 0
            })
        )
        logger.info(f"Added new proxy: {proxy}")

    async def add_proxies(self, proxies: List[str]):
        """批量添加代理"""
        pipeline = self.redis.pipeline()
        for proxy in proxies:
            pipeline.hset(
                self.proxy_key,
                proxy,
                json.dumps({
                    "score": self.initial_score,
                    "added_at": datetime.now().isoformat(),
                    "last_used": None,
                    "success_count": 0,
                    "fail_count": 0
                })
            )
        await pipeline.execute()
        logger.info(f"Added {len(proxies)} new proxies")

    async def get_proxy(self) -> Optional[str]:
        """获取一个可用代理"""
        proxies = await self.redis.hgetall(self.proxy_key)
        if not proxies:
            return None

        # 按分数排序代理
        valid_proxies = []
        for proxy, data in proxies.items():
            proxy_data = json.loads(data)
            if proxy_data["score"] > self.min_score:
                valid_proxies.append((proxy, proxy_data["score"]))

        if not valid_proxies:
            return None

        # 根据分数加权随机选择
        total_score = sum(score for _, score in valid_proxies)
        if total_score <= 0:
            return None

        r = random.uniform(0, total_score)
        current_sum = 0
        for proxy, score in valid_proxies:
            current_sum += score
            if current_sum >= r:
                # 更新最后使用时间
                data = json.loads(proxies[proxy])
                data["last_used"] = datetime.now().isoformat()
                await self.redis.hset(self.proxy_key, proxy, json.dumps(data))
                return proxy

        return valid_proxies[-1][0] if valid_proxies else None

    async def report_proxy_status(self, proxy: str, success: bool):
        """报告代理使用状态"""
        data = await self.redis.hget(self.proxy_key, proxy)
        if not data:
            return

        proxy_data = json.loads(data)
        if success:
            proxy_data["success_count"] += 1
            proxy_data["score"] = min(
                self.max_score,
                proxy_data["score"] * 1.2
            )
        else:
            proxy_data["fail_count"] += 1
            proxy_data["score"] *= 0.5

        if proxy_data["score"] <= self.min_score:
            # 删除不可用代理
            await self.redis.hdel(self.proxy_key, proxy)
            logger.warning(f"Removed bad proxy: {proxy}")
        else:
            await self.redis.hset(
                self.proxy_key,
                proxy,
                json.dumps(proxy_data)
            )

    async def clean_proxies(self, max_age_hours: int = 24):
        """清理老旧代理"""
        proxies = await self.redis.hgetall(self.proxy_key)
        if not proxies:
            return

        now = datetime.now()
        for proxy, data in proxies.items():
            proxy_data = json.loads(data)
            added_at = datetime.fromisoformat(proxy_data["added_at"])
            if (now - added_at).total_seconds() > max_age_hours * 3600:
                await self.redis.hdel(self.proxy_key, proxy)
                logger.info(f"Cleaned old proxy: {proxy}")

    async def get_stats(self) -> Dict[str, int]:
        """获取代理池统计信息"""
        proxies = await self.redis.hgetall(self.proxy_key)
        if not proxies:
            return {"total": 0, "good": 0, "bad": 0}

        good_count = 0
        bad_count = 0
        for data in proxies.values():
            proxy_data = json.loads(data)
            if proxy_data["score"] > self.min_score:
                good_count += 1
            else:
                bad_count += 1

        return {
            "total": len(proxies),
            "good": good_count,
            "bad": bad_count
        } 