import mmh3
from typing import Optional
import aioredis
from loguru import logger
from config.settings import BLOOM_CONFIG

class BloomFilter:
    def __init__(
        self,
        redis_url: str,
        key: str = "spark:bloom",
        capacity: int = BLOOM_CONFIG["capacity"],
        error_rate: float = BLOOM_CONFIG["error_rate"]
    ):
        """
        初始化Bloom Filter
        
        Args:
            redis_url: Redis连接URL
            key: Bloom Filter在Redis中的键名
            capacity: 预期元素数量
            error_rate: 可接受的错误率
        """
        self.redis_url = redis_url
        self.key = key
        self.capacity = capacity
        self.error_rate = error_rate
        
        # 计算最优参数
        self.size = self.get_size(capacity, error_rate)
        self.hash_count = self.get_hash_count(self.size, capacity)
        
        self.redis: Optional[aioredis.Redis] = None
        
    async def __aenter__(self):
        self.redis = await aioredis.from_url(self.redis_url)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.redis:
            await self.redis.close()
            
    @staticmethod
    def get_size(capacity: int, error_rate: float) -> int:
        """计算最优位数组大小"""
        size = -1 * capacity * (2.303 * (error_rate ** 2))
        return int(size)
        
    @staticmethod
    def get_hash_count(size: int, capacity: int) -> int:
        """计算最优哈希函数数量"""
        k = (size / capacity) * 0.693
        return int(k)
        
    def _get_offsets(self, item: str) -> list[int]:
        """计算元素的所有哈希位置"""
        offsets = []
        for seed in range(self.hash_count):
            hash_val = mmh3.hash(item, seed)
            offset = abs(hash_val) % self.size
            offsets.append(offset)
        return offsets
        
    async def add(self, item: str) -> bool:
        """
        添加元素到Bloom Filter
        
        Returns:
            bool: True表示元素是新的，False表示可能已存在
        """
        offsets = self._get_offsets(item)
        
        # 检查所有位是否都已设置
        all_set = True
        pipe = self.redis.pipeline()
        
        for offset in offsets:
            # 使用GETBIT检查位是否已设置
            pipe.getbit(self.key, offset)
        
        # 执行检查
        results = await pipe.execute()
        all_set = all(results)
        
        if not all_set:
            # 设置新的位
            pipe = self.redis.pipeline()
            for offset in offsets:
                pipe.setbit(self.key, offset, 1)
            await pipe.execute()
            
        return not all_set
        
    async def exists(self, item: str) -> bool:
        """
        检查元素是否可能存在
        
        Returns:
            bool: True表示元素可能存在，False表示一定不存在
        """
        offsets = self._get_offsets(item)
        
        pipe = self.redis.pipeline()
        for offset in offsets:
            pipe.getbit(self.key, offset)
            
        results = await pipe.execute()
        return all(results)
        
    async def bulk_exists(self, items: list[str]) -> list[bool]:
        """批量检查多个元素"""
        results = []
        for item in items:
            exists = await self.exists(item)
            results.append(exists)
        return results
    
    async def clear(self):
        """清空Bloom Filter"""
        await self.redis.delete(self.key)
        logger.info(f"Cleared Bloom Filter: {self.key}")
        
    async def get_info(self) -> dict:
        """获取Bloom Filter信息"""
        return {
            "size": self.size,
            "hash_count": self.hash_count,
            "capacity": self.capacity,
            "error_rate": self.error_rate,
            "key": self.key
        } 