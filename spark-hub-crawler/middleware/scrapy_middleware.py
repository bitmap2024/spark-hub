from typing import Optional, Any
from scrapy import signals
from scrapy.http import Request, Response
from scrapy.spiders import Spider
from scrapy.exceptions import IgnoreRequest, NotConfigured
from loguru import logger
import asyncio
from datetime import datetime
from middleware.metrics import CrawlerMetrics
from middleware.bloom import BloomFilter
from middleware.proxy import ProxyPool
from urllib.parse import urlparse

class MetricsMiddleware:
    """Prometheus监控中间件"""
    
    @classmethod
    def from_crawler(cls, crawler):
        middleware = cls()
        crawler.signals.connect(middleware.spider_opened, signal=signals.spider_opened)
        return middleware
        
    def spider_opened(self, spider):
        self.metrics = CrawlerMetrics()
        
    def process_request(self, request: Request, spider: Spider):
        domain = urlparse(request.url).netloc
        request.meta['metrics_tracker'] = self.metrics.track_request(domain)
        return None
        
    def process_response(self, request: Request, response: Response, spider: Spider):
        tracker = request.meta.get('metrics_tracker')
        if tracker:
            tracker.set_status(response.status)
        return response
        
    def process_exception(self, request: Request, exception: Exception, spider: Spider):
        self.metrics.record_error(type(exception).__name__)
        return None

class BloomFilterMiddleware:
    """URL去重中间件"""
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('BLOOM_FILTER_ENABLED', True):
            raise NotConfigured
            
        return cls(
            redis_url=crawler.settings.get('REDIS_URL'),
            key_prefix=crawler.settings.get('BLOOM_FILTER_KEY', 'spark:bloom')
        )
        
    def __init__(self, redis_url: str, key_prefix: str):
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.filters: dict[str, BloomFilter] = {}
        
    async def get_filter(self, spider_name: str) -> BloomFilter:
        """获取或创建BloomFilter"""
        if spider_name not in self.filters:
            key = f"{self.key_prefix}:{spider_name}"
            bloom = BloomFilter(redis_url=self.redis_url, key=key)
            await bloom.__aenter__()
            self.filters[spider_name] = bloom
        return self.filters[spider_name]
        
    async def process_request(self, request: Request, spider: Spider):
        if request.meta.get('dont_filter', False):
            return None
            
        bloom = await self.get_filter(spider.name)
        exists = await bloom.exists(request.url)
        
        if exists:
            raise IgnoreRequest(f"URL already seen: {request.url}")
            
        await bloom.add(request.url)
        return None
        
    async def spider_closed(self, spider: Spider):
        if spider.name in self.filters:
            await self.filters[spider.name].__aexit__(None, None, None)
            del self.filters[spider.name]

class ProxyMiddleware:
    """代理中间件"""
    
    @classmethod
    def from_crawler(cls, crawler):
        if not crawler.settings.getbool('PROXY_ENABLED', False):
            raise NotConfigured
            
        return cls(
            redis_url=crawler.settings.get('REDIS_URL'),
            proxy_key=crawler.settings.get('PROXY_KEY', 'spark:proxies')
        )
        
    def __init__(self, redis_url: str, proxy_key: str):
        self.proxy_pool = ProxyPool(redis_url=redis_url, proxy_key=proxy_key)
        
    async def process_request(self, request: Request, spider: Spider):
        if 'proxy' not in request.meta:
            proxy = await self.proxy_pool.get_proxy()
            if proxy:
                request.meta['proxy'] = proxy
                request.meta['proxy_original'] = proxy
        return None
        
    async def process_response(self, request: Request, response: Response, spider: Spider):
        proxy = request.meta.get('proxy_original')
        if proxy:
            if response.status in [200, 201, 301, 302, 304]:
                await self.proxy_pool.report_proxy_status(proxy, True)
            else:
                await self.proxy_pool.report_proxy_status(proxy, False)
        return response
        
    async def process_exception(self, request: Request, exception: Exception, spider: Spider):
        proxy = request.meta.get('proxy_original')
        if proxy:
            await self.proxy_pool.report_proxy_status(proxy, False)
        return None

class RetryMiddleware:
    """重试中间件"""
    
    RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]
    
    def __init__(self, settings):
        self.max_retry_times = settings.getint('RETRY_TIMES', 3)
        self.retry_http_codes = set(settings.getlist('RETRY_HTTP_CODES', self.RETRY_HTTP_CODES))
        self.priority_adjust = settings.getint('RETRY_PRIORITY_ADJUST', -1)
        
    @classmethod
    def from_crawler(cls, crawler):
        return cls(crawler.settings)
        
    def process_response(self, request: Request, response: Response, spider: Spider):
        if request.meta.get('dont_retry', False):
            return response
            
        if response.status in self.retry_http_codes:
            return self._retry(request, response.status, spider) or response
            
        return response
        
    def process_exception(self, request: Request, exception: Exception, spider: Spider):
        if request.meta.get('dont_retry', False):
            return None
            
        return self._retry(request, exception, spider)
        
    def _retry(self, request: Request, reason: Any, spider: Spider):
        retries = request.meta.get('retry_times', 0) + 1
        
        if retries <= self.max_retry_times:
            logger.info(f"Retrying {request.url} (failed {retries} times): {reason}")
            
            retryreq = request.copy()
            retryreq.meta['retry_times'] = retries
            retryreq.dont_filter = True
            retryreq.priority = request.priority + self.priority_adjust
            
            if isinstance(reason, int):
                retryreq.meta['retry_status'] = reason
            
            return retryreq
        
        logger.error(f"Gave up retrying {request.url} (failed {retries} times): {reason}")
        return None 