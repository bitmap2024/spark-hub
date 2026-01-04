from typing import Optional, Dict, Any, List
import asyncio
import aiohttp
from datetime import datetime
from urllib.parse import urlparse
from loguru import logger
from .optimization import (
    AsyncOptimizer,
    MemoryOptimizer,
    ResourcePool,
    MetricsCollector,
    AdaptiveThrottling,
    BatchProcessor
)
from config.settings import CRAWL_RULES, DEFAULT_HEADERS
from config.optimization import (
    CONCURRENCY_SETTINGS,
    ASYNC_SETTINGS,
    MEMORY_SETTINGS,
    NETWORK_SETTINGS
)

class SparkSpider:
    """Spark爬虫基类"""
    
    name = 'spark_spider'
    
    def __init__(
        self,
        name: str = None,
        allowed_domains: List[str] = None,
        start_urls: List[str] = None,
        rules: List[Dict[str, Any]] = None,
        custom_settings: Dict[str, Any] = None,
        *args,
        **kwargs
    ):
        """初始化爬虫"""
        if name:
            self.name = name
            
        # 基础配置
        self.allowed_domains = allowed_domains or CRAWL_RULES.get(self.name, {}).get('allowed_domains', [])
        self.start_urls = start_urls or CRAWL_RULES.get(self.name, {}).get('start_urls', [])
        self.custom_settings = custom_settings or {}
        
        # 初始化优化组件
        self.async_optimizer = AsyncOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.metrics_collector = MetricsCollector()
        self.throttling = AdaptiveThrottling()
        self.batch_processor = BatchProcessor()
        
        # 初始化资源池
        self.connection_pool = ResourcePool(
            max_size=CONCURRENCY_SETTINGS['connection_pool_size']
        )
        
        # 初始化统计
        self.stats = {
            'pages_crawled': 0,
            'items_scraped': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    async def setup(self):
        """设置爬虫环境"""
        await self.async_optimizer.setup()
    
    async def cleanup(self):
        """清理资源"""
        await self.async_optimizer.cleanup()
        await self.batch_processor.flush()
    
    def is_valid_url(self, url: str) -> bool:
        """检查URL是否有效"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
        except:
            return False
    
    async def fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        """获取页面内容"""
        if not self.is_valid_url(url):
            return None
            
        start_time = datetime.now()
        try:
            async with self.async_optimizer.session.get(url, headers=DEFAULT_HEADERS) as response:
                # 调整请求延迟
                delay = self.throttling.adjust_delay(response.status)
                await asyncio.sleep(delay)
                
                if response.status == 200:
                    # 压缩响应数据
                    content = await response.read()
                    compressed_content = self.memory_optimizer.compress_response(content)
                    
                    # 解析数据
                    item = await self.parse_response(response, compressed_content)
                    
                    # 批量处理
                    if item:
                        await self.batch_processor.process_item(item)
                    
                    # 记录指标
                    duration = (datetime.now() - start_time).total_seconds()
                    self.metrics_collector.record_request(duration, True)
                    
                    return item
                else:
                    logger.warning(f"Failed to fetch {url}, status: {response.status}")
                    self.metrics_collector.record_request(
                        (datetime.now() - start_time).total_seconds(),
                        False
                    )
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            self.stats['errors'] += 1
            return None
    
    async def parse_response(
        self,
        response: aiohttp.ClientResponse,
        content: bytes
    ) -> Optional[Dict[str, Any]]:
        """解析响应数据"""
        try:
            # 解压内容
            decompressed_content = self.memory_optimizer.decompress_response(content)
            
            # 示例解析逻辑
            item = {
                'url': str(response.url),
                'status': response.status,
                'headers': dict(response.headers),
                'content': decompressed_content.decode('utf-8'),
                'crawled_at': datetime.now().isoformat()
            }
            
            self.stats['items_scraped'] += 1
            return item
            
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            return None
    
    async def crawl(self):
        """开始爬取"""
        try:
            await self.setup()
            
            # 创建任务
            tasks = [
                self.fetch_page(url)
                for url in self.start_urls
            ]
            
            # 并发执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            valid_results = [
                result for result in results
                if isinstance(result, dict)
            ]
            
            return valid_results
            
        finally:
            await self.cleanup()
    
    def run(self):
        """运行爬虫"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.crawl())

class DynamicSpider(SparkSpider):
    """支持JavaScript渲染的爬虫"""
    
    async def setup(self):
        """设置动态爬虫环境"""
        await super().setup()
        
        # 初始化Playwright
        from playwright.async_api import async_playwright
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=['--no-sandbox']
        )
        
    async def cleanup(self):
        """清理动态爬虫资源"""
        if hasattr(self, 'browser'):
            await self.browser.close()
        if hasattr(self, 'playwright'):
            await self.playwright.stop()
        await super().cleanup()
    
    async def fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        """获取动态页面内容"""
        if not self.is_valid_url(url):
            return None
            
        start_time = datetime.now()
        try:
            page = await self.browser.new_page()
            await page.goto(url, wait_until='networkidle')
            
            # 获取渲染后的内容
            content = await page.content()
            title = await page.title()
            
            # 压缩内容
            compressed_content = self.memory_optimizer.compress_response(
                content.encode('utf-8')
            )
            
            # 构建数据项
            item = {
                'url': url,
                'title': title,
                'content': compressed_content,
                'screenshot': await page.screenshot(
                    full_page=True,
                    type='jpeg',
                    quality=80
                ),
                'crawled_at': datetime.now().isoformat()
            }
            
            # 记录指标
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics_collector.record_request(duration, True)
            
            await self.batch_processor.process_item(item)
            return item
            
        except Exception as e:
            logger.error(f"Error fetching dynamic page {url}: {str(e)}")
            self.stats['errors'] += 1
            return None
            
        finally:
            if 'page' in locals():
                await page.close()

class ProxySpider(SparkSpider):
    """支持代理的爬虫"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proxy_pool = ResourcePool(
            max_size=NETWORK_SETTINGS['proxy_settings']['proxy_pool_size']
        )
    
    async def fetch_page(self, url: str) -> Optional[Dict[str, Any]]:
        """使用代理获取页面"""
        if not self.is_valid_url(url):
            return None
            
        proxy = await self.proxy_pool.get()
        try:
            async with self.async_optimizer.session.get(
                url,
                headers=DEFAULT_HEADERS,
                proxy=proxy
            ) as response:
                # 处理响应
                if response.status == 200:
                    content = await response.read()
                    return await self.parse_response(response, content)
                else:
                    logger.warning(f"Proxy request failed: {url}, status: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Proxy request error: {url}, {str(e)}")
            return None
        finally:
            self.proxy_pool.put(proxy) 