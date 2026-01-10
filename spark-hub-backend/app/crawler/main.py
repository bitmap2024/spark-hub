import os
import sys
from typing import Optional, Dict, Any
import argparse
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from loguru import logger
from core.spider import SparkSpider, DynamicSpider, ProxySpider
from config.settings import CRAWL_RULES

def setup_logging():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/crawler.log",
        rotation="500 MB",
        retention="10 days"
    )

def get_spider_class(spider_type: str):
    """获取爬虫类"""
    spider_types = {
        'basic': SparkSpider,
        'dynamic': DynamicSpider,
        'proxy': ProxySpider
    }
    return spider_types.get(spider_type, SparkSpider)

def get_settings():
    """获取Scrapy设置"""
    settings = get_project_settings()
    
    # 基础设置
    settings.update({
        'BOT_NAME': 'spark-hub-crawler',
        'SPIDER_MODULES': ['core.spider'],
        'NEWSPIDER_MODULE': 'core.spider',
        'USER_AGENT': 'Spark Hub Crawler (+https://example.com)',
        
        # 并发设置
        'CONCURRENT_REQUESTS': 32,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'CONCURRENT_REQUESTS_PER_IP': 8,
        
        # 下载设置
        'DOWNLOAD_DELAY': 1,
        'DOWNLOAD_TIMEOUT': 30,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        
        # 重试设置
        'RETRY_ENABLED': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 522, 524, 408, 429],
        
        # 缓存设置
        'HTTPCACHE_ENABLED': True,
        'HTTPCACHE_EXPIRATION_SECS': 0,
        'HTTPCACHE_DIR': 'httpcache',
        'HTTPCACHE_IGNORE_HTTP_CODES': [],
        'HTTPCACHE_STORAGE': 'scrapy.extensions.httpcache.FilesystemCacheStorage',
        
        # 管道设置
        'ITEM_PIPELINES': {
            'pipeline.pipelines.ValidationPipeline': 100,
            'pipeline.pipelines.DuplicatesPipeline': 200,
            'pipeline.pipelines.MongoDBPipeline': 300,
            'pipeline.pipelines.S3Pipeline': 400,
            'pipeline.pipelines.ImagesPipeline': 500,
        },
        
        # 中间件设置
        'SPIDER_MIDDLEWARES': {
            'middleware.scrapy_middleware.BloomFilterMiddleware': 100,
            'middleware.scrapy_middleware.MetricsMiddleware': 200,
        },
        'DOWNLOADER_MIDDLEWARES': {
            'middleware.scrapy_middleware.ProxyMiddleware': 100,
            'middleware.scrapy_middleware.RetryMiddleware': 200,
        },
        
        # 扩展设置
        'EXTENSIONS': {
            'scrapy.extensions.telnet.TelnetConsole': None,
            'scrapy.extensions.corestats.CoreStats': 100,
            'scrapy.extensions.memusage.MemoryUsage': 200,
            'scrapy.extensions.logstats.LogStats': 300,
        },
    })
    
    return settings

def run_spider(
    spider_name: str,
    spider_type: str = 'basic',
    allowed_domains: Optional[list[str]] = None,
    start_urls: Optional[list[str]] = None,
    custom_settings: Optional[Dict[str, Any]] = None
):
    """运行爬虫"""
    try:
        # 创建爬虫进程
        process = CrawlerProcess(get_settings())
        
        # 获取爬虫类
        spider_class = get_spider_class(spider_type)
        
        # 获取爬虫规则
        spider_rules = CRAWL_RULES.get(spider_name, {})
        
        # 启动爬虫
        process.crawl(
            spider_class,
            name=spider_name,
            allowed_domains=allowed_domains or spider_rules.get('allowed_domains'),
            start_urls=start_urls or spider_rules.get('start_urls'),
            rules=spider_rules.get('rules'),
            custom_settings=custom_settings or spider_rules.get('custom_settings')
        )
        
        # 启动进程
        process.start()
        
    except Exception as e:
        logger.error(f"Error running spider: {str(e)}")
        sys.exit(1)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Spark Hub Crawler")
    parser.add_argument("spider_name", help="爬虫名称")
    parser.add_argument("--type", choices=['basic', 'dynamic', 'proxy'], default='basic', help="爬虫类型")
    parser.add_argument("--domains", nargs="+", help="允许的域名列表")
    parser.add_argument("--urls", nargs="+", help="起始URL列表")
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 运行爬虫
    run_spider(
        spider_name=args.spider_name,
        spider_type=args.type,
        allowed_domains=args.domains,
        start_urls=args.urls
    )

if __name__ == "__main__":
    main() 