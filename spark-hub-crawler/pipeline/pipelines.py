from typing import Dict, Any, Optional
from datetime import datetime
import json
import boto3
from motor.motor_asyncio import AsyncIOMotorClient
from scrapy.exceptions import DropItem
from loguru import logger
from config.settings import MONGO_CONFIG, S3_CONFIG

class MongoDBPipeline:
    """MongoDB存储管道"""
    
    def __init__(
        self,
        mongo_url: str = MONGO_CONFIG["url"],
        db_name: str = MONGO_CONFIG["db"],
        collection_name: str = MONGO_CONFIG["collections"]["items"]
    ):
        self.mongo_url = mongo_url
        self.db_name = db_name
        self.collection_name = collection_name
        self.client: Optional[AsyncIOMotorClient] = None
        
    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_url=crawler.settings.get('MONGO_URL', MONGO_CONFIG["url"]),
            db_name=crawler.settings.get('MONGO_DB', MONGO_CONFIG["db"]),
            collection_name=crawler.settings.get(
                'MONGO_COLLECTION',
                MONGO_CONFIG["collections"]["items"]
            )
        )
        
    async def open_spider(self, spider):
        """爬虫启动时连接数据库"""
        self.client = AsyncIOMotorClient(self.mongo_url)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        # 创建索引
        await self.collection.create_index('url', unique=True)
        await self.collection.create_index('crawled_at')
        
    async def close_spider(self, spider):
        """爬虫关闭时关闭数据库连接"""
        if self.client:
            self.client.close()
            
    async def process_item(self, item: Dict[str, Any], spider):
        """处理爬取到的数据项"""
        try:
            # 添加元数据
            item['spider'] = spider.name
            item['updated_at'] = datetime.now().isoformat()
            
            # 更新或插入数据
            await self.collection.update_one(
                {'url': item['url']},
                {'$set': item},
                upsert=True
            )
            
            return item
            
        except Exception as e:
            logger.error(f"Error saving to MongoDB: {str(e)}")
            raise DropItem(f"Failed to save item: {str(e)}")

class S3Pipeline:
    """S3存储管道"""
    
    def __init__(
        self,
        bucket: str = S3_CONFIG["bucket"],
        prefix: str = S3_CONFIG["prefix"],
        region: str = S3_CONFIG["region"]
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.s3 = boto3.client('s3', region_name=region)
        
    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            bucket=crawler.settings.get('S3_BUCKET', S3_CONFIG["bucket"]),
            prefix=crawler.settings.get('S3_PREFIX', S3_CONFIG["prefix"]),
            region=crawler.settings.get('AWS_REGION', S3_CONFIG["region"])
        )
        
    def process_item(self, item: Dict[str, Any], spider):
        """处理爬取到的数据项"""
        try:
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d/%H%M%S')
            filename = f"{self.prefix}/{spider.name}/{timestamp}/{item['url'].split('/')[-1]}.json"
            
            # 上传到S3
            self.s3.put_object(
                Bucket=self.bucket,
                Key=filename,
                Body=json.dumps(item, ensure_ascii=False),
                ContentType='application/json'
            )
            
            # 添加S3路径到item
            item['s3_path'] = f"s3://{self.bucket}/{filename}"
            
            return item
            
        except Exception as e:
            logger.error(f"Error saving to S3: {str(e)}")
            raise DropItem(f"Failed to save item: {str(e)}")

class DuplicatesPipeline:
    """数据去重管道"""
    
    def __init__(self):
        self.urls_seen = set()
        
    def process_item(self, item: Dict[str, Any], spider):
        if item['url'] in self.urls_seen:
            raise DropItem(f"Duplicate item found: {item['url']}")
        self.urls_seen.add(item['url'])
        return item

class ValidationPipeline:
    """数据验证管道"""
    
    required_fields = {'url', 'title', 'content', 'crawled_at'}
    
    def process_item(self, item: Dict[str, Any], spider):
        for field in self.required_fields:
            if not item.get(field):
                raise DropItem(f"Missing {field} in {item['url']}")
        return item

class ImagesPipeline:
    """图片处理管道"""
    
    def __init__(
        self,
        store_uri: str = None,
        download_func=None,
        settings=None
    ):
        self.store_uri = store_uri
        self.download_func = download_func
        self.settings = settings
        
    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            store_uri=crawler.settings.get('IMAGES_STORE'),
            download_func=crawler.engine.download,
            settings=crawler.settings
        )
        
    async def process_item(self, item: Dict[str, Any], spider):
        if 'screenshot' in item:
            try:
                # 生成文件名
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"screenshots/{spider.name}/{timestamp}_{item['url'].split('/')[-1]}.jpg"
                
                # 保存截图
                with open(filename, 'wb') as f:
                    f.write(item['screenshot'])
                    
                # 更新item
                item['screenshot_path'] = filename
                del item['screenshot']  # 删除原始截图数据
                
            except Exception as e:
                logger.error(f"Error saving screenshot: {str(e)}")
                
        return item 