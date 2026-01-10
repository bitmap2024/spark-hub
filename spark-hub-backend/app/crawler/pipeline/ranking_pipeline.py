from typing import Dict, Any, List
import numpy as np
from datetime import datetime
from urllib.parse import urlparse
from collections import defaultdict
from loguru import logger
import networkx as nx
from redis import Redis
from sklearn.preprocessing import MinMaxScaler
from bs4 import BeautifulSoup
import re
import time
from config.settings import REDIS_CONFIG, RANKING_CONFIG

class ModernRankingPipeline:
    """现代搜索引擎排名算法"""
    
    def __init__(
        self,
        redis_url: str = REDIS_CONFIG["url"],
        config: Dict = RANKING_CONFIG
    ):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.config = config
        self.graph = nx.DiGraph()
        self.scaler = MinMaxScaler()
        
    def calculate_content_quality(self, item: Dict[str, Any]) -> float:
        """计算内容质量分数"""
        score = 0.0
        content = item.get("content", "")
        
        if not content:
            return score
            
        # 1. 内容长度评分
        length = len(content)
        if length > 2000:
            score += 1.0
        elif length > 1000:
            score += 0.7
        elif length > 500:
            score += 0.4
            
        # 2. 结构完整性
        if item.get("title"):
            score += 0.5
        if item.get("meta_description"):
            score += 0.3
            
        # 3. 图文混排评分
        images = item.get("images", [])
        if images:
            score += min(len(images) * 0.2, 1.0)
            
        # 4. 代码质量（如果是技术文章）
        code_blocks = item.get("code_blocks", [])
        if code_blocks:
            score += min(len(code_blocks) * 0.3, 1.0)
            
        # 5. 内部链接评分
        internal_links = [
            link for link in item.get("links", [])
            if self.get_domain(link) == self.get_domain(item["url"])
        ]
        if internal_links:
            score += min(len(internal_links) * 0.1, 0.5)
            
        return min(score, 5.0)  # 最高5分
        
    def calculate_time_score(self, timestamp: str) -> float:
        """计算时效性分数"""
        try:
            pub_time = datetime.fromisoformat(timestamp)
            now = datetime.now()
            age_days = (now - pub_time).days
            
            # 时效性衰减函数
            if age_days <= 7:  # 一周内
                return 1.0
            elif age_days <= 30:  # 一月内
                return 0.8
            elif age_days <= 90:  # 三月内
                return 0.6
            elif age_days <= 365:  # 一年内
                return 0.4
            else:
                return 0.2
                
        except:
            return 0.5  # 默认分数
            
    def calculate_user_metrics(self, url: str) -> float:
        """计算用户行为指标"""
        try:
            metrics = self.redis.hgetall(f"metrics:{url}")
            if not metrics:
                return 0.5  # 默认分数
                
            # 1. 跳出率评分
            bounce_rate = float(metrics.get("bounce_rate", 0))
            bounce_score = 1 - min(bounce_rate, 0.8)  # 跳出率越低越好
            
            # 2. 平均访问时长
            avg_time = float(metrics.get("avg_time", 0))
            time_score = min(avg_time / 300, 1)  # 5分钟封顶
            
            # 3. 点击率
            ctr = float(metrics.get("ctr", 0))
            ctr_score = min(ctr * 10, 1)  # CTR * 10，最高1分
            
            return (bounce_score + time_score + ctr_score) / 3
            
        except:
            return 0.5
            
    def calculate_technical_score(self, item: Dict[str, Any]) -> float:
        """计算技术实现分数"""
        score = 0.0
        
        # 1. 移动适配
        if item.get("mobile_friendly", False):
            score += 1.0
            
        # 2. 页面加载速度
        load_time = float(item.get("load_time", 5000))  # 默认5000ms
        if load_time < 1000:  # 1秒内
            score += 1.0
        elif load_time < 2000:  # 2秒内
            score += 0.7
        elif load_time < 3000:  # 3秒内
            score += 0.4
            
        # 3. HTTPS
        if item["url"].startswith("https"):
            score += 0.5
            
        # 4. 结构化数据
        if item.get("structured_data"):
            score += 0.5
            
        return min(score, 3.0)  # 最高3分
        
    def calculate_authority_score(self, domain: str) -> float:
        """计算域名权威度"""
        try:
            # 1. 基础PageRank得分
            pagerank = self.get_domain_rank(domain)
            
            # 2. 域名年龄
            domain_info = self.redis.hgetall(f"domain:{domain}")
            age_years = float(domain_info.get("age_years", 0))
            age_score = min(age_years / 10, 1)  # 10年封顶
            
            # 3. 反向链接质量
            backlinks = int(domain_info.get("quality_backlinks", 0))
            backlink_score = min(backlinks / 1000, 1)  # 1000个优质反链封顶
            
            # 4. 社交信号
            social = int(domain_info.get("social_signals", 0))
            social_score = min(social / 10000, 1)  # 10000个社交信号封顶
            
            return (pagerank + age_score + backlink_score + social_score) / 4
            
        except:
            return 0.5
            
    def calculate_final_score(self, item: Dict[str, Any]) -> float:
        """计算最终排名分数"""
        # 1. 基础PageRank
        pagerank = self.get_pagerank(item["url"])
        
        # 2. 内容质量
        content_score = self.calculate_content_quality(item)
        
        # 3. 时效性
        time_score = self.calculate_time_score(
            item.get("published_at", datetime.now().isoformat())
        )
        
        # 4. 用户行为
        user_score = self.calculate_user_metrics(item["url"])
        
        # 5. 技术实现
        tech_score = self.calculate_technical_score(item)
        
        # 6. 域名权威度
        authority = self.calculate_authority_score(
            self.get_domain(item["url"])
        )
        
        # 权重配置
        weights = self.config["ranking_weights"]
        
        # 计算加权得分
        final_score = (
            pagerank * weights["pagerank"] +
            content_score * weights["content"] +
            time_score * weights["time"] +
            user_score * weights["user"] +
            tech_score * weights["technical"] +
            authority * weights["authority"]
        )
        
        return final_score
        
    def process_item(self, item: Dict[str, Any], spider) -> Dict[str, Any]:
        """处理爬取项"""
        try:
            # 1. 更新图结构（用于PageRank）
            self._update_graph(item)
            
            # 2. 计算综合排名分数
            score = self.calculate_final_score(item)
            
            # 3. 存储分数
            self.redis.hset(
                "ranking:scores",
                item["url"],
                float(score)
            )
            
            # 4. 更新item
            item["ranking_score"] = score
            
            return item
            
        except Exception as e:
            logger.error(f"Error in ranking pipeline: {str(e)}")
            return item
            
    def _update_graph(self, item: Dict[str, Any]):
        """更新链接图"""
        current_url = item["url"]
        outgoing_links = item.get("links", [])
        
        # 添加当前节点
        if not self.graph.has_node(current_url):
            self.graph.add_node(
                current_url,
                domain=self.get_domain(current_url),
                title=item.get("title", ""),
                last_updated=datetime.now().isoformat()
            )
        
        # 添加出链
        for link in outgoing_links:
            if not self.graph.has_node(link):
                self.graph.add_node(
                    link,
                    domain=self.get_domain(link),
                    title="",
                    last_updated=datetime.now().isoformat()
                )
            self.graph.add_edge(current_url, link)
            
    def get_domain(self, url: str) -> str:
        """获取URL的域名"""
        return urlparse(url).netloc
        
    def get_pagerank(self, url: str) -> float:
        """获取URL的PageRank值"""
        try:
            rank = self.redis.hget("pagerank:scores", url)
            return float(rank) if rank else 0.0
        except:
            return 0.0
            
    def get_domain_rank(self, domain: str) -> float:
        """获取域名的PageRank值"""
        try:
            rank = self.redis.hget("pagerank:domain_scores", domain)
            return float(rank) if rank else 0.0
        except:
            return 0.0
            
    def close_spider(self, spider):
        """爬虫关闭时的清理工作"""
        # 最后更新一次排名
        self._update_all_rankings()
        
        # 记录统计信息
        stats = {
            "total_pages": self.graph.number_of_nodes(),
            "total_links": self.graph.number_of_edges(),
            "total_domains": len({
                self.get_domain(url)
                for url in self.graph.nodes
            }),
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis.hmset("ranking:stats", stats)
        logger.info(f"Ranking pipeline closed: {stats}")
        
    def _update_all_rankings(self):
        """更新所有页面的排名"""
        try:
            # 1. 重新计算PageRank
            pagerank = nx.pagerank(
                self.graph,
                alpha=self.config["pagerank"]["damping_factor"],
                tol=self.config["pagerank"]["convergence_threshold"],
                max_iter=self.config["pagerank"]["max_iterations"]
            )
            
            # 2. 更新Redis中的分数
            pipeline = self.redis.pipeline()
            
            for url, rank in pagerank.items():
                pipeline.hset(
                    "pagerank:scores",
                    url,
                    float(rank)
                )
                
            pipeline.execute()
            
        except Exception as e:
            logger.error(f"Error updating rankings: {str(e)}") 