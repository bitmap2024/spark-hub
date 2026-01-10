from typing import Dict, Any, List
import re
from datetime import datetime
from urllib.parse import urlparse
from loguru import logger
from redis import Redis
from bs4 import BeautifulSoup
import jieba
import jieba.analyse
from textblob import TextBlob
from config.settings import REDIS_CONFIG, BAIDU_ALGO_CONFIG

class BaiduAlgorithmsPipeline:
    """百度搜索引擎算法实现"""
    
    def __init__(
        self,
        redis_url: str = REDIS_CONFIG["url"],
        config: Dict = BAIDU_ALGO_CONFIG
    ):
        self.redis = Redis.from_url(redis_url, decode_responses=True)
        self.config = config
        
    def process_item(self, item: Dict[str, Any], spider) -> Dict[str, Any]:
        """处理爬取项"""
        try:
            # 1. 冰桶算法 - 反作弊检测
            if self.detect_spam(item):
                item["is_spam"] = True
                return item
                
            # 2. 绿萝算法 - 内容质量评估
            quality_score = self.evaluate_content_quality(item)
            item["content_quality_score"] = quality_score
            
            # 3. 飓风算法 - 移动体验评估
            mobile_score = self.evaluate_mobile_experience(item)
            item["mobile_score"] = mobile_score
            
            # 4. 石榴算法 - 时效性评估
            freshness_score = self.evaluate_freshness(item)
            item["freshness_score"] = freshness_score
            
            # 5. 清风算法 - 虚假信息检测
            credibility_score = self.evaluate_credibility(item)
            item["credibility_score"] = credibility_score
            
            # 存储评分
            self._store_scores(item)
            
            return item
            
        except Exception as e:
            logger.error(f"Error in Baidu algorithms: {str(e)}")
            return item
            
    def detect_spam(self, item: Dict[str, Any]) -> bool:
        """冰桶算法：检测作弊行为"""
        try:
            content = item.get("content", "")
            url = item["url"]
            
            # 1. 关键词堆砌检测
            keywords = jieba.analyse.extract_tags(content, topK=20)
            keyword_density = {}
            for word in keywords:
                density = content.count(word) / len(content)
                keyword_density[word] = density
                if density > self.config["spam"]["keyword_density_threshold"]:
                    logger.warning(f"Keyword stuffing detected: {url}")
                    return True
            
            # 2. 隐藏内容检测
            if self._detect_hidden_content(item):
                logger.warning(f"Hidden content detected: {url}")
                return True
            
            # 3. 重复内容检测
            if self._check_duplicate_content(content):
                logger.warning(f"Duplicate content detected: {url}")
                return True
            
            # 4. 垃圾外链检测
            spam_links = self._detect_spam_links(item.get("links", []))
            if spam_links:
                logger.warning(f"Spam links detected: {url}")
                return True
            
            # 5. 作弊行为模式
            if self._detect_spam_patterns(item):
                logger.warning(f"Spam patterns detected: {url}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in spam detection: {str(e)}")
            return False
            
    def evaluate_content_quality(self, item: Dict[str, Any]) -> float:
        """绿萝算法：评估内容质量"""
        try:
            content = item.get("content", "")
            if not content:
                return 0.0
                
            score = 0.0
            
            # 1. 原创性评分
            originality = self._calculate_originality(content)
            score += originality * 0.3
            
            # 2. 结构完整性
            structure_score = self._evaluate_structure(item)
            score += structure_score * 0.2
            
            # 3. 专业性评分
            expertise_score = self._evaluate_expertise(content)
            score += expertise_score * 0.2
            
            # 4. 可读性评分
            readability = self._calculate_readability(content)
            score += readability * 0.15
            
            # 5. 多媒体丰富度
            media_score = self._evaluate_media_richness(item)
            score += media_score * 0.15
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.error(f"Error in content quality evaluation: {str(e)}")
            return 0.5
            
    def evaluate_mobile_experience(self, item: Dict[str, Any]) -> float:
        """飓风算法：评估移动端体验"""
        try:
            score = 0.0
            
            # 1. 移动适配检查
            if item.get("mobile_friendly", False):
                score += 0.3
                
            # 2. 页面加载速度
            load_time = float(item.get("load_time", 5000))  # 默认5000ms
            if load_time < 1000:
                score += 0.3
            elif load_time < 2000:
                score += 0.2
            elif load_time < 3000:
                score += 0.1
                
            # 3. 资源优化
            if self._check_resource_optimization(item):
                score += 0.2
                
            # 4. 交互友好度
            if self._check_interaction_friendly(item):
                score += 0.1
                
            # 5. 视口配置
            if item.get("viewport_configured", False):
                score += 0.1
                
            return score
            
        except Exception as e:
            logger.error(f"Error in mobile experience evaluation: {str(e)}")
            return 0.5
            
    def evaluate_freshness(self, item: Dict[str, Any]) -> float:
        """石榴算法：评估内容时效性"""
        try:
            # 1. 发布时间评分
            pub_time = datetime.fromisoformat(
                item.get("published_at", datetime.now().isoformat())
            )
            age_days = (datetime.now() - pub_time).days
            
            if age_days <= 1:  # 24小时内
                time_score = 1.0
            elif age_days <= 7:  # 一周内
                time_score = 0.8
            elif age_days <= 30:  # 一月内
                time_score = 0.6
            elif age_days <= 90:  # 三月内
                time_score = 0.4
            else:
                time_score = 0.2
                
            # 2. 更新频率评分
            update_score = self._calculate_update_frequency(item)
            
            # 3. 实时性评分
            realtime_score = self._evaluate_realtime_nature(item)
            
            # 4. 时效性内容识别
            temporal_score = self._detect_temporal_content(item)
            
            # 加权平均
            weights = self.config["freshness"]["weights"]
            final_score = (
                time_score * weights["time"] +
                update_score * weights["update"] +
                realtime_score * weights["realtime"] +
                temporal_score * weights["temporal"]
            )
            
            return final_score
            
        except Exception as e:
            logger.error(f"Error in freshness evaluation: {str(e)}")
            return 0.5
            
    def evaluate_credibility(self, item: Dict[str, Any]) -> float:
        """清风算法：评估信息可信度"""
        try:
            score = 0.0
            
            # 1. 域名可信度
            domain_score = self._evaluate_domain_credibility(item["url"])
            score += domain_score * 0.3
            
            # 2. 内容真实性
            truth_score = self._evaluate_content_truth(item)
            score += truth_score * 0.3
            
            # 3. 引用来源评分
            reference_score = self._evaluate_references(item)
            score += reference_score * 0.2
            
            # 4. 情感倾向分析
            sentiment_score = self._analyze_sentiment(item)
            score += sentiment_score * 0.1
            
            # 5. 用户反馈
            feedback_score = self._get_user_feedback(item["url"])
            score += feedback_score * 0.1
            
            return score
            
        except Exception as e:
            logger.error(f"Error in credibility evaluation: {str(e)}")
            return 0.5
            
    def _detect_hidden_content(self, item: Dict[str, Any]) -> bool:
        """检测隐藏内容"""
        html = item.get("html", "")
        if not html:
            return False
            
        # 检查CSS是否有隐藏内容
        hidden_patterns = [
            r"display:\s*none",
            r"visibility:\s*hidden",
            r"opacity:\s*0",
            r"height:\s*0",
            r"position:\s*absolute;\s*left:\s*-\d+px"
        ]
        
        for pattern in hidden_patterns:
            if re.search(pattern, html, re.I):
                return True
                
        return False
        
    def _check_duplicate_content(self, content: str) -> bool:
        """检查重复内容"""
        # 计算内容指纹
        content_hash = hash(content)
        
        # 检查Redis中是否存在
        exists = self.redis.sismember("content_hashes", content_hash)
        if not exists:
            self.redis.sadd("content_hashes", content_hash)
            
        return exists
        
    def _detect_spam_links(self, links: List[str]) -> bool:
        """检测垃圾外链"""
        spam_domains = self.redis.smembers("spam_domains")
        for link in links:
            domain = urlparse(link).netloc
            if domain in spam_domains:
                return True
        return False
        
    def _detect_spam_patterns(self, item: Dict[str, Any]) -> bool:
        """检测作弊行为模式"""
        # 1. 检查链接密度
        content_length = len(item.get("content", ""))
        links_count = len(item.get("links", []))
        if content_length > 0:
            link_density = links_count / content_length
            if link_density > self.config["spam"]["link_density_threshold"]:
                return True
                
        # 2. 检查关键词密度
        keywords = item.get("keywords", [])
        if len(keywords) > self.config["spam"]["max_keywords"]:
            return True
            
        return False
        
    def _calculate_originality(self, content: str) -> float:
        """计算内容原创性"""
        # 使用simhash等算法计算相似度
        return 0.8  # 示例分数
        
    def _evaluate_structure(self, item: Dict[str, Any]) -> float:
        """评估内容结构"""
        score = 0.0
        
        # 检查标题
        if item.get("title"):
            score += 0.2
            
        # 检查描述
        if item.get("meta_description"):
            score += 0.2
            
        # 检查段落结构
        if self._has_good_paragraphs(item.get("content", "")):
            score += 0.2
            
        # 检查小标题
        if item.get("subheadings"):
            score += 0.2
            
        # 检查格式化
        if self._has_good_formatting(item.get("content", "")):
            score += 0.2
            
        return score
        
    def _evaluate_expertise(self, content: str) -> float:
        """评估专业性"""
        # 使用专业词库进行匹配
        return 0.7  # 示例分数
        
    def _calculate_readability(self, content: str) -> float:
        """计算可读性分数"""
        try:
            blob = TextBlob(content)
            # 使用Flesch Reading Ease等算法
            return 0.8  # 示例分数
        except:
            return 0.5
            
    def _evaluate_media_richness(self, item: Dict[str, Any]) -> float:
        """评估多媒体丰富度"""
        score = 0.0
        
        # 图片评分
        images = item.get("images", [])
        score += min(len(images) * 0.2, 0.4)
        
        # 视频评分
        videos = item.get("videos", [])
        score += min(len(videos) * 0.3, 0.4)
        
        # 其他多媒体
        other_media = item.get("other_media", [])
        score += min(len(other_media) * 0.1, 0.2)
        
        return score
        
    def _check_resource_optimization(self, item: Dict[str, Any]) -> bool:
        """检查资源优化"""
        # 检查图片是否优化
        images = item.get("images", [])
        for image in images:
            if image.get("size", 0) > self.config["mobile"]["max_image_size"]:
                return False
                
        # 检查JS/CSS是否优化
        if item.get("js_size", 0) > self.config["mobile"]["max_js_size"]:
            return False
            
        if item.get("css_size", 0) > self.config["mobile"]["max_css_size"]:
            return False
            
        return True
        
    def _check_interaction_friendly(self, item: Dict[str, Any]) -> bool:
        """检查交互友好度"""
        # 检查点击区域大小
        if not item.get("touch_elements_optimized", False):
            return False
            
        # 检查字体大小
        if not item.get("font_size_optimized", False):
            return False
            
        return True
        
    def _calculate_update_frequency(self, item: Dict[str, Any]) -> float:
        """计算更新频率分数"""
        try:
            url = item["url"]
            updates = self.redis.lrange(f"updates:{url}", 0, -1)
            if not updates:
                return 0.5
                
            # 计算平均更新间隔
            update_times = [datetime.fromisoformat(t) for t in updates]
            intervals = []
            for i in range(len(update_times)-1):
                interval = (update_times[i+1] - update_times[i]).total_seconds()
                intervals.append(interval)
                
            if not intervals:
                return 0.5
                
            avg_interval = sum(intervals) / len(intervals)
            
            # 评分规则
            if avg_interval < 3600:  # 1小时内
                return 1.0
            elif avg_interval < 86400:  # 24小时内
                return 0.8
            elif avg_interval < 604800:  # 一周内
                return 0.6
            else:
                return 0.4
                
        except Exception as e:
            logger.error(f"Error calculating update frequency: {str(e)}")
            return 0.5
            
    def _evaluate_realtime_nature(self, item: Dict[str, Any]) -> float:
        """评估实时性"""
        # 检查是否包含实时信息
        content = item.get("content", "")
        realtime_patterns = [
            r"实时",
            r"直播",
            r"最新",
            r"突发",
            r"刚刚",
            r"小时前"
        ]
        
        score = 0.0
        for pattern in realtime_patterns:
            if re.search(pattern, content):
                score += 0.2
                
        return min(score, 1.0)
        
    def _detect_temporal_content(self, item: Dict[str, Any]) -> float:
        """检测时效性内容"""
        content = item.get("content", "")
        
        # 时效性关键词
        temporal_patterns = [
            r"\d{4}年\d{1,2}月\d{1,2}日",
            r"今天",
            r"昨天",
            r"上周",
            r"本月",
            r"近期",
            r"预计",
            r"预测"
        ]
        
        score = 0.0
        for pattern in temporal_patterns:
            if re.search(pattern, content):
                score += 0.15
                
        return min(score, 1.0)
        
    def _evaluate_domain_credibility(self, url: str) -> float:
        """评估域名可信度"""
        domain = urlparse(url).netloc
        
        # 1. 检查是否是可信域名
        if self.redis.sismember("trusted_domains", domain):
            return 1.0
            
        # 2. 检查域名年龄
        domain_info = self.redis.hgetall(f"domain:{domain}")
        age_years = float(domain_info.get("age_years", 0))
        
        # 3. 检查域名评分
        domain_score = float(domain_info.get("credibility_score", 0.5))
        
        return (min(age_years/10, 1) + domain_score) / 2
        
    def _evaluate_content_truth(self, item: Dict[str, Any]) -> float:
        """评估内容真实性"""
        content = item.get("content", "")
        
        # 1. 检查谣言特征
        if self._check_rumor_patterns(content):
            return 0.0
            
        # 2. 检查内容一致性
        consistency_score = self._check_content_consistency(item)
        
        # 3. 检查信息源可靠性
        source_score = self._check_source_reliability(item)
        
        return (consistency_score + source_score) / 2
        
    def _evaluate_references(self, item: Dict[str, Any]) -> float:
        """评估引用来源"""
        references = item.get("references", [])
        if not references:
            return 0.3
            
        score = 0.0
        for ref in references:
            # 检查引用域名可信度
            domain = urlparse(ref).netloc
            if self.redis.sismember("trusted_domains", domain):
                score += 0.2
                
        return min(score, 1.0)
        
    def _analyze_sentiment(self, item: Dict[str, Any]) -> float:
        """分析情感倾向"""
        content = item.get("content", "")
        try:
            blob = TextBlob(content)
            # 将情感极性转换为0-1分数
            polarity = (blob.sentiment.polarity + 1) / 2
            return polarity
        except:
            return 0.5
            
    def _get_user_feedback(self, url: str) -> float:
        """获取用户反馈分数"""
        try:
            feedback = self.redis.hgetall(f"feedback:{url}")
            if not feedback:
                return 0.5
                
            # 计算好评率
            positive = int(feedback.get("positive", 0))
            negative = int(feedback.get("negative", 0))
            total = positive + negative
            
            if total == 0:
                return 0.5
                
            return positive / total
            
        except:
            return 0.5
            
    def _store_scores(self, item: Dict[str, Any]):
        """存储评分结果"""
        url = item["url"]
        scores = {
            "spam_detected": int(item.get("is_spam", False)),
            "content_quality": item.get("content_quality_score", 0),
            "mobile_score": item.get("mobile_score", 0),
            "freshness_score": item.get("freshness_score", 0),
            "credibility_score": item.get("credibility_score", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        self.redis.hmset(f"baidu_scores:{url}", scores)
        logger.info(f"Stored Baidu algorithm scores for: {url}") 