#!/usr/bin/env python3
import logging
import random
from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app import crud, schemas
from app.db.session import SessionLocal
from app.models.user import User
from app.models.knowledge_base import KnowledgeBase
from app.models.paper import Paper
from app.models.tag import Tag

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 示例用户数据
users_data = [
    {
        "username": "用户1",
        "email": "user1@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=1",
        "location": "北京",
        "experience": "互联网从业者，热爱技术分享",
        "gender": "男",
        "age": 28,
        "school": "北京大学"
    },
    {
        "username": "用户2",
        "email": "user2@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=2",
        "location": "上海",
        "experience": "人工智能研究员，专注机器学习",
        "gender": "女",
        "age": 26,
        "school": "复旦大学"
    },
    {
        "username": "用户3",
        "email": "user3@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=3",
        "location": "广州",
        "experience": "数据科学家，擅长数据分析与可视化",
        "gender": "男",
        "age": 30,
        "school": "中山大学"
    },
    {
        "username": "用户4",
        "email": "user4@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=4",
        "location": "深圳",
        "experience": "全栈开发工程师，喜欢探索新技术",
        "gender": "男",
        "age": 25,
        "school": "华南理工大学"
    },
    {
        "username": "用户5",
        "email": "user5@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=5",
        "location": "杭州",
        "experience": "产品经理，关注用户体验与产品设计",
        "gender": "女",
        "age": 27,
        "school": "浙江大学"
    },
    {
        "username": "用户6",
        "email": "user6@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=6",
        "location": "成都",
        "experience": "自然语言处理研究员，专注大语言模型",
        "gender": "男",
        "age": 29,
        "school": "电子科技大学"
    },
    {
        "username": "用户7",
        "email": "user7@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=7",
        "location": "武汉",
        "experience": "计算机视觉工程师，专注目标检测算法",
        "gender": "女",
        "age": 31,
        "school": "华中科技大学"
    },
    {
        "username": "用户8",
        "email": "user8@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=8",
        "location": "南京",
        "experience": "算法工程师，专注推荐系统",
        "gender": "男",
        "age": 26,
        "school": "东南大学"
    },
    {
        "username": "用户9",
        "email": "user9@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=9",
        "location": "西安",
        "experience": "数据工程师，擅长大数据处理",
        "gender": "男",
        "age": 28,
        "school": "西安交通大学"
    },
    {
        "username": "用户10",
        "email": "user10@example.com",
        "password": "password123",
        "avatar": "https://api.dicebear.com/7.x/avataaars/svg?seed=10",
        "location": "天津",
        "experience": "机器学习工程师，热爱开源项目",
        "gender": "女",
        "age": 27,
        "school": "南开大学"
    },
]

# 示例标签数据
tags_data = [
    "人工智能", "机器学习", "深度学习", "自然语言处理", "计算机视觉",
    "数据挖掘", "大数据", "推荐系统", "强化学习", "神经网络",
    "知识图谱", "语音识别", "图像处理", "生成对抗网络", "注意力机制",
    "迁移学习", "元学习", "联邦学习", "图神经网络", "自动机器学习",
    "情感分析", "对话系统", "文本摘要", "语义分割", "姿态估计",
    "人脸识别", "目标检测", "实例分割", "行为识别", "异常检测",
    "多模态学习", "时序预测", "因果推断", "贝叶斯网络", "随机森林",
    "支持向量机", "决策树", "XGBoost", "聚类算法", "降维技术"
]

# 示例知识库数据
knowledge_bases_data = [
    {
        "title": "人工智能伦理研究",
        "description": "关于人工智能伦理问题的研究集合，包括隐私保护、公平性、透明度等方面",
        "tags": ["人工智能", "伦理", "隐私保护"]
    },
    {
        "title": "机器学习算法优化",
        "description": "各种机器学习算法的优化技巧和最佳实践，包括梯度下降、正则化等方法",
        "tags": ["机器学习", "算法优化", "梯度下降", "正则化"]
    },
    {
        "title": "大语言模型知识整合",
        "description": "大型语言模型（LLM）的综合研究，包含架构、训练方法、应用场景和最新成果",
        "tags": ["自然语言处理", "大语言模型", "Transformer", "预训练"]
    },
    {
        "title": "计算机视觉研究集锦",
        "description": "计算机视觉领域的前沿研究和实践案例，包含目标检测、图像分割、视频分析等方向",
        "tags": ["计算机视觉", "目标检测", "图像分割", "视频分析"]
    },
    {
        "title": "强化学习理论与实践",
        "description": "强化学习算法的理论基础、最新研究方向和实际应用案例的综合集合",
        "tags": ["强化学习", "马尔可夫决策过程", "Q-learning", "策略梯度"]
    },
    {
        "title": "知识图谱构建与应用",
        "description": "知识图谱的构建方法、存储技术、查询优化及实际应用案例",
        "tags": ["知识图谱", "语义网", "图数据库", "关系抽取"]
    },
    {
        "title": "生成式人工智能研究",
        "description": "生成式AI模型的研究进展，包括GAN、扩散模型、大型语言模型等",
        "tags": ["生成对抗网络", "扩散模型", "生成式AI", "文本生成"]
    },
    {
        "title": "图神经网络前沿研究",
        "description": "图神经网络的最新研究进展，包括图卷积网络、图注意力网络等",
        "tags": ["图神经网络", "图卷积", "图注意力", "节点表示学习"]
    },
    {
        "title": "多模态学习最新进展",
        "description": "多模态学习领域的研究进展，包括跨模态表示、对齐技术和融合方法等",
        "tags": ["多模态学习", "跨模态表示", "视觉语言模型", "多模态融合"]
    },
    {
        "title": "联邦学习隐私保护",
        "description": "联邦学习中的隐私保护机制研究，包括差分隐私、安全聚合和加密计算等技术",
        "tags": ["联邦学习", "隐私保护", "差分隐私", "加密计算"]
    },
    {
        "title": "自动驾驶感知系统",
        "description": "自动驾驶中的感知系统研究，包括障碍物检测、道路分割、交通标志识别等",
        "tags": ["自动驾驶", "计算机视觉", "目标检测", "语义分割"]
    },
    {
        "title": "医疗影像AI诊断",
        "description": "AI在医疗影像诊断中的应用研究，包括CT、MRI、X光等多种模态",
        "tags": ["医疗AI", "计算机视觉", "医学图像分析", "辅助诊断"]
    },
    {
        "title": "情感计算与用户体验",
        "description": "情感计算在用户体验优化中的应用研究，包括情感识别、情感生成等方向",
        "tags": ["情感计算", "用户体验", "情感识别", "人机交互"]
    },
    {
        "title": "因果推断与可解释AI",
        "description": "因果推断方法在可解释AI中的应用研究，建立可解释的机器学习模型",
        "tags": ["因果推断", "可解释AI", "因果发现", "反事实解释"]
    },
    {
        "title": "时间序列预测方法",
        "description": "时间序列数据的预测方法研究，包括传统统计方法和深度学习方法的比较",
        "tags": ["时间序列", "预测分析", "LSTM", "Transformer"]
    }
]

# 示例论文数据模板
paper_templates = [
    {
        "title": "{主题}领域中{技术}的应用研究",
        "authors": ["作者A", "作者B", "作者C"],
        "abstract": "本文研究了{主题}领域中{技术}的应用。我们提出了一种新的方法来解决{问题}，并通过实验证明了其有效性。实验结果表明，我们的方法在{指标}上比现有方法提高了{数字}%。",
        "publish_date": "{date}",
        "doi": "10.1234/example.{year}.{number}",
        "url": "https://example.com/papers/{id}"
    },
    {
        "title": "{技术}：一种解决{主题}问题的新方法",
        "authors": ["作者X", "作者Y"],
        "abstract": "我们提出了一种名为{技术}的新方法，用于解决{主题}中的{问题}问题。该方法基于{基础理论}，并引入了{创新点}来提高性能。在{数据集}上的实验表明，该方法在{指标}方面优于现有方法。",
        "publish_date": "{date}",
        "doi": "10.5678/method.{year}.{number}",
        "url": "https://example.org/papers/{id}"
    },
    {
        "title": "{主题}的综述：从{技术A}到{技术B}",
        "authors": ["作者M", "作者N", "作者O", "作者P"],
        "abstract": "本文对{主题}领域进行了全面的综述，涵盖了从传统的{技术A}到最新的{技术B}的发展历程。我们分析了不同方法的优缺点，并讨论了该领域的未来研究方向。此外，我们还总结了{主题}在实际应用中面临的挑战。",
        "publish_date": "{date}",
        "doi": "10.9012/survey.{year}.{number}",
        "url": "https://example.net/papers/{id}"
    },
    {
        "title": "基于{技术}的{主题}{问题}解决方案",
        "authors": ["作者E", "作者F", "作者G"],
        "abstract": "针对{主题}领域中的{问题}，本文提出了一种基于{技术}的解决方案。我们设计了新的{创新点}模块，并在{数据集}上验证了其性能。结果表明，我们的方法在{指标}指标上取得了显著提升，为{主题}领域带来了新的研究视角。",
        "publish_date": "{date}",
        "doi": "10.3456/solution.{year}.{number}",
        "url": "https://example.com/solutions/{id}"
    },
    {
        "title": "{主题}中的{技术}：挑战与机遇",
        "authors": ["作者R", "作者S", "作者T"],
        "abstract": "本文探讨了在{主题}领域应用{技术}的当前挑战与未来机遇。我们分析了现有{问题}的解决进展，并提出了一个新的研究框架。通过在{数据集}上的实验验证，我们证明了该框架能够有效提升{指标}，为后续研究指明了方向。",
        "publish_date": "{date}",
        "doi": "10.7890/challenges.{year}.{number}",
        "url": "https://example.org/challenges/{id}"
    }
]

# 技术关键词
technologies = [
    "深度学习", "卷积神经网络", "循环神经网络", "Transformer", "注意力机制",
    "BERT", "GPT", "扩散模型", "强化学习", "联邦学习", "图神经网络",
    "知识蒸馏", "自监督学习", "元学习", "迁移学习", "对比学习",
    "深度强化学习", "多模态学习", "持续学习", "图注意力网络",
    "VIT", "CLIP", "DALL-E", "Stable Diffusion", "AlphaFold",
    "NeRF", "DETR", "Swin Transformer", "MaskRCNN", "YOLO",
    "Llama", "Claude", "BART", "T5", "XLNet",
    "EfficientNet", "ResNet", "DenseNet", "U-Net", "FasterRCNN"
]

# 主题关键词
topics = [
    "自然语言处理", "计算机视觉", "推荐系统", "语音识别", "医疗诊断",
    "金融预测", "气候建模", "自动驾驶", "机器翻译", "情感分析",
    "文本摘要", "图像分割", "目标检测", "人脸识别", "姿态估计",
    "视频理解", "问答系统", "信息检索", "知识图谱", "异常检测",
    "生物信息学", "材料科学", "量子计算", "机器人控制", "智能家居",
    "智慧农业", "智慧城市", "工业4.0", "可穿戴设备", "增强现实"
]

# 问题关键词
problems = [
    "数据稀疏性", "过拟合", "泛化能力", "计算效率", "可解释性",
    "鲁棒性", "偏见", "不平衡数据", "标签噪声", "领域适应",
    "少样本学习", "数据隐私", "模型压缩", "实时推理", "多任务学习",
    "类别不平衡", "长尾分布", "模型蒸馏", "知识融合", "特征表示",
    "灾难性遗忘", "梯度消失", "梯度爆炸", "样本效率", "噪声鲁棒性",
    "对抗攻击", "分布偏移", "模型校准", "推理延迟", "能源效率"
]

# 指标关键词
metrics = [
    "准确率", "精确率", "召回率", "F1分数", "AUC", "BLEU分数",
    "ROUGE分数", "平均精度", "均方误差", "交叉熵损失",
    "计算时间", "参数数量", "推理速度", "内存占用", "模型大小",
    "MAP", "NDCG", "METEOR", "CIDEr", "IoU",
    "Dice系数", "PSNR", "SSIM", "困惑度", "ROC曲线",
    "精度-召回曲线", "FLOPs", "吞吐量", "能耗效率", "校准误差"
]

# 数据集关键词
datasets = [
    "ImageNet", "COCO", "MNIST", "CIFAR-10", "MS MARCO",
    "SQuAD", "WikiText", "GLUE", "WMT", "LibriSpeech",
    "UCI机器学习库", "Amazon评论数据集", "Yelp评论数据集", "MovieLens", "Netflix Prize",
    "VoxCeleb", "AudioSet", "KITTI", "Pascal VOC", "CelebA",
    "Kinetics", "AVA", "NYU Depth", "YouTube-8M", "CLEVR",
    "LFW", "Open Images", "QuAC", "HotpotQA", "CoQA",
    "Flickr30k", "VQA", "WebVision", "Places", "ShapeNet",
    "ModelNet", "Reddit评论数据集", "MIMIC-III", "ChestX-ray14", "Stanford Cars"
]

# 基础理论关键词
theories = [
    "贝叶斯理论", "信息论", "概率图模型", "最优控制", "统计学习理论",
    "博弈论", "决策理论", "凸优化", "稀疏表示", "流形学习",
    "核方法", "时间序列分析", "马尔可夫过程", "压缩感知", "谱聚类",
    "变分推断", "最大似然估计", "随机过程", "测度论", "函数分析",
    "最大熵原理", "最小描述长度", "结构风险最小化", "PAC学习", "VC维理论",
    "经验风险最小化", "正则化理论", "对偶理论", "随机梯度下降", "反向传播"
]

# 创新点关键词
innovations = [
    "多头注意力机制", "残差连接", "门控机制", "自适应学习率", "对抗训练",
    "梯度裁剪", "批量归一化", "dropout", "层归一化", "权重共享",
    "跳跃连接", "密集连接", "胶囊网络", "注意力引导", "多任务学习架构",
    "混合专家模型", "条件计算", "知识蒸馏", "稀疏激活", "神经架构搜索",
    "自注意力", "特征金字塔", "对比损失", "软标签", "课程学习",
    "混合精度训练", "量化感知训练", "渐进式训练", "多尺度处理", "动态卷积",
    "可变形卷积", "协同过滤", "显式反馈", "隐式反馈", "图注意力网络"
]

def seed_db() -> None:
    """初始化数据库并插入测试数据"""
    db = SessionLocal()
    try:
        # 创建用户
        created_users = []
        for user_data in users_data:
            user_in = schemas.UserCreate(
                username=user_data["username"],
                email=user_data["email"],
                password=user_data["password"],
                avatar=user_data.get("avatar"),
                location=user_data.get("location"),
                experience=user_data.get("experience"),
                gender=user_data.get("gender"),
                age=user_data.get("age"),
                school=user_data.get("school"),
            )
            user = crud.user.get_by_email(db, email=user_in.email)
            if not user:
                user = crud.user.create(db, obj_in=user_in)
                logger.info(f"创建用户: {user.username}")
            created_users.append(user)
        
        # 创建标签
        created_tags = []
        for tag_name in tags_data:
            tag = crud.tag.get_by_name(db, name=tag_name)
            if not tag:
                tag_in = schemas.TagCreate(name=tag_name)
                tag = crud.tag.create(db, obj_in=tag_in)
                logger.info(f"创建标签: {tag.name}")
            created_tags.append(tag)
        
        # 关注关系 - 扩展关注关系以包含更多用户
        follow_pairs = [
            (0, 1), (0, 2), (0, 5), (0, 8),
            (1, 0), (1, 3), (1, 6), (1, 9),
            (2, 0), (2, 4), (2, 7), (2, 9),
            (3, 1), (3, 4), (3, 6), (3, 8),
            (4, 0), (4, 2), (4, 7), (4, 9),
            (5, 0), (5, 3), (5, 7), (5, 9),
            (6, 1), (6, 2), (6, 4), (6, 8),
            (7, 0), (7, 3), (7, 5), (7, 9),
            (8, 1), (8, 2), (8, 4), (8, 6),
            (9, 0), (9, 3), (9, 5), (9, 7)
        ]
        for follower_idx, followed_idx in follow_pairs:
            follower = created_users[follower_idx]
            followed = created_users[followed_idx]
            if followed.id not in [u.id for u in follower.following]:
                follower.following.append(followed)
                db.commit()
                logger.info(f"创建关注关系: {follower.username} -> {followed.username}")
        
        # 创建知识库
        created_kbs = []
        for idx, kb_data in enumerate(knowledge_bases_data):
            # 分配给不同用户
            user = created_users[idx % len(created_users)]
            kb_in = schemas.KnowledgeBaseCreate(
                title=kb_data["title"],
                description=kb_data["description"],
                user_id=user.id,
                tags=kb_data["tags"]
            )
            kb = crud.knowledge_base.get_by_title(db, title=kb_in.title)
            if not kb:
                kb = crud.knowledge_base.create_with_owner(
                    db, obj_in=kb_in, owner_id=user.id
                )
                logger.info(f"创建知识库: {kb.title}")
            created_kbs.append(kb)
        
        # 填充论文数据到知识库
        paper_id = 1
        for kb_idx, kb in enumerate(created_kbs):
            # 每个知识库3-10个论文
            papers_count = random.randint(3, 10)
            for i in range(papers_count):
                # 选择一个模板
                template = random.choice(paper_templates)
                
                # 根据知识库主题，选择相关的技术和主题
                kb_tags = [t.name for t in kb.tags]
                main_topic = random.choice(kb_tags if kb_tags else topics)
                
                # 生成随机日期（过去3年内）
                days_ago = random.randint(0, 365 * 3)
                random_date = datetime.now() - timedelta(days=days_ago)
                date_str = random_date.strftime("%Y-%m-%d")
                year = random_date.year
                
                # 生成2-4个作者名
                authors_count = random.randint(2, 4)
                first_names = ["张", "王", "李", "赵", "刘", "陈", "杨", "黄", "周", "吴", 
                               "郑", "孙", "马", "朱", "林", "胡", "高", "何", "罗", "郭",
                               "戴", "宋", "邓", "许", "傅", "沈", "曾", "彭", "吕", "苏"]
                last_names = ["伟", "芳", "娜", "秀英", "敏", "静", "强", "磊", "洋", "艳", 
                              "勇", "军", "杰", "娟", "涛", "明", "超", "霞", "平", "刚",
                              "博", "琳", "欣", "宇", "建华", "晶", "文", "婷", "玉", "萍"]
                authors = []
                for _ in range(authors_count):
                    author_name = random.choice(first_names) + random.choice(last_names)
                    authors.append(author_name)
                
                # 组装标题和摘要
                tech = random.choice(technologies)
                topic = main_topic
                problem = random.choice(problems)
                metric = random.choice(metrics)
                improvement = random.randint(5, 40)
                theory = random.choice(theories)
                innovation = random.choice(innovations)
                
                # 为综述模板准备技术A和技术B
                tech_a = random.choice(technologies[:10])  # 从前10个技术中随机选择作为"技术A"
                tech_b = random.choice(technologies[10:])  # 从后10个技术中随机选择作为"技术B"
                
                # 确保数据集变量存在
                dataset = random.choice(datasets)
                
                # 根据模板生成论文数据
                title = template["title"].format(
                    主题=topic, 
                    技术=tech, 
                    技术A=tech_a, 
                    技术B=tech_b,
                    问题=problem
                )
                
                # 根据不同模板，准备不同的替换变量
                if "主题" in template["abstract"] and "技术A" in template["abstract"]:
                    abstract = template["abstract"].format(
                        主题=topic,
                        技术=tech,
                        技术A=tech_a,
                        技术B=tech_b
                    )
                else:
                    abstract = template["abstract"].format(
                        主题=topic,
                        技术=tech,
                        问题=problem,
                        指标=metric,
                        数字=improvement,
                        基础理论=theory,
                        创新点=innovation,
                        数据集=dataset
                    )
                
                doi = template["doi"].format(year=year, number=1000+paper_id)
                url = template["url"].format(id=paper_id)
                
                # 创建论文
                paper_in = schemas.PaperCreate(
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    publish_date=date_str,
                    doi=doi,
                    url=url,
                    knowledge_base_id=kb.id
                )
                
                # 检查论文是否已存在
                existing_paper = db.query(Paper).filter(
                    Paper.title == title,
                    Paper.knowledge_base_id == kb.id
                ).first()
                
                if not existing_paper:
                    paper = crud.paper.create_with_knowledge_base(
                        db, obj_in=paper_in, knowledge_base_id=kb.id
                    )
                    paper_id += 1
                    logger.info(f"创建论文: {paper.title}")
        
        # 用户点赞知识库
        likes_count = 0
        for user in created_users:
            # 每个用户点赞3-8个知识库
            like_count = random.randint(3, 8)
            # 随机选择知识库点赞
            kbs_to_like = random.sample(created_kbs, min(like_count, len(created_kbs)))
            for kb in kbs_to_like:
                if crud.knowledge_base.like(db, kb_id=kb.id, user_id=user.id):
                    likes_count += 1
        logger.info(f"创建知识库点赞: {likes_count}个")
        
        # 添加一些消息
        messages_count = 0
        for i in range(40):  # 增加到40条消息
            # 随机选择发送者和接收者
            sender_idx = random.randint(0, len(created_users) - 1)
            receiver_idx = random.randint(0, len(created_users) - 1)
            while receiver_idx == sender_idx:
                receiver_idx = random.randint(0, len(created_users) - 1)
            
            sender = created_users[sender_idx]
            receiver = created_users[receiver_idx]
            
            # 生成随机消息内容
            greetings = ["你好", "嗨", "早上好", "下午好", "晚上好", "问个问题", "请教一下", "想讨论一下"]
            questions = [
                "最近在研究什么课题？", "有什么好的论文推荐吗？", 
                "对机器学习有了解吗？", "能分享一下你的知识库吗？", 
                "我想请教一个问题", "最近有什么新发现？",
                "对这篇论文有什么看法？", "这个研究方向怎么样？",
                "能给我一些学习建议吗？", "我们可以合作研究吗？",
                "你对大语言模型有什么见解？", "能推荐几个数据集吗？",
                "你觉得这个算法如何？", "有什么好的开源项目推荐？",
                "最近有什么好的会议吗？", "对这个研究领域怎么看？"
            ]
            
            content = f"{random.choice(greetings)}，{receiver.username}！{random.choice(questions)}"
            
            # 创建消息
            message_in = schemas.MessageCreate(
                content=content,
                receiver_id=receiver.id
            )
            message = crud.message.create_with_sender(
                db, obj_in=message_in, sender_id=sender.id
            )
            messages_count += 1
        
        logger.info(f"创建消息: {messages_count}条")
        
        # 标记一些消息为已读
        read_count = crud.message.mark_as_read(db, user_id1=created_users[0].id, user_id2=created_users[1].id)
        read_count += crud.message.mark_as_read(db, user_id1=created_users[2].id, user_id2=created_users[3].id)
        read_count += crud.message.mark_as_read(db, user_id1=created_users[4].id, user_id2=created_users[5].id)
        logger.info(f"标记已读消息: {read_count}条")
    
    finally:
        db.close()


def main() -> None:
    logger.info("创建示例数据")
    seed_db()
    logger.info("示例数据创建完成")


if __name__ == "__main__":
    main() 