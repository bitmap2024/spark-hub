// 帖子类型定义
export interface Post {
  id: string;
  title: string;
  content: string;
  author: {
    id: string;
    username: string;
    avatar: string;
  };
  createdAt: string;
  likes: number;
  comments: Comment[];
}

// 评论类型定义
export interface Comment {
  id: string;
  content: string;
  author: {
    id: string;
    username: string;
    avatar: string;
  };
  createdAt: string;
  likes: number;
}

// 模拟数据
export const DUMMY_POSTS: Post[] = [
  {
    id: "1",
    title: "大家好，我是新来的，请多关照！",
    content: "刚刚加入这个平台，希望能认识更多朋友。我喜欢旅行、摄影和美食，欢迎大家来交流！",
    author: {
      id: "user1",
      username: "旅行达人",
      avatar: "",
    },
    createdAt: "2小时前",
    likes: 24,
    comments: [
      {
        id: "c1",
        content: "欢迎欢迎！有什么问题随时问",
        author: {
          id: "user2",
          username: "老用户",
          avatar: "",
        },
        createdAt: "1小时前",
        likes: 5,
      },
      {
        id: "c2",
        content: "我也喜欢旅行，可以分享一些好玩的地方",
        author: {
          id: "user3",
          username: "摄影爱好者",
          avatar: "",
        },
        createdAt: "30分钟前",
        likes: 3,
      }
    ]
  },
  {
    id: "2",
    title: "推荐一款超好用的摄影APP",
    content: "最近发现了一款超好用的修图软件，效果非常棒，而且操作简单。大家有兴趣可以试试，叫「光影魔术师」，支持各种高级调色和一键美化功能。",
    author: {
      id: "user4",
      username: "科技达人",
      avatar: "",
    },
    createdAt: "5小时前",
    likes: 56,
    comments: [
      {
        id: "c3",
        content: "我也在用这个，确实不错！",
        author: {
          id: "user5",
          username: "图片编辑",
          avatar: "",
        },
        createdAt: "4小时前",
        likes: 8,
      }
    ]
  },
  {
    id: "3",
    title: "分享一个美食攻略",
    content: "上周去了成都，吃了很多好吃的，强烈推荐这几家店：1. 华兴街的担担面 2. 宽窄巷子的钵钵鸡 3. 春熙路的麻辣香锅。味道都非常正宗，价格也实惠！",
    author: {
      id: "user6",
      username: "美食家",
      avatar: "",
    },
    createdAt: "1天前",
    likes: 120,
    comments: [
      {
        id: "c4",
        content: "记下了，下次去成都一定去尝尝",
        author: {
          id: "user7",
          username: "吃货一枚",
          avatar: "",
        },
        createdAt: "20小时前",
        likes: 15,
      },
      {
        id: "c5",
        content: "华兴街的担担面是真的好吃，强烈推荐！",
        author: {
          id: "user8",
          username: "本地吃货",
          avatar: "",
        },
        createdAt: "18小时前",
        likes: 12,
      }
    ]
  }
];

// 获取存储的帖子
export const getStoredPosts = (): Post[] => {
  try {
    const saved = localStorage.getItem('community-posts');
    return saved ? JSON.parse(saved) : DUMMY_POSTS;
  } catch (error) {
    console.error('Error loading posts from localStorage', error);
    return DUMMY_POSTS;
  }
};

// 保存帖子到本地存储
export const savePosts = (posts: Post[]): void => {
  try {
    localStorage.setItem('community-posts', JSON.stringify(posts));
  } catch (error) {
    console.error('Error saving posts to localStorage', error);
  }
};

// 添加新帖子
export const addPost = (newPost: Post): void => {
  const currentPosts = getStoredPosts();
  const updatedPosts = [newPost, ...currentPosts];
  savePosts(updatedPosts);
  
  // 触发自定义事件通知其他组件
  window.dispatchEvent(new CustomEvent('post-added', { detail: newPost }));
};

// 生成时间戳文本
export const getTimeAgo = (): string => {
  return "刚刚";
};

// 格式化日期
export const formatDate = (date: Date): string => {
  return date.toLocaleString('zh-CN', { 
    year: 'numeric', 
    month: 'numeric', 
    day: 'numeric',
    hour: 'numeric',
    minute: 'numeric'
  });
}; 