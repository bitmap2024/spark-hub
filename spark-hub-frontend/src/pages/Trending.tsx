import React, { useState, useEffect } from "react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import { useAllKnowledgeBases } from "@/lib/api";
import { Link, useNavigate } from "react-router-dom";
import UserAvatar from "@/components/UserAvatar";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const Trending: React.FC = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const { data: knowledgeBases, isLoading } = useAllKnowledgeBases();
  const [trendingItems, setTrendingItems] = useState<any[]>([]);
  const [trendingUsers, setTrendingUsers] = useState<any[]>([]);
  const [activeTab, setActiveTab] = useState<"content" | "users">("content");
  const navigate = useNavigate();
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  const handleSearch = (query: string) => {
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };
  
  // 生成模拟的热门内容
  useEffect(() => {
    if (!knowledgeBases) return;
    
    // 制作100条热门内容
    const generateTrendingItems = () => {
      // 将原有知识库按照星数排序
      const sortedKnowledgeBases = [...knowledgeBases].sort((a, b) => b.stars - a.stars);
      
      // 不够100个则复制并修改
      let result = [...sortedKnowledgeBases];
      
      if (result.length >= 100) {
        return result.slice(0, 100);
      }
      
      const extraNeeded = 100 - result.length;
      
      for (let i = 0; i < extraNeeded; i++) {
        const original = sortedKnowledgeBases[i % sortedKnowledgeBases.length];
        const newItem = {
          ...original,
          id: original.id + 10000 + i,
          title: `热门: ${original.title} #${i + 1}`,
          stars: Math.floor(original.stars * 2 + Math.random() * 100), // 热门内容有更多星标
          viewCount: Math.floor(Math.random() * 10000) + 5000, // 添加浏览量
          liked: Math.floor(Math.random() * 500) + 100, // 添加点赞数
        };
        result.push(newItem);
      }
      
      // 随机选择热门话题标签
      const hotTags = ["AI应用", "大模型研究", "计算机视觉", "数据科学", "机器学习", "区块链", "深度学习新进展", "医疗AI"];
      
      // 确保每个条目有一个热门标签
      result = result.map(item => {
        const randomTagIndex = Math.floor(Math.random() * hotTags.length);
        const hotTag = hotTags[randomTagIndex];
        
        if (!item.tags.includes(hotTag)) {
          return {
            ...item,
            tags: [...item.tags, hotTag]
          };
        }
        return item;
      });
      
      // 按照星数重新排序
      return result.sort((a, b) => b.stars - a.stars).slice(0, 100);
    };
    
    setTrendingItems(generateTrendingItems());
    
    // 生成热门用户
    const generateTrendingUsers = () => {
      // 从知识库作者中提取用户信息并增加用户数据
      const userMap = new Map();
      
      knowledgeBases.forEach(kb => {
        if (!userMap.has(kb.userId)) {
          userMap.set(kb.userId, {
            id: kb.userId,
            username: `用户${kb.userId}`,
            followers: Math.floor(Math.random() * 10000) + 500,
            following: Math.floor(Math.random() * 1000) + 50,
            posts: Math.floor(Math.random() * 50) + 5,
            avatarSrc: `https://api.dicebear.com/7.x/avataaars/svg?seed=${kb.userId}`,
            tags: ["创作者", "知识达人"],
            bio: `热门创作者，分享${kb.tags[0] || "专业"}领域知识`,
            verified: Math.random() > 0.7,
          });
        } else {
          // 增加已有用户的统计
          const user = userMap.get(kb.userId);
          user.posts += 1;
          user.followers += Math.floor(Math.random() * 100);
        }
      });
      
      // 转为数组并按粉丝数排序
      const users = Array.from(userMap.values());
      const sortedUsers = users.sort((a, b) => b.followers - a.followers);
      
      // 补充到50个用户
      let result = [...sortedUsers];
      
      if (result.length >= 50) {
        return result.slice(0, 50);
      }
      
      const extraNeeded = 50 - result.length;
      const userTags = ["技术专家", "科研学者", "行业KOL", "领域专家", "教育工作者"];
      
      for (let i = 0; i < extraNeeded; i++) {
        const userId = 10000 + i;
        const randomTagIndex = Math.floor(Math.random() * userTags.length);
        const userTag = userTags[randomTagIndex];
        
        const newUser = {
          id: userId,
          username: `热门用户${userId}`,
          followers: Math.floor(Math.random() * 50000) + 10000,
          following: Math.floor(Math.random() * 1000) + 100,
          posts: Math.floor(Math.random() * 100) + 10,
          avatarSrc: `https://api.dicebear.com/7.x/avataaars/svg?seed=${userId}`,
          tags: [userTag, "创作者"],
          bio: "分享前沿知识和专业见解",
          verified: Math.random() > 0.5,
        };
        
        result.push(newUser);
      }
      
      return result.sort((a, b) => b.followers - a.followers);
    };
    
    setTrendingUsers(generateTrendingUsers());
  }, [knowledgeBases]);
  
  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onLoginClick={handleLoginClick} onSearch={handleSearch} />
      <LeftSidebar />
      {/* 主体内容区域，右侧主区域布局 */}
      <div className="ml-64 mt-16 p-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-white">热门推荐</h1>
          <Tabs 
            defaultValue="content" 
            value={activeTab} 
            onValueChange={(value) => setActiveTab(value as "content" | "users")}
            className="w-72"
          >
            <TabsList className="grid w-full grid-cols-2 bg-[#1f1f1f]">
              <TabsTrigger value="content" className="text-white">热门内容</TabsTrigger>
              <TabsTrigger value="users" className="text-white">热门用户</TabsTrigger>
            </TabsList>
          </Tabs>
        </div>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-white">加载中...</div>
          </div>
        ) : (
          <>
            {/* 热门内容 */}
            {activeTab === "content" && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {trendingItems.map((item) => (
                  <Link 
                    key={item.id} 
                    to={`/knowledge-base/${item.id}`}
                    className="group"
                  >
                    <div className="bg-[#1f1f1f] rounded-lg overflow-hidden transition transform hover:scale-[1.02] hover:shadow-xl">
                      <div className="relative h-40 bg-gray-900">
                        <img 
                          src={`https://picsum.photos/seed/${item.id}/500/300`} 
                          alt={item.title} 
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
                          <h3 className="text-white font-medium truncate">{item.title}</h3>
                        </div>
                        
                        {/* 热门标签 */}
                        <div className="absolute top-2 right-2 bg-red-500 text-white text-xs px-2 py-1 rounded">
                          热门
                        </div>
                      </div>
                      <div className="p-4">
                        <div className="flex items-center mb-2">
                          <UserAvatar 
                            username={`用户${item.userId}`} 
                            avatarSrc={`https://api.dicebear.com/7.x/avataaars/svg?seed=${item.userId}`}
                            size="sm"
                            className="w-6 h-6 mr-2" 
                          />
                          <span className="text-gray-400 text-sm">用户{item.userId}</span>
                        </div>
                        <div className="text-gray-500 text-xs flex gap-2 mb-2">
                          <span>{item.stars} 收藏</span>
                          <span>•</span>
                          <span>{item.viewCount || Math.floor(Math.random() * 10000) + 1000} 浏览</span>
                        </div>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {item.tags.slice(0, 2).map((tag: string, idx: number) => (
                            <span key={idx} className="px-2 py-0.5 bg-gray-800 text-xs rounded-full text-blue-300">
                              {tag}
                            </span>
                          ))}
                          {item.tags.length > 2 && (
                            <span className="text-xs text-gray-500">+{item.tags.length - 2}</span>
                          )}
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}
            
            {/* 热门用户 */}
            {activeTab === "users" && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
                {trendingUsers.map((user) => (
                  <Link 
                    key={user.id} 
                    to={`/user/${user.username}`}
                    className="group"
                  >
                    <div className="bg-[#1f1f1f] rounded-lg overflow-hidden transition transform hover:scale-[1.02] hover:shadow-xl">
                      <div className="p-6 flex flex-col items-center">
                        <div className="relative mb-4">
                          <UserAvatar 
                            username={user.username} 
                            avatarSrc={user.avatarSrc}
                            size="lg"
                            className="w-24 h-24 border-2 border-gray-700" 
                          />
                          {user.verified && (
                            <div className="absolute bottom-0 right-0 bg-blue-500 text-white rounded-full p-1">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                                <path fillRule="evenodd" d="M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                              </svg>
                            </div>
                          )}
                        </div>
                        <h3 className="text-white font-medium text-lg mb-1">{user.username}</h3>
                        <p className="text-gray-400 text-sm mb-3 text-center">{user.bio}</p>
                        <div className="flex justify-center gap-4 mb-3 text-sm text-gray-300">
                          <div className="flex flex-col items-center">
                            <span className="font-bold">{user.followers.toLocaleString()}</span>
                            <span className="text-xs text-gray-400">粉丝</span>
                          </div>
                          <div className="flex flex-col items-center">
                            <span className="font-bold">{user.following.toLocaleString()}</span>
                            <span className="text-xs text-gray-400">关注</span>
                          </div>
                          <div className="flex flex-col items-center">
                            <span className="font-bold">{user.posts}</span>
                            <span className="text-xs text-gray-400">作品</span>
                          </div>
                        </div>
                        <div className="flex flex-wrap gap-1 justify-center">
                          {user.tags.map((tag: string, idx: number) => (
                            <span key={idx} className="px-2 py-0.5 bg-gray-800 text-xs rounded-full text-blue-300">
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>
                  </Link>
                ))}
              </div>
            )}
          </>
        )}
      </div>
      
      <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
        <DialogContent className="sm:max-w-md">
          <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Trending; 