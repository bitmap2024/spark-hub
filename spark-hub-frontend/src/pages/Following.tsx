import React, { useState, useEffect } from "react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import { useAllKnowledgeBases, useCurrentUser } from "@/lib/api";
import { Link } from "react-router-dom";
import UserAvatar from "@/components/UserAvatar";

const Following: React.FC = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const { data: currentUser, isLoading: isUserLoading } = useCurrentUser();
  const { data: knowledgeBases, isLoading: isKBLoading } = useAllKnowledgeBases();
  const [followingContent, setFollowingContent] = useState<any[]>([]);
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  // 生成关注者的内容
  useEffect(() => {
    if (!knowledgeBases || !currentUser) return;
    
    // 生成模拟的关注内容
    const generateFollowingContent = () => {
      // 用户的关注列表
      let followedUsers = currentUser.followingList || [];
      
      // 如果关注列表为空，为了演示效果，生成一些随机关注的用户
      if (!followedUsers || followedUsers.length === 0) {
        followedUsers = Array.from({ length: 10 }, (_, i) => 200 + i);
      }
      
      // 过滤出这些用户创建的知识库
      let result = knowledgeBases.filter(kb => followedUsers.includes(kb.userId));
      
      // 如果内容不够30条，通过复制和修改创建更多
      if (result.length < 30) {
        const extraNeeded = 30 - result.length;
        const originalItems = result.length > 0 ? result : knowledgeBases.slice(0, 5);
        
        for (let i = 0; i < extraNeeded; i++) {
          const original = originalItems[i % originalItems.length];
          const userId = followedUsers[i % followedUsers.length];
          
          const newItem = {
            ...original,
            id: original.id + 20000 + i,
            userId: userId, // 确保这是关注的用户创建的
            title: `关注更新: ${original.title} ${i + 1}`,
            description: `您关注的用户${userId}发布了新的知识库内容: ${original.description.slice(0, 30)}...`,
            createdAt: new Date().toISOString().split('T')[0], // 今天创建的
            updatedAt: new Date().toISOString().split('T')[0]
          };
          result.push(newItem);
        }
      }
      
      // 按最新更新排序
      return result.sort((a, b) => 
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
      );
    };
    
    setFollowingContent(generateFollowingContent());
  }, [knowledgeBases, currentUser]);
  
  const isLoading = isUserLoading || isKBLoading;
  
  // 未登录状态
  if (!isLoading && !currentUser) {
    return (
      <div className="min-h-screen bg-[#121212]">
        <Header onLoginClick={handleLoginClick} />
        <LeftSidebar />
        
        <div className="ml-64 mt-16 h-[calc(100vh-4rem)] flex flex-col items-center justify-center">
          <div className="text-center max-w-md mx-auto p-8 rounded-lg">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-24 w-24 text-gray-500 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <h2 className="mt-6 text-2xl font-bold text-white">登录后查看关注内容</h2>
            <p className="mt-2 text-gray-400">登录后可以看到你关注的创作者发布的最新内容</p>
            <button 
              onClick={handleLoginClick}
              className="mt-6 bg-[#fe2c55] text-white px-8 py-3 rounded-full hover:bg-[#fe2c55]/90 transition"
            >
              立即登录
            </button>
          </div>
        </div>
        
        <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
          <DialogContent className="sm:max-w-md">
            <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
          </DialogContent>
        </Dialog>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onLoginClick={handleLoginClick} />
      <LeftSidebar />
      {/* 主体内容区域，右侧主区域布局 */}
      <div className="ml-64 mt-16 p-8">
        <h1 className="text-2xl font-bold text-white mb-6">关注更新 ({followingContent.length})</h1>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-white">加载中...</div>
          </div>
        ) : followingContent.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-400">
            <svg width="64" height="64" fill="none" viewBox="0 0 48 48">
              <rect width="48" height="48" rx="24" fill="#232526"/>
              <path d="M24 16v8m0 0v4m0-4h4m-4 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/>
            </svg>
            <p className="mt-4">还没有关注的内容<br/>快去关注感兴趣的创作者吧</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {followingContent.map((item) => (
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
                    
                    {/* 更新标签 */}
                    <div className="absolute top-2 right-2 bg-blue-500 text-white text-xs px-2 py-1 rounded">
                      新更新
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
                    <p className="text-gray-500 text-xs line-clamp-2 mb-2">
                      {item.description}
                    </p>
                    <div className="text-gray-500 text-xs flex gap-2">
                      <span>{new Date(item.updatedAt).toLocaleDateString('zh-CN')} 更新</span>
                      <span>•</span>
                      <span>{item.papers.length} 篇论文</span>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
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

export default Following; 