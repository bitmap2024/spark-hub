import React, { useState, useEffect } from "react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import { useAllKnowledgeBases, useCurrentUser } from "@/lib/api";
import { Link } from "react-router-dom";
import UserAvatar from "@/components/UserAvatar";

const Friends: React.FC = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const { data: currentUser, isLoading: isUserLoading } = useCurrentUser();
  const { data: knowledgeBases, isLoading: isKBLoading } = useAllKnowledgeBases();
  const [friendsContent, setFriendsContent] = useState<any[]>([]);
  const [friendList, setFriendList] = useState<number[]>([]);
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  // 生成随机朋友列表
  useEffect(() => {
    if (!currentUser) return;
    
    // 模拟生成10个朋友ID (创作者)
    const generateFriends = () => {
      // 在实际应用中，朋友列表应该从API获取
      // 这里我们随机生成一些用户ID作为朋友
      const friends = Array.from({ length: 10 }, (_, i) => 200 + i);
      setFriendList(friends);
    };
    
    generateFriends();
  }, [currentUser]);
  
  // 生成朋友的内容
  useEffect(() => {
    if (!knowledgeBases || friendList.length === 0) return;
    
    // 生成模拟的朋友内容
    const generateFriendsContent = () => {
      // 过滤出朋友创建的知识库
      let result = knowledgeBases.filter(kb => friendList.includes(kb.userId));
      
      // 如果内容不够50条，通过复制和修改创建更多
      if (result.length < 50) {
        const extraNeeded = 50 - result.length;
        const originalItems = result.length > 0 ? result : knowledgeBases.slice(0, 5);
        
        for (let i = 0; i < extraNeeded; i++) {
          const original = originalItems[i % originalItems.length];
          const friendId = friendList[i % friendList.length];
          
          const newItem = {
            ...original,
            id: original.id + 30000 + i,
            userId: friendId, // 确保这是朋友创建的
            title: `好友动态: ${original.title} ${i + 1}`,
            description: `您的好友用户${friendId}发布了新的知识库内容: ${original.description.slice(0, 30)}...`,
            createdAt: new Date(Date.now() - Math.floor(Math.random() * 30) * 86400000).toISOString().split('T')[0], // 最近30天内
            updatedAt: new Date(Date.now() - Math.floor(Math.random() * 7) * 86400000).toISOString().split('T')[0] // 最近7天内
          };
          result.push(newItem);
        }
      }
      
      // 按最新更新排序
      return result.sort((a, b) => 
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
      );
    };
    
    setFriendsContent(generateFriendsContent());
  }, [knowledgeBases, friendList]);
  
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
            <h2 className="mt-6 text-2xl font-bold text-white">登录后查看朋友动态</h2>
            <p className="mt-2 text-gray-400">登录后可以看到你朋友的最新更新和动态</p>
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
      
      {/* 朋友列表区域 */}
      <div className="ml-64 mt-16 p-8">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-white">好友动态</h1>
          <div className="text-sm text-gray-400">{friendList.length} 位好友 • {friendsContent.length} 个动态</div>
        </div>
        
        {/* 朋友列表 */}
        <div className="flex space-x-4 overflow-x-auto pb-4 mb-8">
          {friendList.map((friendId) => (
            <Link 
              key={friendId}
              to={`/user/${friendId}`}
              className="flex flex-col items-center min-w-[80px]"
            >
              <UserAvatar 
                username={`用户${friendId}`}
                avatarSrc={`https://api.dicebear.com/7.x/avataaars/svg?seed=${friendId}`}
                size="lg"
                className="w-16 h-16 mb-2 border-2 border-blue-500 p-1 rounded-full" 
              />
              <span className="text-white text-sm whitespace-nowrap">用户{friendId}</span>
            </Link>
          ))}
        </div>
        
        {/* 好友动态内容 */}
        <h2 className="text-xl font-semibold text-white mb-4">最新动态</h2>
        
        {isLoading ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-white">加载中...</div>
          </div>
        ) : friendsContent.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-400">
            <svg width="64" height="64" fill="none" viewBox="0 0 48 48">
              <rect width="48" height="48" rx="24" fill="#232526"/>
              <path d="M24 16v8m0 0v4m0-4h4m-4 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/>
            </svg>
            <p className="mt-4">暂无朋友动态<br/>添加更多朋友获取更多内容</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {friendsContent.map((item) => (
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
                    
                    {/* 好友标签 */}
                    <div className="absolute top-2 right-2 bg-green-500 text-white text-xs px-2 py-1 rounded">
                      好友
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
                      <span>{item.stars} 收藏</span>
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

export default Friends; 