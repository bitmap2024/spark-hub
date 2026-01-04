import React, { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { useCurrentUser, getAllKnowledgeBases } from "@/lib/api";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import UserAvatar from "@/components/UserAvatar";

const LikedVideos = () => {
  const navigate = useNavigate();
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const { data: currentUser, isLoading } = useCurrentUser();
  const [likedKnowledgeBases, setLikedKnowledgeBases] = useState<any[]>([]);
  const [isLoadingLikes, setIsLoadingLikes] = useState(true);
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };

  useEffect(() => {
    const fetchLikedVideos = async () => {
      if (!currentUser) return;
      
      try {
        setIsLoadingLikes(true);
        const allKnowledgeBases = await getAllKnowledgeBases();
        
        // 过滤出当前用户喜欢的知识库
        // 假设currentUser.likedKnowledgeBases包含用户喜欢的知识库ID
        if (currentUser.likedKnowledgeBases && Array.isArray(currentUser.likedKnowledgeBases)) {
          const liked = allKnowledgeBases.filter(kb => 
            currentUser.likedKnowledgeBases.includes(kb.id)
          );
          setLikedKnowledgeBases(liked);
        }
      } catch (error) {
        console.error("获取喜欢的视频失败:", error);
      } finally {
        setIsLoadingLikes(false);
      }
    };
    
    fetchLikedVideos();
  }, [currentUser]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white">加载中...</div>
      </div>
    );
  }

  if (!currentUser) {
    return (
      <div className="min-h-screen bg-[#121212] flex">
        <Header onLoginClick={handleLoginClick} />
        <LeftSidebar />
        <div className="flex-1 ml-64 mt-16 flex flex-col items-center justify-center text-white">
          <div className="text-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-24 w-24 text-gray-500 mx-auto" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            <h2 className="mt-6 text-2xl font-bold">请登录查看</h2>
            <p className="mt-2 text-gray-400">登录后才能查看你喜欢的视频</p>
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
      <div className="ml-64 mt-16 p-8">
        <h1 className="text-2xl font-bold text-white mb-6">我喜欢的视频（{likedKnowledgeBases.length}）</h1>
        
        {isLoadingLikes ? (
          <div className="flex justify-center items-center h-64">
            <div className="text-white">加载中...</div>
          </div>
        ) : likedKnowledgeBases.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-gray-400">
            <svg width="64" height="64" fill="none" viewBox="0 0 48 48">
              <rect width="48" height="48" rx="24" fill="#232526"/>
              <path d="M24 16v8m0 0v4m0-4h4m-4 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/>
            </svg>
            <p className="mt-4">暂无喜欢的视频</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {likedKnowledgeBases.map((kb) => (
              <Link 
                key={kb.id} 
                to={`/knowledge-base/${kb.id}`}
                className="group"
              >
                <div className="bg-[#1f1f1f] rounded-lg overflow-hidden transition transform hover:scale-[1.02] hover:shadow-xl">
                  <div className="relative h-48 bg-gray-900">
                    <img 
                      src={`https://picsum.photos/seed/${kb.id}/500/300`} 
                      alt={kb.title} 
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
                      <h3 className="text-white font-medium truncate">{kb.title}</h3>
                    </div>
                  </div>
                  <div className="p-4">
                    <div className="flex items-center mb-2">
                      <UserAvatar 
                        username={`用户${kb.userId}`} 
                        avatarSrc={`https://api.dicebear.com/7.x/avataaars/svg?seed=${kb.userId}`}
                        size="sm"
                        className="w-6 h-6 mr-2" 
                      />
                      <span className="text-gray-400 text-sm">用户{kb.userId}</span>
                    </div>
                    <div className="text-gray-500 text-xs flex gap-2">
                      <span>{kb.stars} 收藏</span>
                      <span>•</span>
                      <span>{kb.papers.length} 篇论文</span>
                    </div>
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default LikedVideos;
