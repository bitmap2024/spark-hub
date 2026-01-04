
import React from "react";
import { useParams } from "react-router-dom";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { MessageSquare, MapPin, School, User, Calendar } from "lucide-react";
import { useUserByUsername, useCurrentUser, useUserKnowledgeBasesByUsername } from "@/lib/api";
import UserAvatar from "@/components/UserAvatar";

const UserProfilePage: React.FC = () => {
  const { username } = useParams<{ username: string }>();
  const [isLoginOpen, setIsLoginOpen] = React.useState(false);
  const [isUserFollowing, setIsUserFollowing] = React.useState(false);
  const [activeTab, setActiveTab] = React.useState<"knowledgeBases" | "likes">("knowledgeBases");
  
  const { data: currentUserData, isLoading: isCurrentUserLoading } = useCurrentUser();
  const { data: userData, isLoading: isUserLoading } = useUserByUsername(username || "");
  const { data: knowledgeBases, isLoading: isKnowledgeBasesLoading } = useUserKnowledgeBasesByUsername(username || "");
  
  const isCurrentUser = currentUserData?.username === username;

  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };

  const handleSendMessage = () => {
    if (!currentUserData) {
      setIsLoginOpen(true);
      return;
    }
    // Navigate to messages would go here
  };

  const handleFollow = () => {
    if (!currentUserData) {
      setIsLoginOpen(true);
      return;
    }
    setIsUserFollowing(prev => !prev);
  };

  if (isUserLoading || !userData) {
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white">加载中...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onLoginClick={handleLoginClick} />
      <LeftSidebar />
      <div className="ml-64 mt-16">
        {/* 顶部横向大卡片 */}
        <div className="relative bg-gradient-to-r from-[#232526] to-[#414345] h-56 flex items-end px-12 pb-6">
          <UserAvatar 
            username={userData.username}
            avatarSrc={userData.avatar}
            size="lg"
            className="h-28 w-28 border-4 border-white" 
          />
          <div className="ml-8 text-white">
            <div className="text-2xl font-bold">{userData.username}</div>
            <div className="mt-2 text-gray-300">抖音号：{userData.id}</div>
            <div className="flex gap-6 mt-2 text-gray-200">
              <span>关注 <b>{userData.following}</b></span>
              <span>粉丝 <b>{userData.followers}</b></span>
              <span>获赞 <b>0</b></span>
            </div>
          </div>
          {isCurrentUser ? (
            <Button className="absolute right-12 bottom-6" variant="outline">
              编辑资料
            </Button>
          ) : (
            <div className="absolute right-12 bottom-6 flex gap-4">
              <Button 
                variant={isUserFollowing ? "outline" : "default"}
                onClick={handleFollow}
              >
                {isUserFollowing ? "已关注" : "关注"}
              </Button>
              <Button variant="outline" onClick={handleSendMessage}>
                <MessageSquare className="h-4 w-4 mr-2" />
                发私信
              </Button>
            </div>
          )}
        </div>
        
        {/* 用户详细资料区域 */}
        <div className="max-w-full px-12 py-8">
          <div className="bg-[#1a1a1a] rounded-lg p-6 mb-6 border border-gray-800">
            <h3 className="text-white text-lg font-medium mb-4">个人信息</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="flex items-center text-gray-300">
                <User className="h-5 w-5 mr-3 text-gray-500" />
                <div>
                  <div className="text-sm text-gray-500">性别</div>
                  <div className="mt-1">男</div>
                </div>
              </div>
              <div className="flex items-center text-gray-300">
                <Calendar className="h-5 w-5 mr-3 text-gray-500" />
                <div>
                  <div className="text-sm text-gray-500">年龄</div>
                  <div className="mt-1">25</div>
                </div>
              </div>
              <div className="flex items-center text-gray-300">
                <School className="h-5 w-5 mr-3 text-gray-500" />
                <div>
                  <div className="text-sm text-gray-500">学校</div>
                  <div className="mt-1">北京大学</div>
                </div>
              </div>
              <div className="flex items-center text-gray-300">
                <MapPin className="h-5 w-5 mr-3 text-gray-500" />
                <div>
                  <div className="text-sm text-gray-500">位置</div>
                  <div className="mt-1">{userData.location || "未设置"}</div>
                </div>
              </div>
            </div>
            <div className="mt-6">
              <div className="text-sm text-gray-500">个人简介</div>
              <div className="mt-2 text-gray-300">{userData.experience || "这个人很神秘，没有留下任何介绍~"}</div>
            </div>
          </div>
          
          {/* Tab区域 */}
          <div className="flex border-b border-gray-800 mb-6">
            <button 
              className={`py-3 px-6 font-medium border-b-2 ${
                activeTab === "knowledgeBases" 
                  ? "text-white border-primary" 
                  : "text-gray-400 border-transparent"
              }`}
              onClick={() => setActiveTab("knowledgeBases")}
            >
              知识库 {knowledgeBases?.length || 0}
            </button>
            <button 
              className={`py-3 px-6 font-medium border-b-2 ${
                activeTab === "likes" 
                  ? "text-white border-primary" 
                  : "text-gray-400 border-transparent"
              }`}
              onClick={() => setActiveTab("likes")}
            >
              喜欢 0
            </button>
          </div>
          
          {/* 内容区域 */}
          <div className="mt-6">
            {activeTab === "knowledgeBases" ? (
              isKnowledgeBasesLoading ? (
                <div className="flex justify-center items-center h-64">
                  <p className="text-gray-400">加载中...</p>
                </div>
              ) : knowledgeBases && knowledgeBases.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {/* 知识库列表将在这里渲染 */}
                  <div className="text-gray-300">知识库内容将在这里显示</div>
                </div>
              ) : (
                <div className="flex flex-col justify-center items-center h-64 text-gray-400">
                  <svg width="64" height="64" fill="none" viewBox="0 0 48 48"><rect width="48" height="48" rx="24" fill="#232526"/><path d="M24 16v8m0 0v4m0-4h4m-4 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/></svg>
                  <p className="mt-4">暂无内容<br/>该账号还未发布过作品哦~</p>
                </div>
              )
            ) : (
              <div className="flex flex-col justify-center items-center h-64 text-gray-400">
                <svg width="64" height="64" fill="none" viewBox="0 0 48 48"><rect width="48" height="48" rx="24" fill="#232526"/><path d="M24 16v8m0 0v4m0-4h4m-4 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/></svg>
                <p className="mt-4">暂无喜欢的内容</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default UserProfilePage;
