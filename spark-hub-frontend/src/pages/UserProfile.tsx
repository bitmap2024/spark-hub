import React, { useState, useEffect } from "react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { useParams, Link, useNavigate } from "react-router-dom";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { useTranslation } from "react-i18next";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogFooter 
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from "@/components/ui/select";
import { MessageSquare, Bookmark } from "lucide-react";
import { 
  useCurrentUser, 
  useUserByUsername, 
  useUser,
  useUserKnowledgeBases, 
  useUserKnowledgeBasesByUsername,
  isFollowing,
  useUpdateCurrentUser
} from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import MessageButton from "@/components/MessageButton";
import UserAvatar from "@/components/UserAvatar";
import { format } from "date-fns";

// 浏览历史记录类型
interface BrowsingHistoryItem {
  id: number;
  title: string;
  type: string;
  timestamp: Date;
  imageUrl?: string;
  knowledgeBaseId?: number;
}

// 收藏项目类型
interface FavoriteItem {
  id: number;
  title: string;
  type: string;
  addedAt: Date;
  description?: string;
  knowledgeBaseId?: number;
}

// 喜欢项目类型
interface LikedItem {
  id: number;
  title: string;
  type: string;
  likedAt: Date;
  creator: string;
  knowledgeBaseId?: number;
}

// 模拟浏览历史数据
const mockBrowsingHistory: BrowsingHistoryItem[] = [
  {
    id: 1,
    title: "深度学习基础",
    type: "知识库",
    timestamp: new Date(2023, 11, 20, 14, 30),
    knowledgeBaseId: 3
  },
  {
    id: 2,
    title: "机器学习入门",
    type: "知识库",
    timestamp: new Date(2023, 11, 18, 10, 15),
    knowledgeBaseId: 2
  },
  {
    id: 3,
    title: "自然语言处理",
    type: "知识库",
    timestamp: new Date(2023, 11, 15, 16, 45),
    knowledgeBaseId: 5
  }
];

// 模拟收藏数据
const mockFavorites: FavoriteItem[] = [
  {
    id: 1,
    title: "强化学习综述",
    type: "知识库",
    addedAt: new Date(2023, 11, 25, 9, 10),
    description: "强化学习算法的全面概述与应用案例",
    knowledgeBaseId: 7
  },
  {
    id: 2,
    title: "图神经网络入门",
    type: "知识库",
    addedAt: new Date(2023, 11, 22, 15, 30),
    description: "图神经网络基础理论与实现",
    knowledgeBaseId: 8
  },
  {
    id: 3,
    title: "计算机视觉技术",
    type: "知识库",
    addedAt: new Date(2023, 11, 19, 11, 45),
    description: "从传统CV到深度学习的发展与应用",
    knowledgeBaseId: 9
  }
];

// 模拟喜欢数据
const mockLikes: LikedItem[] = [
  {
    id: 1,
    title: "生成式人工智能最新进展",
    type: "知识库",
    likedAt: new Date(2023, 11, 28, 13, 20),
    creator: "AI研究员",
    knowledgeBaseId: 10
  },
  {
    id: 2,
    title: "多模态学习方法",
    type: "知识库",
    likedAt: new Date(2023, 11, 26, 10, 15),
    creator: "数据科学家",
    knowledgeBaseId: 11
  },
  {
    id: 3,
    title: "转换器架构详解",
    type: "知识库",
    likedAt: new Date(2023, 11, 23, 16, 45),
    creator: "NLP专家",
    knowledgeBaseId: 12
  }
];

interface UserProfileProps {
  isCurrentUser?: boolean;
}

const UserProfile: React.FC<UserProfileProps> = ({ isCurrentUser = false }) => {
  const { t } = useTranslation();
  const { username } = useParams<{ username: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  
  // State management - place all useState hooks at the top to maintain consistent order
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"knowledgeBases" | "likes" | "history" | "favorites">("knowledgeBases");
  const [isUserFollowing, setIsUserFollowing] = useState(false);
  const [browsingHistory, setBrowsingHistory] = useState<BrowsingHistoryItem[]>([]);
  const [favorites, setFavorites] = useState<FavoriteItem[]>([]);
  const [likes, setLikes] = useState<LikedItem[]>([]);
  const [editForm, setEditForm] = useState({
    gender: "男",
    age: "25",
    school: "北京大学",
    experience: "热爱生活，热爱分享，记录美好瞬间"
  });
  
  // Get current user data
  const { data: currentUserData, isLoading: isCurrentUserLoading } = useCurrentUser();
  
  // Get data for the profile user
  const { data: userDataByUsername, isLoading: isUserByUsernameLoading } = useUserByUsername(
    username || ""
  );
  
  // Determine user ID for profile - from username or use 0 for current user
  const userId = isCurrentUser 
    ? 0 
    : (userDataByUsername?.id || parseInt(username?.replace(/\D/g, '') || "1"));
  
  // Get user data - either current user or specified user
  const userData = isCurrentUser 
    ? currentUserData 
    : userDataByUsername;
  
  const isUserLoading = isCurrentUser 
    ? isCurrentUserLoading 
    : isUserByUsernameLoading;
  
  // Get knowledge bases for the user
  const { data: knowledgeBases, isLoading: isKnowledgeBasesLoading } = isCurrentUser 
    ? useUserKnowledgeBases(0) 
    : useUserKnowledgeBasesByUsername(username || "");
  
  const updateUserMutation = useUpdateCurrentUser();
  
  // Check if current user is following the profile user
  useEffect(() => {
    const checkFollowing = async () => {
      if (userData && !isCurrentUser && currentUserData) {
        try {
          const following = await isFollowing(userData.id);
          setIsUserFollowing(following);
        } catch (error) {
          console.error("Error checking follow status:", error);
        }
      }
    };
    
    if (userData && currentUserData) {
      checkFollowing();
    }
  }, [userData, isCurrentUser, currentUserData]);

  // Initialize edit form with user data when available
  useEffect(() => {
    if (userData) {
      setEditForm({
        gender: (userData as any).gender || "男",
        age: (userData as any).age?.toString() || "25",
        school: (userData as any).school || "北京大学",
        experience: userData.experience || "热爱生活，热爱分享，记录美好瞬间"
      });
    }
  }, [userData]);

  // 加载浏览历史
  useEffect(() => {
    // 这里应该调用API获取浏览历史数据
    // 目前使用模拟数据
    setBrowsingHistory(mockBrowsingHistory);
  }, []);

  // 加载收藏数据
  useEffect(() => {
    // 这里应该调用API获取收藏数据
    // 目前使用模拟数据
    setFavorites(mockFavorites);
  }, []);

  // 加载喜欢数据
  useEffect(() => {
    // 这里应该调用API获取喜欢数据
    // 目前使用模拟数据
    setLikes(mockLikes);
  }, []);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setEditForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSelectChange = (name: string, value: string) => {
    setEditForm(prev => ({ ...prev, [name]: value }));
  };

  const handleSave = async () => {
    if (isCurrentUser) {
      try {
        await updateUserMutation.mutateAsync({
          gender: editForm.gender,
          age: parseInt(editForm.age),
          school: editForm.school,
          experience: editForm.experience
        });
        
        setIsEditOpen(false);
        toast({
          title: t('toast.saveSuccess'),
          description: t('toast.profileUpdated'),
        });
      } catch (error) {
        console.error("更新资料失败:", error);
        toast({
          title: t('toast.updateFailed'),
          description: t('common.errorSaving', '保存个人资料时出错'),
          variant: "destructive"
        });
      }
    } else {
      setIsEditOpen(false);
      // 非当前用户不能编辑资料
    }
  };

  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  const handleSendMessage = () => {
    if (!currentUserData) {
      setIsLoginOpen(true);
      return;
    }
    
    navigate(`/messages/${userId}`);
  };

  const handleFollow = async () => {
    if (!currentUserData) {
      setIsLoginOpen(true);
      return;
    }
    
    try {
      if (isUserFollowing) {
        // We'll manually update the UI without using the useUnfollowUser mutation
        setIsUserFollowing(false);
        toast({
          title: t('toast.unfollowSuccess'),
          description: t('toast.unfollowed', { username: userData?.username }),
        });
        // In a real application, call the actual API here
      } else {
        // We'll manually update the UI without using the useFollowUser mutation
        setIsUserFollowing(true);
        toast({
          title: t('toast.followSuccess'),
          description: t('toast.followed', { username: userData?.username }),
        });
        // In a real application, call the actual API here
      }
    } catch (error) {
      console.error("关注操作失败:", error);
    }
  };

  if (isUserLoading || isCurrentUserLoading || !userData) {
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white">{t('common.loading')}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onLoginClick={handleLoginClick} />
      <LeftSidebar />
      {/* 主体内容区域，右侧主区域布局 */}
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
            <div className="mt-2 text-gray-300">{t('user.sparkId')}：{userData.id}</div>
            <div className="flex gap-6 mt-2 text-gray-200">
              <span>{t('user.following')} <b>{userData.following}</b></span>
              <span>{t('user.followers')} <b>{userData.followers}</b></span>
              <span>{t('user.likes')} <b>0</b></span>
            </div>
            <div className="mt-3 text-gray-300 max-w-md opacity-80 line-clamp-2">
              {userData.experience || editForm.experience || "热爱生活，热爱分享，记录美好瞬间"}
            </div>
          </div>
          {isCurrentUser ? (
            <Button className="absolute right-12 bottom-6" variant="outline" onClick={() => setIsEditOpen(true)}>
              {t('user.editProfile')}
            </Button>
          ) : (
            <div className="absolute right-12 bottom-6 flex gap-4">
              <Button 
                variant={isUserFollowing ? "outline" : "default"}
                onClick={handleFollow}
              >
                {isUserFollowing ? t('user.unfollow') : t('user.follow')}
              </Button>
              <Button variant="outline" onClick={handleSendMessage}>
                <MessageSquare className="h-4 w-4 mr-2" />
                {t('user.sendMessage')}
              </Button>
            </div>
          )}
        </div>
        {/* 下方Tab和内容区 */}
        <div className="max-w-full px-8">
          <div className="flex border-b border-gray-800 mb-6">
            <button 
              className={`py-3 px-6 font-medium border-b-2 ${
                activeTab === "knowledgeBases" 
                  ? "text-white border-primary" 
                  : "text-gray-400 border-transparent"
              }`}
              onClick={() => setActiveTab("knowledgeBases")}
            >
              {t('tabs.knowledgeBases')} {knowledgeBases?.length || 0}
            </button>
            <button 
              className={`py-3 px-6 font-medium border-b-2 ${
                activeTab === "likes" 
                  ? "text-white border-primary" 
                  : "text-gray-400 border-transparent"
              }`}
              onClick={() => setActiveTab("likes")}
            >
              {t('tabs.likes')} {likes.length}
            </button>
            <button 
              className={`py-3 px-6 font-medium border-b-2 ${
                activeTab === "favorites" 
                  ? "text-white border-primary" 
                  : "text-gray-400 border-transparent"
              }`}
              onClick={() => setActiveTab("favorites")}
            >
              {t('tabs.favorites')} {favorites.length}
            </button>
            <button 
              className={`py-3 px-6 font-medium border-b-2 ${
                activeTab === "history" 
                  ? "text-white border-primary" 
                  : "text-gray-400 border-transparent"
              }`}
              onClick={() => setActiveTab("history")}
            >
              {t('tabs.history')} {browsingHistory.length}
            </button>
          </div>
          <div className="mt-6">
            {activeTab === "knowledgeBases" ? (
              isKnowledgeBasesLoading ? (
                <div className="flex justify-center items-center h-64">
                  <p>{t('common.loading')}</p>
                </div>
              ) : knowledgeBases && knowledgeBases.length > 0 ? (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {knowledgeBases.map((kb) => (
                    <Link 
                      key={kb.id} 
                      to={`/knowledge-base/${kb.id}`}
                      className="block"
                    >
                      <div className="bg-card rounded-lg p-5 border border-gray-700 hover:border-gray-600 transition-colors h-full">
                        <div className="flex justify-between items-start mb-2">
                          <h3 className="font-bold text-lg text-blue-400 truncate mr-2">{kb.title}</h3>
                          <div className="flex items-center space-x-3 text-sm shrink-0">
                            <div className="flex items-center text-yellow-500">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.783-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
                              </svg>
                              <span>{kb.stars}</span>
                            </div>
                            <div className="flex items-center text-blue-500">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z" />
                              </svg>
                              <span>{kb.forks}</span>
                            </div>
                            <div className="flex items-center text-purple-500">
                              <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                              </svg>
                              <span>{kb.papers.length}</span>
                            </div>
                          </div>
                        </div>
                        <p className="text-sm text-gray-300 mb-4 line-clamp-2">{kb.description}</p>
                        
                        <div className="flex items-center mb-3 text-xs text-gray-400">
                          <div className="mr-4 whitespace-nowrap">{t('knowledgeBase.updatedAt')}: {new Date(kb.updatedAt).toLocaleDateString('zh-CN')}</div>
                          <div className="whitespace-nowrap">{t('knowledgeBase.createdAt')}: {new Date(kb.createdAt).toLocaleDateString('zh-CN')}</div>
                        </div>
                        
                        <div className="mb-3">
                          {kb.papers.length > 0 && (
                            <div className="text-sm text-gray-300 border-t border-gray-700 pt-3 pb-2">
                              <div className="font-medium mb-2">{t('knowledgeBase.recentPapers')}:</div>
                              {kb.papers.slice(0, 1).map((paper) => (
                                <div key={paper.id} className="flex items-center mb-2">
                                  <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-2 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                  </svg>
                                  <div className="truncate max-w-[200px]">
                                    <span className="text-blue-400 truncate">{paper.title}</span>
                                    <span className="text-xs text-gray-500 ml-2 truncate">（{paper.authors.join(", ")}）</span>
                                  </div>
                                </div>
                              ))}
                              {kb.papers.length > 1 && (
                                <div className="text-xs text-gray-500 italic">{t('knowledgeBase.morePapers', { count: kb.papers.length - 1 })}</div>
                              )}
                            </div>
                          )}
                        </div>
                        
                        <div className="flex flex-wrap gap-2 mt-auto">
                          {kb.tags.slice(0, 3).map((tag, index) => (
                            <span 
                              key={index} 
                              className="px-2 py-1 bg-gray-800 text-xs rounded-full text-blue-300"
                            >
                              {tag}
                            </span>
                          ))}
                          {kb.tags.length > 3 && (
                            <span className="text-xs text-gray-500 self-center">+{kb.tags.length - 3}</span>
                          )}
                        </div>
                      </div>
                    </Link>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col justify-center items-center h-64 text-gray-400">
                  <svg width="64" height="64" fill="none" viewBox="0 0 48 48"><rect width="48" height="48" rx="24" fill="#232526"/><path d="M24 16v8m0 0v4m0-4h4m-4 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/></svg>
                  <p className="mt-4">{t('knowledgeBase.noContent')}<br/>{t('knowledgeBase.noContentDesc')}</p>
                </div>
              )
            ) : activeTab === "likes" ? (
              likes.length > 0 ? (
                <div className="space-y-4">
                  {likes.map((item) => (
                    <div 
                      key={item.id} 
                      className="bg-[#1E1E1E] p-4 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors cursor-pointer"
                      onClick={() => item.knowledgeBaseId && navigate(`/knowledge-base/${item.knowledgeBaseId}`)}
                    >
                      <div className="flex items-center">
                        <div className="flex-grow">
                          <h3 className="text-lg font-medium text-blue-400">{item.title}</h3>
                          <div className="flex items-center mt-1">
                            <span className="text-sm text-gray-400 mr-4">{item.type}</span>
                            <span className="text-sm text-gray-400 mr-4">{t('paper.author')}: {item.creator}</span>
                            <span className="text-sm text-gray-400">
                              {t('common.likedAt', '喜欢于')}: {format(item.likedAt, "yyyy-MM-dd HH:mm")}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center">
                          <svg 
                            xmlns="http://www.w3.org/2000/svg" 
                            className="h-5 w-5 text-red-500 mr-3" 
                            viewBox="0 0 20 20" 
                            fill="currentColor"
                          >
                            <path 
                              fillRule="evenodd" 
                              d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" 
                              clipRule="evenodd" 
                            />
                          </svg>
                          <div className="text-gray-500">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col justify-center items-center h-64 text-gray-400">
                  <svg width="64" height="64" fill="none" viewBox="0 0 48 48"><rect width="48" height="48" rx="24" fill="#232526"/><path d="M24 16v8m0 0v4m0-4h4m-4 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/></svg>
                  <p className="mt-4">{t('common.noLikes', '暂无喜欢的内容')}</p>
                </div>
              )
            ) : activeTab === "favorites" ? (
              favorites.length > 0 ? (
                <div className="space-y-4">
                  {favorites.map((item) => (
                    <div 
                      key={item.id} 
                      className="bg-[#1E1E1E] p-4 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors cursor-pointer"
                      onClick={() => item.knowledgeBaseId && navigate(`/knowledge-base/${item.knowledgeBaseId}`)}
                    >
                      <div className="flex items-center">
                        <div className="flex-grow">
                          <h3 className="text-lg font-medium text-blue-400">{item.title}</h3>
                          <p className="text-sm text-gray-300 mt-1 mb-2">{item.description}</p>
                          <div className="flex items-center mt-1">
                            <span className="text-sm text-gray-400 mr-4">{item.type}</span>
                            <span className="text-sm text-gray-400">
                              {t('common.favoritedAt', '收藏于')}: {format(item.addedAt, "yyyy-MM-dd HH:mm")}
                            </span>
                          </div>
                        </div>
                        <div className="flex items-center">
                          <Bookmark className="h-5 w-5 text-blue-400 mr-3" />
                          <div className="text-gray-500">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                            </svg>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col justify-center items-center h-64 text-gray-400">
                  <svg width="64" height="64" fill="none" viewBox="0 0 48 48"><rect width="48" height="48" rx="24" fill="#232526"/><path d="M24 16v8m0 0v4m0-4h4m-4 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/></svg>
                  <p className="mt-4">{t('common.noFavorites', '暂无收藏内容')}</p>
                </div>
              )
            ) : (
              // 浏览历史内容
              browsingHistory.length > 0 ? (
                <div className="space-y-4">
                  {browsingHistory.map((item) => (
                    <div 
                      key={item.id} 
                      className="bg-[#1E1E1E] p-4 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors cursor-pointer"
                      onClick={() => item.knowledgeBaseId && navigate(`/knowledge-base/${item.knowledgeBaseId}`)}
                    >
                      <div className="flex items-center">
                        <div className="flex-grow">
                          <h3 className="text-lg font-medium text-blue-400">{item.title}</h3>
                          <div className="flex items-center mt-1">
                            <span className="text-sm text-gray-400 mr-4">{item.type}</span>
                            <span className="text-sm text-gray-400">
                              {format(item.timestamp, "yyyy-MM-dd HH:mm")}
                            </span>
                          </div>
                        </div>
                        <div className="text-gray-500">
                          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="flex flex-col justify-center items-center h-64 text-gray-400">
                  <svg width="64" height="64" fill="none" viewBox="0 0 48 48"><rect width="48" height="48" rx="24" fill="#232526"/><path d="M24 16v8m0 0v4m0-4h4m0 0h-4" stroke="#666" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/><rect x="12" y="28" width="24" height="8" rx="2" fill="#232526" stroke="#666" strokeWidth="2"/></svg>
                  <p className="mt-4">{t('common.noHistory', '暂无浏览记录')}</p>
                </div>
              )
            )}
          </div>
        </div>
      </div>
      
      {/* 编辑资料对话框 */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent className="bg-card text-white">
          <DialogHeader>
            <DialogTitle>{t('user.editProfile')}</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="gender" className="text-right">{t('user.gender')}</Label>
              <Select 
                value={editForm.gender} 
                onValueChange={(value) => handleSelectChange("gender", value)}
              >
                <SelectTrigger className="col-span-3">
                  <SelectValue placeholder={t('user.gender')} />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="男">{t('user.male')}</SelectItem>
                  <SelectItem value="女">{t('user.female')}</SelectItem>
                  <SelectItem value="其他">{t('user.other')}</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="age" className="text-right">{t('user.age')}</Label>
              <Input
                id="age"
                name="age"
                value={editForm.age}
                onChange={handleInputChange}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="school" className="text-right">{t('user.school')}</Label>
              <Input
                id="school"
                name="school"
                value={editForm.school}
                onChange={handleInputChange}
                className="col-span-3"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="experience" className="text-right">{t('user.experience')}</Label>
              <Textarea
                id="experience"
                name="experience"
                value={editForm.experience}
                onChange={handleInputChange}
                className="col-span-3"
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsEditOpen(false)}>{t('common.cancel')}</Button>
            <Button onClick={handleSave}>{t('common.save')}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* 登录对话框 */}
      <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
        <DialogContent className="bg-card text-white">
          <DialogHeader>
            <DialogTitle>{t('auth.login')}</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="username" className="text-right">{t('user.username')}</Label>
              <Input
                id="username"
                className="col-span-3"
                placeholder={t('user.username')}
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="password" className="text-right">{t('auth.password')}</Label>
              <Input
                id="password"
                type="password"
                className="col-span-3"
                placeholder={t('auth.password')}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsLoginOpen(false)}>{t('common.cancel')}</Button>
            <Button>{t('auth.login')}</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default UserProfile;
