import React, { useState, useEffect, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Heart, MessageSquare, Share, Bookmark, Plus, Check, MessageCircle, MoreHorizontal, HeartOff, UserMinus, AlertTriangle, Keyboard } from "lucide-react";
import { useAllKnowledgeBases, useCurrentUser, getUserByUsername } from "@/lib/api";
import UserAvatar from "@/components/UserAvatar";

interface KnowledgeBase {
  id: number;
  title: string;
  description: string;
  tags: string[];
  papers: any[];
  stars: number;
  forks: number;
  userId: number;
  createdAt: string;
  updatedAt: string;
}

interface KnowledgeBaseVideoFeedProps {
  sourceType: "recommend" | "friends" | "following";
}

const KnowledgeBaseVideoFeed: React.FC<KnowledgeBaseVideoFeedProps> = ({ sourceType }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const { data: knowledgeBases, isLoading } = useAllKnowledgeBases();
  const { data: currentUser } = useCurrentUser();
  const navigate = useNavigate();
  const [filteredKnowledgeBases, setFilteredKnowledgeBases] = useState<KnowledgeBase[]>([]);
  const [owners, setOwners] = useState<Record<number, any>>({});
  const [isDataGenerated, setIsDataGenerated] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [transitionDirection, setTransitionDirection] = useState<'up' | 'down'>('down');
  const [isPlaying, setIsPlaying] = useState(true);
  const [isFollowing, setIsFollowing] = useState<Record<number, boolean>>({});
  const [likedItems, setLikedItems] = useState<Record<number, boolean>>({});
  const [bookmarkedItems, setBookmarkedItems] = useState<Record<number, boolean>>({});
  const [showMoreOptions, setShowMoreOptions] = useState(false);
  const [showMessages, setShowMessages] = useState(false);
  const videoRef = useRef<HTMLDivElement>(null);
  const moreOptionsRef = useRef<HTMLDivElement>(null);
  
  // 添加更多视频内容
  const videoContents = [
    "https://images.unsplash.com/photo-1522542550221-31fd19575a2d?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Mnx8YWklMjBicmFpbnxlbnwwfHwwfHx8MA%3D%3D",
    "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8M3x8YWklMjBicmFpbnxlbnwwfHwwfHx8MA%3D%3D",
    "https://images.unsplash.com/photo-1620330989164-870d89a04dd0?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8OXx8YWklMjBicmFpbnxlbnwwfHwwfHx8MA%3D%3D",
    "https://images.unsplash.com/photo-1562408590-e32931084e23?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MTV8fHJvYm90fGVufDB8fDB8fHww",
    "https://images.unsplash.com/photo-1589254065878-42c9da997008?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8N3x8cm9ib3R8ZW58MHx8MHx8fDA%3D",
    // 添加更多图片
    "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1535378620166-273708d44e4c?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1485827404703-89b55fcc595e?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1591696205602-2f950c417cb9?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1677442135137-3743cad5025d?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1675514576762-bc0f50221607?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?auto=format&fit=crop&q=60&ixlib=rb-4.0.3",
    "https://images.unsplash.com/photo-1633412802994-5c058f151b66?auto=format&fit=crop&q=60&ixlib=rb-4.0.3"
  ];

  // 创建一个函数来生成大量知识库数据
  const generateKnowledgeBases = (originalData: KnowledgeBase[], minCount: number): KnowledgeBase[] => {
    if (!originalData || originalData.length === 0) return [];
    
    if (originalData.length >= minCount) {
      return originalData;
    }
    
    console.log(`生成数据，原有数量: ${originalData.length}，目标数量: ${minCount}`);
    
    // 复制现有数据并修改以创建更多条目
    const extraNeeded = minCount - originalData.length;
    let result = [...originalData];
    
    for (let i = 0; i < extraNeeded; i++) {
      const original = originalData[i % originalData.length];
      const newKB: KnowledgeBase = {
        ...original,
        id: original.id + 10000 + i, // 创建新的唯一ID
        title: `${original.title} - 变体 ${i + 1}`, // 修改标题
        stars: Math.floor(original.stars * (0.5 + Math.random())), // 随机修改星数
        description: `${original.description} (${i + 1})`, // 修改描述
        createdAt: new Date(new Date(original.createdAt).getTime() + i * 86400000).toISOString().split('T')[0], // 每个项目日期+1天
        updatedAt: new Date(new Date(original.updatedAt).getTime() + i * 86400000).toISOString().split('T')[0]
      };
      result.push(newKB);
    }
    
    console.log(`生成完成，新数据量: ${result.length}`);
    return result;
  };

  // 根据sourceType过滤知识库
  useEffect(() => {
    if (!knowledgeBases || !currentUser) return;

    let filtered: KnowledgeBase[] = [];
    
    if (sourceType === "recommend") {
      // 推荐：随机排序或按照收藏数排序的知识库
      filtered = [...knowledgeBases].sort((a, b) => b.stars - a.stars);
    } else if (sourceType === "following") {
      // 关注：只显示用户关注的创作者创建的知识库
      if (currentUser.followingList && currentUser.followingList.length > 0) {
        filtered = knowledgeBases.filter(kb => 
          currentUser.followingList?.includes(kb.userId)
        );
      } else {
        // 如果没有关注任何人，默认显示一些知识库
        filtered = [...knowledgeBases].sort(() => Math.random() - 0.5).slice(0, 10);
      }
    } else if (sourceType === "friends") {
      // 朋友：相互关注的用户创建的知识库
      // 假设朋友是双向关注的关系
      if (currentUser.followingList && currentUser.followingList.length > 0) {
        filtered = knowledgeBases.filter(kb => {
          // 获取知识库创建者
          const creator = { id: kb.userId, followingList: [] };
          // 检查创建者是否也关注了当前用户
          return currentUser.followingList?.includes(kb.userId) && 
                 creator.followingList.includes(currentUser.id);
        });
        
        // 如果没有朋友关系，默认显示一些知识库
        if (filtered.length === 0) {
          filtered = [...knowledgeBases].sort(() => Math.random() - 0.5).slice(0, 10);
        }
      } else {
        // 默认显示一些随机知识库
        filtered = [...knowledgeBases].sort(() => Math.random() - 0.5).slice(0, 10);
      }
    }

    // 如果筛选后没有内容，默认显示推荐
    if (filtered.length === 0) {
      filtered = [...knowledgeBases].sort((a, b) => b.stars - a.stars);
    }
    
    // 确保有足够多的数据
    const minItemCount = sourceType === "recommend" ? 120 : 30;
    const generatedData = generateKnowledgeBases(filtered, minItemCount);
    
    // 随机打乱顺序，使体验更加真实
    if (sourceType === "recommend") {
      generatedData.sort((a, b) => b.stars - a.stars);
    } else {
      generatedData.sort(() => Math.random() - 0.5);
    }
    
    setFilteredKnowledgeBases(generatedData);
    setIsDataGenerated(true);
    
    // 获取所有知识库创建者的信息
    const fetchOwners = async () => {
      const ownersMap: Record<number, any> = {};
      const followingMap: Record<number, boolean> = {};
      // 只获取前20个知识库的创建者信息，避免过多API调用
      const kbsToFetch = generatedData.slice(0, 20);
      for (const kb of kbsToFetch) {
        try {
          const user = await getUserByUsername(`用户${kb.userId}`);
          ownersMap[kb.userId] = user;
          // 随机设置是否关注 (模拟数据)
          followingMap[kb.userId] = Math.random() > 0.5;
        } catch (error) {
          console.error(`获取用户信息失败: ${kb.userId}`, error);
          // 创建默认用户信息
          ownersMap[kb.userId] = { 
            username: `用户${kb.userId}`, 
            avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${kb.userId}` 
          };
          followingMap[kb.userId] = false;
        }
      }
      setOwners(ownersMap);
      setIsFollowing(followingMap);
    };
    
    fetchOwners();
  }, [knowledgeBases, currentUser, sourceType]);
  
  const handleNext = () => {
    if (filteredKnowledgeBases && currentIndex < filteredKnowledgeBases.length - 1) {
      setTransitionDirection('down');
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentIndex(prevIndex => prevIndex + 1);
        setTimeout(() => {
          setIsTransitioning(false);
        }, 50);
      }, 200);
    }
  };
  
  const handlePrevious = () => {
    if (currentIndex > 0) {
      setTransitionDirection('up');
      setIsTransitioning(true);
      setTimeout(() => {
        setCurrentIndex(prevIndex => prevIndex - 1);
        setTimeout(() => {
          setIsTransitioning(false);
        }, 50);
      }, 200);
    }
  };
  
  const handleKnowledgeBaseClick = (kbId: number) => {
    navigate(`/knowledge-base/${kbId}`);
  };
  
  // 键盘导航 - 修改空格键为播放/暂停
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'ArrowUp' || event.key === 'ArrowLeft') {
        handlePrevious();
      } else if (event.key === 'ArrowDown' || event.key === 'ArrowRight') {
        // 只使用方向键切换视频
        handleNext();
      } else if (event.key === ' ') {
        // 空格键控制播放/暂停
        setIsPlaying(prevState => !prevState);
        event.preventDefault(); // 防止空格键滚动页面
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [currentIndex, filteredKnowledgeBases]); // eslint-disable-line react-hooks/exhaustive-deps
  
  // 添加鼠标滚轮事件
  useEffect(() => {
    const handleWheel = (event: WheelEvent) => {
      // 检测滚轮方向
      if (event.deltaY > 0) {
        // 向下滚动
        handleNext();
      } else if (event.deltaY < 0) {
        // 向上滚动
        handlePrevious();
      }
      // 防止默认滚动行为
      event.preventDefault();
    };
    
    // 只有在视频区域内滚动时才触发
    const videoElement = videoRef.current;
    if (videoElement) {
      videoElement.addEventListener('wheel', handleWheel, { passive: false });
    }
    
    return () => {
      if (videoElement) {
        videoElement.removeEventListener('wheel', handleWheel);
      }
    };
  }, [currentIndex, filteredKnowledgeBases]); // eslint-disable-line react-hooks/exhaustive-deps
  
  const handleFollowClick = (userId: number, event: React.MouseEvent) => {
    event.stopPropagation();
    setIsFollowing(prev => ({
      ...prev,
      [userId]: !prev[userId]
    }));
  };

  const handleLikeClick = (kbId: number, event: React.MouseEvent) => {
    event.stopPropagation();
    setLikedItems(prev => ({
      ...prev,
      [kbId]: !prev[kbId]
    }));
  };

  const handleBookmarkClick = (kbId: number, event: React.MouseEvent) => {
    event.stopPropagation();
    setBookmarkedItems(prev => ({
      ...prev,
      [kbId]: !prev[kbId]
    }));
  };

  const handleShareClick = (kbId: number, event: React.MouseEvent) => {
    event.stopPropagation();
    // 分享逻辑，这里简单提示一下
    alert(`已复制分享链接: /knowledge-base/${kbId}`);
  };
  
  // 格式化数字函数
  const formatNumber = (num: number): string => {
    if (num >= 10000) {
      return (num / 10000).toFixed(1) + '万';
    }
    return num.toString();
  };
  
  // 添加关闭菜单的事件监听
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (moreOptionsRef.current && !moreOptionsRef.current.contains(event.target as Node)) {
        setShowMoreOptions(false);
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);
  
  // 模拟评论数据
  const mockComments = [
    { id: 1, user: { name: "青秀卖土豆的", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=123" }, content: "一想到你拉的尿冲进下水道和别人的粪在一起就吃醋 😭 😭", time: "11小时前", location: "广西", likes: 299 },
    { id: 2, user: { name: "乐乐乐", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=456" }, content: "dy最完美这一块 /", time: "16小时前", location: "英国", likes: 155, hasAuthorReply: true },
    { id: 3, user: { name: "啊哈", avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=789" }, content: "太棒了！", time: "14小时前", location: "广东", likes: 7, hasImage: true }
  ];
  
  if (isLoading || !knowledgeBases || filteredKnowledgeBases.length === 0) {
    return (
      <div className="h-[calc(100vh-4rem)] bg-black flex items-center justify-center">
        <div className="text-white">加载中...</div>
      </div>
    );
  }
  
  const currentKB = filteredKnowledgeBases[currentIndex];
  const videoSrc = videoContents[currentIndex % videoContents.length];
  const owner = owners[currentKB.userId] || { 
    username: `用户${currentKB.userId}`, 
    avatar: `https://api.dicebear.com/7.x/avataaars/svg?seed=${currentKB.userId}` 
  };
  
  return (
    <div className="h-[calc(100vh-4rem)] bg-black flex items-center">
      <div className="relative h-full w-full" ref={videoRef}>
        {/* 视频区域 */}
        <div className={`relative h-full w-full bg-gray-900 overflow-hidden transition-all duration-300 ${
          isTransitioning 
            ? transitionDirection === 'down' 
              ? 'opacity-0 transform translate-y-10' 
              : 'opacity-0 transform -translate-y-10' 
            : 'opacity-100 transform translate-y-0'
        }`}>
          <img 
            src={videoSrc} 
            alt={currentKB.title}
            className={`w-full h-full object-cover ${!isPlaying ? 'filter brightness-75' : ''}`}
          />
          
          {/* 播放/暂停指示器 */}
          {!isPlaying && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="bg-gray-800/70 rounded-full p-6">
                <svg className="w-16 h-16 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                </svg>
              </div>
            </div>
          )}
          
          {/* 视频控制按钮区域 */}
          <div className="absolute inset-0 flex">
            {/* 上一个视频区域 */}
            <div 
              className="w-1/2 h-full cursor-pointer flex items-center justify-start pl-8 opacity-0 hover:opacity-100 transition-opacity"
              onClick={handlePrevious}
            >
              {currentIndex > 0 && (
                <div className="bg-gray-800/50 rounded-full p-4 transform -translate-x-4 hover:translate-x-0 transition-transform">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
                  </svg>
                </div>
              )}
            </div>
            
            {/* 下一个视频区域 */}
            <div 
              className="w-1/2 h-full cursor-pointer flex items-center justify-end pr-8 opacity-0 hover:opacity-100 transition-opacity"
              onClick={handleNext}
            >
              {filteredKnowledgeBases && currentIndex < filteredKnowledgeBases.length - 1 && (
                <div className="bg-gray-800/50 rounded-full p-4 transform translate-x-4 hover:translate-x-0 transition-transform">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
                  </svg>
                </div>
              )}
            </div>
          </div>
          
          {/* 知识库信息区域 */}
          <div className={`absolute bottom-0 left-0 right-0 p-8 bg-gradient-to-t from-black to-transparent transition-all duration-300 ${
            isTransitioning 
              ? 'opacity-0 transform translate-y-8' 
              : 'opacity-100 transform translate-y-0'
          }`}>
            <div className="max-w-4xl mb-4">
              <h3 
                className="text-white text-3xl font-bold mb-4 cursor-pointer"
                onClick={() => handleKnowledgeBaseClick(currentKB.id)}
              >
                {currentKB.title}
              </h3>
              <p className="text-gray-300 text-lg max-w-2xl">{currentKB.description}</p>
              
              <div className="flex flex-wrap gap-2 mt-4">
                {currentKB.tags.map((tag, idx) => (
                  <span key={idx} className="px-3 py-1 bg-gray-800/50 text-sm rounded-full text-blue-300">
                    {tag}
                  </span>
                ))}
              </div>
            </div>
            
            {/* 用户信息区域 - 只显示@用户名 */}
            <div className="flex items-center mb-8">
              <div className="text-white font-medium text-lg">@{owner.username}</div>
            </div>
          </div>
          
          {/* 右侧操作按钮 - 抖音风格 */}
          <div className={`absolute right-3 bottom-40 flex flex-col items-center space-y-5 transition-all duration-300 ${
            isTransitioning 
              ? 'opacity-0 transform translate-x-8' 
              : 'opacity-100 transform translate-x-0'
          }`}>
            {/* 用户头像及关注按钮 */}
            <div className="flex flex-col items-center">
              <div className="relative">
                <UserAvatar 
                  username={owner.username}
                  avatarSrc={owner.avatar}
                  size="lg"
                  className="w-12 h-12 border-2 border-white" 
                />
                {!isFollowing[currentKB.userId] && (
                  <div 
                    className="absolute -bottom-1.5 left-1/2 transform -translate-x-1/2 h-5 w-5 rounded-full bg-pink-500 text-white flex items-center justify-center cursor-pointer"
                    onClick={(e) => handleFollowClick(currentKB.userId, e)}
                  >
                    <Plus className="h-3 w-3" />
                  </div>
                )}
              </div>
            </div>
            
            {/* 喜欢按钮 */}
            <div className="flex flex-col items-center">
              <Button 
                variant="ghost" 
                size="icon" 
                className={`h-10 w-10 rounded-full bg-white text-black hover:bg-white ${likedItems[currentKB.id] ? 'bg-pink-500 text-white' : ''}`}
                onClick={(e) => handleLikeClick(currentKB.id, e)}
              >
                <Heart className={`h-7 w-7 ${likedItems[currentKB.id] ? 'fill-current' : ''}`} />
              </Button>
              <span className="text-white text-xs mt-1">{formatNumber(likedItems[currentKB.id] ? 144001 : 144000)}</span>
            </div>
            
            {/* 评论按钮 */}
            <div className="flex flex-col items-center">
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-10 w-10 rounded-full bg-white text-black hover:bg-white"
                onClick={() => setShowMessages(!showMessages)}
              >
                <MessageCircle className="h-7 w-7" />
              </Button>
              <span className="text-white text-xs mt-1">1205</span>
            </div>
            
            {/* 收藏按钮 */}
            <div className="flex flex-col items-center">
              <Button 
                variant="ghost" 
                size="icon" 
                className={`h-10 w-10 rounded-full bg-white text-black hover:bg-white ${bookmarkedItems[currentKB.id] ? 'bg-yellow-400 text-black' : ''}`}
                onClick={(e) => handleBookmarkClick(currentKB.id, e)}
              >
                <Bookmark className={`h-7 w-7 ${bookmarkedItems[currentKB.id] ? 'fill-current' : ''}`} />
              </Button>
              <span className="text-white text-xs mt-1">{formatNumber(bookmarkedItems[currentKB.id] ? 21001 : 21000)}</span>
            </div>
            
            {/* 分享按钮 */}
            <div className="flex flex-col items-center">
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-10 w-10 rounded-full bg-white text-black hover:bg-white"
                onClick={(e) => handleShareClick(currentKB.id, e)}
              >
                <Share className="h-7 w-7" />
              </Button>
              <span className="text-white text-xs mt-1">7976</span>
            </div>

            {/* 看相关 */}
            {/* <div className="flex flex-col items-center">
              <Button 
                variant="ghost" 
                size="icon"
                className="h-8 w-14 rounded-full text-white text-xs hover:bg-white/10 font-medium"
              >
                看相关
              </Button>
            </div> */}

            {/* 更多按钮 - 横向三点 */}
            <div className="flex flex-col items-center relative" ref={moreOptionsRef}>
              <Button 
                variant="ghost" 
                size="icon" 
                className="h-8 w-8 rounded-full text-white hover:bg-white/10"
                onClick={() => setShowMoreOptions(!showMoreOptions)}
              >
                <MoreHorizontal className="h-5 w-5" />
              </Button>
              
              {/* 更多选项菜单 */}
              {showMoreOptions && (
                <div className="absolute right-0 bottom-10 bg-[#2A2B31] w-64 rounded-xl overflow-hidden shadow-xl z-50">
                  <div className="p-3 grid grid-cols-4 gap-2">
                    <div className="flex flex-col items-center">
                      <div className="bg-[#434449] rounded-full w-14 h-14 flex items-center justify-center mb-1">
                        <HeartOff className="w-7 h-7 text-white" />
                      </div>
                      <span className="text-white text-xs">不感兴趣</span>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="bg-[#434449] rounded-full w-14 h-14 flex items-center justify-center mb-1">
                        <UserMinus className="w-7 h-7 text-white" />
                      </div>
                      <span className="text-white text-xs">取消关注</span>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="bg-[#434449] rounded-full w-14 h-14 flex items-center justify-center mb-1">
                        <AlertTriangle className="w-7 h-7 text-white" />
                      </div>
                      <span className="text-white text-xs">举报</span>
                    </div>
                    <div className="flex flex-col items-center">
                      <div className="bg-[#434449] rounded-full w-14 h-14 flex items-center justify-center mb-1">
                        <Keyboard className="w-7 h-7 text-white" />
                      </div>
                      <span className="text-white text-xs">快捷键列表</span>
                    </div>
                  </div>
                  <div className="mt-2 border-t border-gray-700 py-3 px-4">
                    <div className="flex items-center">
                      <span className="text-white text-xs">Shake9.创作的原声</span>
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-gray-400 text-xs">2389人使用</span>
                      <Button className="bg-[#FF4D4F] text-white hover:bg-[#FF4D4F]/90 text-xs rounded-full px-4 h-8">
                        收藏
                      </Button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
          
          {/* 当前索引/总数指示器 */}
          <div className="absolute top-8 right-8 bg-gray-800/50 px-4 py-2 rounded-full text-white text-sm">
            {currentIndex + 1}/{filteredKnowledgeBases.length}
          </div>

          {/* 来源标识 */}
          <div className="absolute top-8 left-8 bg-gray-800/50 px-4 py-2 rounded-full text-white text-sm">
            {sourceType === "recommend" && "为您推荐"}
            {sourceType === "following" && "关注更新"}
            {sourceType === "friends" && "好友动态"}
          </div>
          
          {/* 添加操作提示 */}
          <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-gray-800/50 px-6 py-2 rounded-full text-white text-sm flex items-center space-x-4">
            <div className="flex items-center">
              <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 19l-7-7 7-7"></path>
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 5l7 7-7 7"></path>
              </svg>
              <span>滑动鼠标</span>
            </div>
            <div className="w-px h-4 bg-gray-400"></div>
            <div className="flex items-center">
              <span className="inline-block border border-white px-2 py-0.5 text-xs rounded mr-2">↓</span>
              <span>下个视频</span>
            </div>
            <div className="w-px h-4 bg-gray-400"></div>
            <div className="flex items-center">
              <span className="inline-block border border-white px-2 py-0.5 text-xs rounded mr-2">空格</span>
              <span>播放/暂停</span>
            </div>
          </div>
        </div>
      </div>

      {/* 从右侧滑出的评论面板 */}
      <div 
        className={`fixed top-16 right-0 h-[calc(100vh-4rem)] w-[450px] bg-[#2A2B31] shadow-xl z-50 transform transition-transform duration-300 ease-in-out ${
          showMessages ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {/* 顶部标签栏 */}
        <div className="flex border-b border-gray-700">
          <div className="px-6 py-2.5 text-gray-400 hover:text-white cursor-pointer text-sm">TA的作品</div>
          <div className="px-6 py-2.5 text-white cursor-pointer border-b-2 border-pink-500 text-sm">评论</div>
          <div className="px-6 py-2.5 text-gray-400 hover:text-white cursor-pointer text-sm">相关推荐</div>
          <button 
            className="ml-auto pr-4 text-gray-400 hover:text-white"
            onClick={() => setShowMessages(false)}
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
            </svg>
          </button>
        </div>
        
        {/* 评论计数 */}
        <div className="px-4 py-2 text-white text-sm">全部评论(293)</div>
        
        {/* 评论列表 */}
        <div className="overflow-y-auto h-[calc(100%-110px)] px-4">
          {mockComments.map((comment) => (
            <div key={comment.id} className="py-3 border-b border-gray-700">
              <div className="flex">
                <img 
                  src={comment.user.avatar} 
                  alt={comment.user.name} 
                  className="w-9 h-9 rounded-full mr-3" 
                />
                <div className="flex-1">
                  <div className="text-white text-sm mb-1">{comment.user.name}</div>
                  <div className="text-white text-sm mb-2">{comment.content}</div>
                  
                  {comment.hasImage && (
                    <div className="mb-2">
                      <img 
                        src="https://api.dicebear.com/7.x/avataaars/svg?seed=girl1" 
                        alt="评论图片" 
                        className="w-32 h-32 rounded-lg object-cover" 
                      />
                    </div>
                  )}
                  
                  <div className="text-gray-500 text-xs flex items-center">
                    <span>{comment.time} · {comment.location}</span>
                  </div>
                  
                  {comment.hasAuthorReply && (
                    <div className="mt-1 px-2 py-0.5 bg-gray-700 text-gray-400 text-xs inline-block rounded">
                      作者回复过
                    </div>
                  )}
                </div>
              </div>
              
              {/* 评论操作 */}
              <div className="mt-2 flex items-center pl-12 text-gray-500 text-xs">
                <button className="flex items-center mr-6">
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"></path>
                  </svg>
                  回复
                </button>
                <button className="flex items-center mr-6">
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"></path>
                  </svg>
                  分享
                </button>
                <button className="flex items-center">
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path>
                  </svg>
                  {comment.likes}
                </button>
              </div>
              
              {/* 展开回复 */}
              {comment.id !== 3 && (
                <div className="mt-2 pl-12 text-gray-500 flex items-center cursor-pointer">
                  <div className="h-px bg-gray-700 flex-grow mr-2"></div>
                  <span className="text-xs">展开{comment.id === 1 ? '27' : '1'}条回复</span>
                  <svg className="w-3 h-3 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7"></path>
                  </svg>
                </div>
              )}
            </div>
          ))}
        </div>
        
        {/* 底部评论框 */}
        <div className="absolute bottom-0 left-0 right-0 bg-[#1F2026] p-2.5 flex items-center">
          <input 
            type="text" 
            placeholder="说点什么..." 
            className="flex-1 bg-[#434449] border-none rounded-full px-3.5 py-1.5 text-white text-xs focus:outline-none" 
          />
          <Button className="ml-2 rounded-full h-7 px-3.5 bg-[#FF4D4F] hover:bg-[#FF4D4F]/90 text-xs">
            发送
          </Button>
        </div>
      </div>
    </div>
  );
};

export default KnowledgeBaseVideoFeed; 