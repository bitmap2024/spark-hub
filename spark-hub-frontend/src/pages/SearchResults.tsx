import React, { useState, useEffect } from "react";
import { useLocation } from "react-router-dom";
import AppLayout from "@/components/layouts/AppLayout";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Badge } from "@/components/ui/badge";
import { Clock, Eye, MessageSquare, ThumbsUp, User, Book, FileText, Video } from "lucide-react";

// 定义搜索结果类型
interface SearchResultItem {
  id: string;
  title: string;
  description: string;
  type: "knowledge" | "article" | "user" | "video";
  imageUrl?: string;
  username?: string;
  avatar?: string;
  date?: string;
  views?: number;
  likes?: number;
  followers?: number;
  comments?: number;
  verified?: boolean;
  tags?: string[];
}

const MOCK_DATA: { [key: string]: SearchResultItem[] } = {
  "烧烤": [
    {
      id: "1",
      title: "破路边烧烤工厂知识库",
      description: "关于破路边烧烤工厂的所有知识和信息，包括菜单、历史、文化和特色",
      type: "knowledge",
      imageUrl: "https://images.unsplash.com/photo-1555939594-58d7cb561ad1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1287&q=80",
      date: "2023-10-15",
      views: 1200,
      tags: ["烧烤", "知识库", "美食"]
    },
    {
      id: "2",
      title: "如何制作完美的烧烤",
      description: "专业烧烤技巧和秘诀分享，从腌制到烤制的全过程指南",
      type: "article",
      imageUrl: "https://images.unsplash.com/photo-1534797258760-1bd2cc95a5bd?ixlib=rb-4.0.3&auto=format&fit=crop&w=1287&q=80",
      username: "烧烤达人",
      date: "2023-11-20",
      likes: 356,
      comments: 42,
      tags: ["烹饪", "技巧", "烧烤"]
    },
    {
      id: "3",
      title: "烧烤店跳舞小姐姐",
      description: "美女跳舞视频，超火爆表演，点燃现场氛围",
      type: "video",
      imageUrl: "https://images.unsplash.com/photo-1517457373958-b7bdd4587205?ixlib=rb-4.0.3&auto=format&fit=crop&w=1369&q=80",
      username: "破路边烧烤工厂",
      views: 5689,
      likes: 892,
      comments: 156,
      tags: ["舞蹈", "表演", "烧烤店"]
    },
    {
      id: "4",
      type: "user",
      title: "破路边烧烤工厂官方账号",
      description: "官方认证账号，分享最新活动和美食",
      avatar: "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?ixlib=rb-4.0.3&auto=format&fit=crop&w=1160&q=80",
      username: "破路边烧烤工厂",
      followers: 12540,
      verified: true
    },
    {
      id: "5",
      type: "user",
      title: "烧烤达人",
      description: "烧烤爱好者，美食博主，分享烧烤技巧和食谱",
      avatar: "https://images.unsplash.com/photo-1492447166138-50c3889fccb1?ixlib=rb-4.0.3&auto=format&fit=crop&w=1287&q=80",
      username: "烧烤达人",
      followers: 4230
    },
    {
      id: "6",
      title: "破路边烧烤工厂跳舞女孩苏州站",
      description: "苏州破路边烧烤工厂特色表演，顾客互动，现场气氛火爆",
      type: "video",
      imageUrl: "https://images.unsplash.com/photo-1572635196237-14b3f281503f?ixlib=rb-4.0.3&auto=format&fit=crop&w=1280&q=80",
      username: "苏州美食探店",
      views: 3462,
      likes: 721,
      date: "2023-12-01",
      tags: ["苏州", "烧烤", "表演"]
    },
    {
      id: "7",
      title: "烧烤店经营秘籍",
      description: "从选址到装修，从采购到营销，全方位解析烧烤店经营要点",
      type: "article",
      imageUrl: "https://images.unsplash.com/photo-1460306855393-0410f61241c7?ixlib=rb-4.0.3&auto=format&fit=crop&w=1352&q=80",
      username: "餐饮商业顾问",
      date: "2023-08-12",
      likes: 289,
      views: 4521,
      tags: ["创业", "经营", "烧烤店"]
    },
    {
      id: "8",
      title: "街边烧烤大排档食品安全指南",
      description: "如何判断街边烧烤的食品安全，避免食品安全风险",
      type: "knowledge",
      imageUrl: "https://images.unsplash.com/photo-1560035285-64808ba47bda?ixlib=rb-4.0.3&auto=format&fit=crop&w=1287&q=80",
      date: "2023-09-25",
      views: 2360,
      likes: 452,
      tags: ["食品安全", "烧烤", "健康"]
    }
  ],
  "跳舞女孩": [
    {
      id: "1",
      title: "破路边烧烤工厂跳舞女孩合集",
      description: "全国各地破路边烧烤工厂跳舞女孩精彩表演集锦",
      type: "video",
      imageUrl: "https://images.unsplash.com/photo-1517457373958-b7bdd4587205?ixlib=rb-4.0.3&auto=format&fit=crop&w=1369&q=80",
      username: "破路边烧烤工厂",
      views: 12589,
      likes: 2892,
      comments: 356,
      tags: ["舞蹈", "表演", "烧烤店"]
    },
    {
      id: "2",
      title: "烧烤店跳舞小姐姐采访实录",
      description: "独家专访：烧烤店跳舞女孩的工作日常与生活故事",
      type: "article",
      imageUrl: "https://images.unsplash.com/photo-1536011269729-f3168c7b4b47?ixlib=rb-4.0.3&auto=format&fit=crop&w=1289&q=80",
      username: "都市生活周刊",
      date: "2023-10-18",
      likes: 876,
      comments: 134,
      tags: ["专访", "舞者", "烧烤店"]
    },
    {
      id: "3",
      type: "user",
      title: "小甜甜舞蹈",
      description: "破路边烧烤工厂金牌舞者，接受商演及活动邀约",
      avatar: "https://images.unsplash.com/photo-1534528741775-53994a69daeb?ixlib=rb-4.0.3&auto=format&fit=crop&w=1364&q=80",
      username: "小甜甜舞蹈",
      followers: 8764,
      verified: true
    },
    {
      id: "4",
      title: "跳舞女孩背后的故事",
      description: "揭秘烧烤店跳舞女孩行业现状，从业者心声与梦想",
      type: "knowledge",
      imageUrl: "https://images.unsplash.com/photo-1516802273409-68526ee1bdd6?ixlib=rb-4.0.3&auto=format&fit=crop&w=1416&q=80",
      date: "2023-11-05",
      views: 6520,
      tags: ["职业", "舞蹈", "生活"]
    }
  ],
  "知识库": [
    {
      id: "1",
      title: "美食知识库",
      description: "全球各地美食文化、制作工艺和历史渊源的综合知识库",
      type: "knowledge",
      imageUrl: "https://images.unsplash.com/photo-1498837167922-ddd27525d352?ixlib=rb-4.0.3&auto=format&fit=crop&w=2340&q=80",
      date: "2023-08-10",
      views: 8700,
      tags: ["美食", "文化", "知识库"]
    },
    {
      id: "2",
      title: "烹饪技巧知识库",
      description: "从入门到精通的烹饪技巧百科全书，包含各类食材处理和烹饪方法",
      type: "knowledge",
      imageUrl: "https://images.unsplash.com/photo-1556911220-e15b29be8c8f?ixlib=rb-4.0.3&auto=format&fit=crop&w=2340&q=80",
      date: "2023-06-22",
      views: 6300,
      tags: ["烹饪", "技巧", "食材"]
    },
    {
      id: "3",
      title: "创建和管理知识库指南",
      description: "如何建立高效实用的企业或个人知识库，组织和分享信息的最佳实践",
      type: "article",
      imageUrl: "https://images.unsplash.com/photo-1507842217343-583bb7270b66?ixlib=rb-4.0.3&auto=format&fit=crop&w=2340&q=80",
      username: "知识管理专家",
      date: "2023-07-15",
      likes: 420,
      comments: 56,
      tags: ["知识管理", "效率", "信息组织"]
    }
  ]
};

const SearchResults = () => {
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);
  const query = searchParams.get("q") || "";
  const [activeTab, setActiveTab] = useState("all");
  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState<SearchResultItem[]>([]);

  useEffect(() => {
    // 模拟搜索API调用
    const fetchSearchResults = async () => {
      setLoading(true);
      try {
        // 模拟数据
        setTimeout(() => {
          // 根据查询返回不同的模拟数据
          const mockResults = MOCK_DATA[query] || MOCK_DATA["烧烤"];
          setResults(mockResults);
          setLoading(false);
        }, 1000);
      } catch (error) {
        console.error("搜索失败:", error);
        setLoading(false);
      }
    };

    fetchSearchResults();
  }, [query, activeTab]);

  const renderItem = (item: SearchResultItem) => {
    switch (item.type) {
      case "knowledge":
        return (
          <Card key={item.id} className="mb-4 hover:shadow-md transition-shadow cursor-pointer">
            <CardContent className="p-4 flex">
              {item.imageUrl && (
                <div className="mr-4 w-28 h-28 overflow-hidden rounded-md">
                  <img src={item.imageUrl} alt={item.title} className="w-full h-full object-cover" />
                </div>
              )}
              <div className="flex-1">
                <div className="flex items-center mb-1">
                  <Book className="h-4 w-4 mr-1 text-blue-400" />
                  <span className="text-xs text-blue-400">知识库</span>
                </div>
                <h3 className="text-lg font-semibold">{item.title}</h3>
                <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                <div className="flex items-center mt-2 text-xs text-gray-400">
                  {item.date && (
                    <span className="flex items-center mr-3">
                      <Clock className="h-3 w-3 mr-1" />
                      {item.date}
                    </span>
                  )}
                  {item.views !== undefined && (
                    <span className="flex items-center mr-3">
                      <Eye className="h-3 w-3 mr-1" />
                      {item.views} 次阅读
                    </span>
                  )}
                </div>
                {item.tags && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {item.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="text-xs bg-gray-700 hover:bg-gray-600">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        );
      
      case "article":
        return (
          <Card key={item.id} className="mb-4 hover:shadow-md transition-shadow cursor-pointer">
            <CardContent className="p-4 flex">
              {item.imageUrl && (
                <div className="mr-4 w-28 h-28 overflow-hidden rounded-md">
                  <img src={item.imageUrl} alt={item.title} className="w-full h-full object-cover" />
                </div>
              )}
              <div className="flex-1">
                <div className="flex items-center mb-1">
                  <FileText className="h-4 w-4 mr-1 text-green-400" />
                  <span className="text-xs text-green-400">文章</span>
                </div>
                <h3 className="text-lg font-semibold">{item.title}</h3>
                <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                <div className="flex items-center mt-2 text-xs text-gray-400">
                  {item.username && (
                    <span className="flex items-center mr-3">
                      <User className="h-3 w-3 mr-1" />
                      {item.username}
                    </span>
                  )}
                  {item.date && (
                    <span className="flex items-center mr-3">
                      <Clock className="h-3 w-3 mr-1" />
                      {item.date}
                    </span>
                  )}
                  {item.likes !== undefined && (
                    <span className="flex items-center mr-3">
                      <ThumbsUp className="h-3 w-3 mr-1" />
                      {item.likes}
                    </span>
                  )}
                  {item.comments !== undefined && (
                    <span className="flex items-center">
                      <MessageSquare className="h-3 w-3 mr-1" />
                      {item.comments}
                    </span>
                  )}
                </div>
                {item.tags && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {item.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="text-xs bg-gray-700 hover:bg-gray-600">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        );
      
      case "video":
        return (
          <Card key={item.id} className="mb-4 hover:shadow-md transition-shadow cursor-pointer">
            <CardContent className="p-4 flex">
              {item.imageUrl && (
                <div className="mr-4 w-36 h-28 overflow-hidden rounded-md relative">
                  <img src={item.imageUrl} alt={item.title} className="w-full h-full object-cover" />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-10 h-10 rounded-full bg-black bg-opacity-50 flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="white">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    </div>
                  </div>
                </div>
              )}
              <div className="flex-1">
                <div className="flex items-center mb-1">
                  <Video className="h-4 w-4 mr-1 text-red-400" />
                  <span className="text-xs text-red-400">视频</span>
                </div>
                <h3 className="text-lg font-semibold">{item.title}</h3>
                <p className="text-sm text-gray-500 mt-1">{item.description}</p>
                <div className="flex items-center mt-2 text-xs text-gray-400">
                  {item.username && (
                    <span className="flex items-center mr-3">
                      <User className="h-3 w-3 mr-1" />
                      {item.username}
                    </span>
                  )}
                  {item.views !== undefined && (
                    <span className="flex items-center mr-3">
                      <Eye className="h-3 w-3 mr-1" />
                      {item.views} 次播放
                    </span>
                  )}
                  {item.likes !== undefined && (
                    <span className="flex items-center mr-3">
                      <ThumbsUp className="h-3 w-3 mr-1" />
                      {item.likes}
                    </span>
                  )}
                  {item.comments !== undefined && (
                    <span className="flex items-center">
                      <MessageSquare className="h-3 w-3 mr-1" />
                      {item.comments}
                    </span>
                  )}
                </div>
                {item.tags && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {item.tags.map(tag => (
                      <Badge key={tag} variant="secondary" className="text-xs bg-gray-700 hover:bg-gray-600">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        );
      
      case "user":
        return (
          <Card key={item.id} className="mb-4 hover:shadow-md transition-shadow cursor-pointer">
            <CardContent className="p-4 flex items-center">
              <Avatar className="h-14 w-14 mr-4">
                <AvatarImage src={item.avatar} alt={item.username} />
                <AvatarFallback>{item.username?.substring(0, 2)}</AvatarFallback>
              </Avatar>
              <div className="flex-1">
                <div className="flex items-center">
                  <h3 className="text-lg font-semibold">{item.title}</h3>
                  {item.verified && (
                    <Badge className="ml-2 bg-blue-500 hover:bg-blue-600" variant="secondary">
                      <svg className="h-3 w-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 0 1 0 1.414l-8 8a1 1 0 0 1-1.414 0l-4-4a1 1 0 0 1 1.414-1.414L8 12.586l7.293-7.293a1 1 0 0 1 1.414 0z" clipRule="evenodd" />
                      </svg>
                      认证
                    </Badge>
                  )}
                </div>
                <p className="text-sm text-gray-500">{item.description}</p>
                {item.followers !== undefined && (
                  <p className="text-xs text-gray-400 mt-1">
                    <span className="flex items-center">
                      <User className="h-3 w-3 mr-1" />
                      {item.followers} 位关注者
                    </span>
                  </p>
                )}
              </div>
              <Button variant="outline" size="sm" className="bg-[#fe2c55] hover:bg-[#fe2c55]/90 text-white border-none">关注</Button>
            </CardContent>
          </Card>
        );
      
      default:
        return null;
    }
  };

  const renderSkeletons = () => {
    return Array(5).fill(0).map((_, index) => (
      <Card key={index} className="mb-4">
        <CardContent className="p-4 flex">
          <Skeleton className="mr-4 w-28 h-28 rounded-md" />
          <div className="flex-1">
            <Skeleton className="h-4 w-24 mb-2" />
            <Skeleton className="h-6 w-3/4 mb-2" />
            <Skeleton className="h-4 w-full mb-2" />
            <Skeleton className="h-4 w-2/3 mb-2" />
            <div className="flex gap-2 mt-1">
              <Skeleton className="h-4 w-16" />
              <Skeleton className="h-4 w-16" />
              <Skeleton className="h-4 w-16" />
            </div>
          </div>
        </CardContent>
      </Card>
    ));
  };

  const countByType = (type: string) => {
    if (type === "all") return results.length;
    return results.filter(item => item.type === type).length;
  };

  return (
    <AppLayout>
      <div className="container mx-auto py-6 px-4">
        <div className="mb-6">
          <h1 className="text-2xl font-bold mb-2">"{query}" 的搜索结果</h1>
          <p className="text-gray-500">找到约 {results.length} 条结果</p>
        </div>

        <Tabs defaultValue="all" value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="mb-6">
            <TabsTrigger value="all">全部 ({countByType("all")})</TabsTrigger>
            <TabsTrigger value="knowledge">知识库 ({countByType("knowledge")})</TabsTrigger>
            <TabsTrigger value="article">文章 ({countByType("article")})</TabsTrigger>
            <TabsTrigger value="video">视频 ({countByType("video")})</TabsTrigger>
            <TabsTrigger value="user">用户 ({countByType("user")})</TabsTrigger>
          </TabsList>
          
          <TabsContent value="all" className="mt-0">
            {loading ? renderSkeletons() : results.map(renderItem)}
          </TabsContent>
          
          <TabsContent value="knowledge" className="mt-0">
            {loading ? 
              renderSkeletons() : 
              results.filter(item => item.type === "knowledge").map(renderItem)
            }
          </TabsContent>
          
          <TabsContent value="article" className="mt-0">
            {loading ? 
              renderSkeletons() : 
              results.filter(item => item.type === "article").map(renderItem)
            }
          </TabsContent>
          
          <TabsContent value="video" className="mt-0">
            {loading ? 
              renderSkeletons() : 
              results.filter(item => item.type === "video").map(renderItem)
            }
          </TabsContent>
          
          <TabsContent value="user" className="mt-0">
            {loading ? 
              renderSkeletons() : 
              results.filter(item => item.type === "user").map(renderItem)
            }
          </TabsContent>
        </Tabs>
      </div>
    </AppLayout>
  );
};

export default SearchResults; 