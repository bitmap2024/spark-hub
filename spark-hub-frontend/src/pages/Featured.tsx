import React, { useState } from "react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { useNavigate, Link } from "react-router-dom";
import { useAllKnowledgeBases } from "@/lib/api";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { useSearchHandler } from "@/lib/navigation";

// 模拟精选知识库分类
const categories = [
  { id: "all", name: "全部" },
  { id: "ai", name: "人工智能" },
  { id: "cv", name: "计算机视觉" },
  { id: "nlp", name: "自然语言处理" },
  { id: "rl", name: "强化学习" },
  { id: "ml", name: "机器学习" }
];

const Featured: React.FC = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const { data: knowledgeBases, isLoading } = useAllKnowledgeBases();
  const navigate = useNavigate();
  const [activeCategory, setActiveCategory] = useState("all");
  const handleSearch = useSearchHandler();
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  // 根据分类筛选知识库
  const getFilteredKnowledgeBases = () => {
    if (!knowledgeBases) return [];
    
    if (activeCategory === "all") {
      return knowledgeBases;
    }
    
    // 根据标签筛选
    return knowledgeBases.filter(kb => {
      const lowercaseTags = kb.tags.map(tag => tag.toLowerCase());
      
      switch (activeCategory) {
        case "ai":
          return lowercaseTags.some(tag => 
            tag.includes("人工智能") || 
            tag.includes("ai") || 
            tag.includes("artificial intelligence"));
        case "cv":
          return lowercaseTags.some(tag => 
            tag.includes("计算机视觉") || 
            tag.includes("computer vision") || 
            tag.includes("图像"));
        case "nlp":
          return lowercaseTags.some(tag => 
            tag.includes("自然语言处理") || 
            tag.includes("nlp") || 
            tag.includes("natural language"));
        case "rl":
          return lowercaseTags.some(tag => 
            tag.includes("强化学习") || 
            tag.includes("reinforcement") || 
            tag.includes("rl"));
        case "ml":
          return lowercaseTags.some(tag => 
            tag.includes("机器学习") || 
            tag.includes("machine learning") || 
            tag.includes("ml"));
        default:
          return true;
      }
    });
  };
  
  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white">加载中...</div>
      </div>
    );
  }
  
  const filteredKnowledgeBases = getFilteredKnowledgeBases();
  
  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onLoginClick={handleLoginClick} onSearch={handleSearch} />
      <LeftSidebar />
      
      {/* 主体内容区域，右侧主区域布局 */}
      <div className="ml-64 mt-16 p-6">
        <div className="mb-6">
          <h1 className="text-2xl font-bold text-white mb-4">精选知识库</h1>
          <p className="text-gray-400">发现高质量的学术知识库，探索前沿研究成果</p>
        </div>
        
        {/* 分类选项卡 */}
        <div className="mb-8">
          <div className="flex items-center space-x-2 overflow-x-auto pb-2">
            {categories.map(category => (
              <Button 
                key={category.id}
                variant={activeCategory === category.id ? "default" : "outline"}
                className="rounded-full text-sm"
                onClick={() => setActiveCategory(category.id)}
              >
                {category.name}
              </Button>
            ))}
          </div>
        </div>
        
        {/* 知识库网格 */}
        {filteredKnowledgeBases.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {filteredKnowledgeBases.map((kb) => (
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
                      <div className="flex items-center text-purple-500">
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                        </svg>
                        <span>{kb.papers.length}</span>
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-sm text-gray-300 mb-4 line-clamp-2">{kb.description}</p>
                  
                  {/* 用户信息区域 */}
                  <div className="flex items-center mb-3">
                    <div className="w-6 h-6 rounded-full bg-blue-500 overflow-hidden mr-2">
                      <img 
                        src={`https://api.dicebear.com/7.x/avataaars/svg?seed=${kb.userId}`} 
                        alt="用户头像" 
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="text-gray-400 text-xs">
                      用户{kb.userId}
                    </div>
                  </div>
                  
                  <div className="flex items-center mb-3 text-xs text-gray-400">
                    <div className="mr-4 whitespace-nowrap">更新于: {new Date(kb.updatedAt).toLocaleDateString('zh-CN')}</div>
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
            <p className="mt-4">暂无内容<br/>该分类下还没有知识库~</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Featured; 