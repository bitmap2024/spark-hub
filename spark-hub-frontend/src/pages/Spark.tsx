import React, { useState, useCallback } from "react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { Link, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Paperclip, Import, Globe, ChevronUp, Search, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import { useCurrentUser, useSearchPapers, useCreateKnowledgeBase } from "@/lib/api";
import { Paper } from "@/lib/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import { useSearchHandler } from "@/lib/navigation";

// 模拟项目类型
interface ProjectType {
  id: string;
  icon: string;
  title: string;
}

const Spark: React.FC = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [searchInput, setSearchInput] = useState("");
  const [searchTrigger, setSearchTrigger] = useState("");
  const [searchType, setSearchType] = useState("全网");
  const navigate = useNavigate();
  const { toast } = useToast();
  const { data: currentUser } = useCurrentUser();
  
  // 使用React Query hooks
  const { 
    data: searchResults = [], 
    isLoading: isSearching, 
    error: searchError
  } = useSearchPapers(searchTrigger);
  
  const { 
    mutate: createKnowledgeBase, 
    isPending: isCreating 
  } = useCreateKnowledgeBase();
  
  const handleGlobalSearch = useSearchHandler();
  
  // 创建一个本地搜索函数
  const handleLocalSearch = useCallback(() => {
    if (!searchInput.trim()) {
      toast({
        title: "请输入搜索内容",
        description: "请输入关键词或研究领域描述",
        variant: "destructive"
      });
      return;
    }
    
    setSearchTrigger(searchInput);
  }, [searchInput, toast]);
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  const handleCreateKnowledgeBase = useCallback(() => {
    if (!currentUser) {
      setIsLoginOpen(true);
      return;
    }
    
    if (searchResults.length === 0) {
      toast({
        title: "创建失败",
        description: "请先搜索并获取论文",
        variant: "destructive"
      });
      return;
    }
    
    // 从搜索词生成知识库标题和描述
    const title = `${searchInput}研究集锦`;
    const description = `关于${searchInput}的学术研究和前沿探索`;
    
    // 从搜索结果提取标签
    const tags = searchInput.split(/\s+/).filter(tag => tag.length > 1);
    
    createKnowledgeBase(
      {
        title,
        description,
        papers: searchResults,
        tags
      },
      {
        onSuccess: () => {
          toast({
            title: "创建成功",
            description: "知识库已成功创建",
          });
          
          // 跳转到用户主页
          navigate(`/user/${currentUser.username}`);
        },
        onError: (error) => {
          toast({
            title: "创建失败",
            description: "创建知识库时发生错误，请稍后再试",
            variant: "destructive"
          });
        }
      }
    );
  }, [searchResults, searchInput, currentUser, createKnowledgeBase, toast, navigate]);
  
  // 如果搜索出错，显示错误
  React.useEffect(() => {
    if (searchError) {
      toast({
        title: "搜索失败",
        description: "未能获取相关论文，请稍后再试",
        variant: "destructive"
      });
    }
  }, [searchError, toast]);
  
  // 模拟项目类型数据
  const projectTypes: ProjectType[] = [
    { id: "weather", icon: "⏱️", title: "知识探索" },
    { id: "habit", icon: "📅", title: "论文分析" },
    { id: "website", icon: "🖥️", title: "个人知识库" },
    { id: "editor", icon: "📝", title: "研究笔记" },
  ];
  
  return (
    <div className="min-h-screen bg-[#121212]">
      <Header 
        onLoginClick={() => setIsLoginOpen(true)} 
        onSearch={handleGlobalSearch}
      />
      <LeftSidebar />
      
      {/* 主体内容区域 */}
      <div className="ml-64 mt-16 min-h-[calc(100vh-4rem)] bg-gradient-to-b from-[#1a2336] via-[#2a2b59] to-[#b9366c] p-8 flex flex-col items-center">
        <div className="max-w-4xl w-full mx-auto text-center mt-16 mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            创建知识库 <span className="text-transparent bg-clip-text bg-gradient-to-r from-pink-500 to-blue-500">♥ Spark</span>
          </h1>
          <p className="text-xl text-white/80 mb-16">
            从想法到知识库，与您的个人AI助手一起探索学术前沿
          </p>
          
          {/* 搜索框 */}
          <div className="relative w-full mb-16 bg-[#1e1e1e]/80 rounded-xl p-1 border border-gray-800 shadow-xl">
            <div className="p-4 text-left">
              <textarea
                placeholder="开始创建知识库，描述您的研究领域、兴趣或想法..."
                className="w-full bg-transparent text-white border-none focus:outline-none text-lg min-h-[140px] resize-none"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
              />
              
              <div className="flex items-center justify-between mt-4 pt-4 border-t border-gray-700">
                <div className="flex items-center space-x-4">
                  <Select
                    value={searchType}
                    onValueChange={setSearchType}
                  >
                    <SelectTrigger className="w-[120px] rounded-lg border-gray-700 text-gray-300 bg-transparent">
                      <SelectValue placeholder="搜索范围" />
                    </SelectTrigger>
                    <SelectContent className="bg-[#1e1e1e] border-gray-700 text-gray-300">
                      <SelectItem value="全网">全网</SelectItem>
                      <SelectItem value="学术检索">学术检索</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                
                <div className="flex items-center gap-2">
                  <Button 
                    variant="default" 
                    className="bg-gradient-to-r from-blue-600 to-pink-600 text-white rounded-lg px-6"
                    onClick={handleLocalSearch}
                    disabled={isSearching}
                  >
                    {isSearching ? (
                      <>
                        <Loader2 size={16} className="mr-2 animate-spin" />
                        <span>检索中...</span>
                      </>
                    ) : (
                      <>
                        <Search size={16} className="mr-2" />
                        <span>检索</span>
                      </>
                    )}
                  </Button>
                </div>
              </div>
            </div>
          </div>
          
          {/* 搜索结果展示 */}
          {searchResults.length > 0 && (
            <div className="bg-[#1e1e1e]/80 rounded-xl p-6 mb-8 border border-gray-800 shadow-xl text-left">
              <h2 className="text-xl font-semibold text-white mb-4">搜索结果 ({searchResults.length})</h2>
              
              <div className="space-y-6">
                {searchResults.map((paper) => (
                  <div key={paper.id} className="border-b border-gray-700 pb-4 last:border-0">
                    <h3 className="text-lg font-medium text-white">{paper.title}</h3>
                    <p className="text-sm text-gray-400 mt-1">
                      {paper.authors.join(', ')} • {paper.publishDate}
                    </p>
                    <p className="text-gray-300 mt-2 text-sm">
                      {paper.abstract.length > 200 
                        ? `${paper.abstract.substring(0, 200)}...` 
                        : paper.abstract
                      }
                    </p>
                    {paper.doi && (
                      <p className="text-xs text-blue-400 mt-2">
                        DOI: {paper.doi}
                      </p>
                    )}
                  </div>
                ))}
              </div>
              
              <div className="mt-6 text-center">
                <Button
                  variant="default"
                  className="bg-gradient-to-r from-blue-600 to-pink-600 text-white rounded-lg px-8 py-2"
                  onClick={handleCreateKnowledgeBase}
                  disabled={isCreating}
                >
                  {isCreating ? (
                    <>
                      <Loader2 size={16} className="mr-2 animate-spin" />
                      <span>创建中...</span>
                    </>
                  ) : (
                    <span>创建知识库</span>
                  )}
                </Button>
              </div>
            </div>
          )}
          
          {/* 模板选择 */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 w-full mt-4">
            {projectTypes.map((type) => (
              <Button
                key={type.id}
                variant="outline"
                className="flex flex-col items-center justify-center h-24 bg-[#1e1e1e]/50 border-gray-700 hover:bg-[#1e1e1e]/80 rounded-xl"
              >
                <span className="text-2xl mb-2">{type.icon}</span>
                <span className="text-white">{type.title}</span>
              </Button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Spark; 