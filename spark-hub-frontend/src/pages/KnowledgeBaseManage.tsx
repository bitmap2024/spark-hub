import React, { useState } from "react";
import { useParams, useNavigate, Link } from "react-router-dom";
import Header from "@/components/Header";
import { useKnowledgeBase, useUser, useCurrentUser } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { 
  ChevronLeft,
  Database,
  FileText,
  Users,
  MessageCircle,
  Plus,
  Search,
  X,
  Edit,
  Trash2,
  Info,
  PanelLeft,
  PanelRight,
  ChevronsLeft,
  ChevronsRight
} from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogFooter 
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
// import { useSearchHandler } from "@/lib/navigation";

const KnowledgeBaseManage: React.FC = () => {
  const { kbId } = useParams<{ kbId: string }>();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedTab, setSelectedTab] = useState<"papers" | "issues" | "members">("papers");
  const [isAddPaperOpen, setIsAddPaperOpen] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [selectedItem, setSelectedItem] = useState<number | null>(null);
  
  // 添加侧边栏状态
  const [leftSidebarVisible, setLeftSidebarVisible] = useState(true);
  const [rightSidebarVisible, setRightSidebarVisible] = useState(true);
  
  // 获取知识库数据
  const { data: kbData, isLoading: isKbLoading } = useKnowledgeBase(parseInt(kbId || "0"));
  
  // 获取知识库所有者数据
  const { data: ownerData } = useUser(kbData?.userId || 0);
  
  // 获取当前用户数据
  const { data: currentUser } = useCurrentUser();
  
  // 判断当前用户是否是知识库所有者
  const isCurrentUserOwner = currentUser?.id === kbData?.userId;
  
  // 简化的搜索处理函数
  const handleSearch = (query: string) => {
    if (query && query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };
  
  // 处理返回按钮点击
  const handleBackClick = () => {
    navigate(`/knowledge-base/${kbId}`);
  };
  
  // 处理添加论文
  const handleAddPaper = () => {
    try {
      // 在实际应用中，这里应该调用API添加论文
      setIsAddPaperOpen(false);
      toast({
        title: "添加成功",
        description: "论文已添加到知识库",
      });
    } catch (error) {
      console.error("添加论文失败", error);
      toast({
        title: "添加失败",
        description: "添加论文时发生错误",
        variant: "destructive"
      });
    }
  };
  
  // 处理删除项目
  const handleDeleteItem = () => {
    try {
      // 在实际应用中，这里应该调用API删除论文/议题/成员
      setIsDeleteDialogOpen(false);
      toast({
        title: "删除成功",
        description: "项目已从知识库中删除",
      });
    } catch (error) {
      console.error("删除项目失败", error);
      toast({
        title: "删除失败",
        description: "删除项目时发生错误",
        variant: "destructive"
      });
    }
  };
  
  // 处理搜索输入变化
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
  };
  
  // 处理搜索提交
  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // 在实际应用中，这里应该根据搜索词过滤列表
    handleSearch(searchQuery);
  };
  
  if (isKbLoading) {
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white">加载中...</div>
      </div>
    );
  }
  
  if (!kbData) {
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white">知识库不存在或已被删除</div>
      </div>
    );
  }
  
  // 如果当前用户不是所有者，重定向回知识库详情页
  if (!isCurrentUserOwner) {
    navigate(`/knowledge-base/${kbId}`);
    return null;
  }
  
  return (
    <div className="min-h-screen bg-[#121212] flex flex-col">
      <Header onSearch={handleSearch} />
      
      <div className="flex flex-1 pt-16 relative">
        {/* 左侧边栏切换按钮 */}
        <div className={`fixed top-20 ${leftSidebarVisible ? 'left-[296px]' : 'left-2'} z-30 transition-all duration-300`}>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setLeftSidebarVisible(!leftSidebarVisible)}
            className="bg-gray-800/50 hover:bg-gray-700"
          >
            {leftSidebarVisible ? <ChevronsLeft className="h-4 w-4" /> : <ChevronsRight className="h-4 w-4" />}
          </Button>
        </div>
        
        {/* 左侧边栏 - 资源区域 */}
        <div 
          className={`fixed left-0 top-16 h-[calc(100vh-64px)] z-20 transition-all duration-300 ease-in-out ${
            leftSidebarVisible ? 'w-80 translate-x-0' : 'w-0 -translate-x-full'
          } border-r border-gray-800 flex flex-col bg-[#121212] overflow-hidden`}
        >
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <h2 className="text-white font-medium">知识来源</h2>
            <button className="text-gray-400 hover:text-white">
              <Database className="h-5 w-5" />
            </button>
          </div>
          
          <div className="p-2 overflow-y-auto flex-1">
            <div className="flex space-x-2 mb-2">
              <Button variant="outline" className="flex-1 bg-transparent border-gray-700">
                <Plus className="h-4 w-4 mr-2" />
                添加
              </Button>
              <Button variant="outline" className="flex-1 bg-transparent border-gray-700">
                <Search className="h-4 w-4 mr-2" />
                搜索
              </Button>
            </div>
            
            <div className="mt-4">
              <div className="flex items-center justify-between text-gray-400 text-sm py-2 px-2">
                <span>选择所有来源</span>
                <input type="checkbox" className="rounded bg-gray-700 border-gray-600" />
              </div>
              
              {kbData.papers && kbData.papers.map((paper) => (
                <div key={paper.id} className="flex items-center justify-between text-gray-300 py-2 px-2 hover:bg-gray-800 rounded">
                  <span className="truncate">{paper.title}</span>
                  <input type="checkbox" className="rounded bg-gray-700 border-gray-600" defaultChecked />
                </div>
              ))}
            </div>
          </div>
        </div>
        
        {/* 中间区域 - 聊天/内容区域 */}
        <div className={`flex-1 flex flex-col transition-all duration-300 ${
          leftSidebarVisible ? 'ml-80' : 'ml-0'
        } ${
          rightSidebarVisible ? 'mr-96' : 'mr-0'
        }`}>
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <h2 className="text-white font-medium">聊天</h2>
            <div className="flex">
              <button className="text-gray-400 hover:text-white mr-2">
                <Search className="h-5 w-5" />
              </button>
              <button className="text-gray-400 hover:text-white">
                <MessageCircle className="h-5 w-5" />
              </button>
            </div>
          </div>
          
          <div className="flex-1 overflow-auto p-4 bg-[#191a24]">
            {/* 对话内容 */}
            <div className="space-y-4">
              {/* 系统消息 */}
              <div className="bg-gray-800/30 rounded-lg p-4 max-w-[90%]">
                <p className="text-gray-300">
                  欢迎使用知识库管理系统。您可以在此处与知识库内容进行对话交流。
                </p>
              </div>
              
              {/* 用户消息示例 */}
              <div className="flex justify-end">
                <div className="bg-primary/20 text-white rounded-lg p-4 max-w-[90%]">
                  <p>这个知识库有哪些相关论文？</p>
                </div>
              </div>
              
              {/* 系统回复示例 */}
              <div className="bg-gray-800/30 rounded-lg p-4 max-w-[90%]">
                <p className="text-gray-300">
                  该知识库包含 {kbData.papers?.length || 0} 篇论文。其中包括：
                </p>
                <ul className="list-disc pl-5 mt-2 text-gray-300">
                  {kbData.papers && kbData.papers.slice(0, 3).map((paper) => (
                    <li key={paper.id}>{paper.title}</li>
                  ))}
                </ul>
                {kbData.papers && kbData.papers.length > 3 && (
                  <p className="text-gray-400 text-sm mt-2">显示 3/{kbData.papers.length}，请继续询问查看更多</p>
                )}
              </div>
            </div>
          </div>
          
          <div className="p-4 border-t border-gray-800">
            <form onSubmit={handleSearchSubmit} className="flex">
              <Input
                type="text"
                placeholder="开始输入..."
                value={searchQuery}
                onChange={handleSearchChange}
                className="bg-gray-800 border-gray-700"
              />
              <Button type="submit" className="ml-2 bg-primary text-white">
                发送
              </Button>
            </form>
          </div>
        </div>
        
        {/* 右侧边栏切换按钮 */}
        <div className={`fixed top-20 ${rightSidebarVisible ? 'right-[384px]' : 'right-2'} z-30 transition-all duration-300`}>
          <Button 
            variant="ghost" 
            size="icon" 
            onClick={() => setRightSidebarVisible(!rightSidebarVisible)}
            className="bg-gray-800/50 hover:bg-gray-700"
          >
            {rightSidebarVisible ? <ChevronsRight className="h-4 w-4" /> : <ChevronsLeft className="h-4 w-4" />}
          </Button>
        </div>
        
        {/* 右侧边栏 - Studio/预览区域 */}
        <div 
          className={`fixed right-0 top-16 h-[calc(100vh-64px)] z-20 transition-all duration-300 ease-in-out ${
            rightSidebarVisible ? 'w-96 translate-x-0' : 'w-0 translate-x-full'
          } border-l border-gray-800 flex flex-col bg-[#121212] overflow-hidden`}
        >
          <div className="p-4 border-b border-gray-800 flex items-center justify-between">
            <h2 className="text-white font-medium">Ai Copilot</h2>
            <button className="text-gray-400 hover:text-white">
              <Info className="h-5 w-5" />
            </button>
          </div>
          
          <div className="p-4 flex-1 overflow-auto">
            {/* 音频卡片 */}
            <div className="bg-gray-800/40 rounded-lg overflow-hidden">
              <div className="bg-gray-800/70 p-5 flex flex-col items-center justify-center">
                <div className="w-20 h-20 bg-gray-700 rounded-full flex items-center justify-center mb-4">
                  <FileText className="h-10 w-10 text-gray-400" />
                </div>
                <h3 className="text-white font-medium text-lg mb-1">深入探究对话</h3>
                <p className="text-gray-400 text-sm">两位主持人</p>
              </div>
              
              <div className="p-4 flex justify-around">
                <Button variant="outline" className="flex-1 mr-2 bg-gray-800/50 border-gray-700 text-white">
                  自定义
                </Button>
                <Button className="flex-1 ml-2 bg-primary text-white hover:bg-primary/90">
                  生成
                </Button>
              </div>
            </div>
            
            {/* 备注区域 */}
            <div className="mt-8">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-white font-medium">备注</h2>
                <button className="text-gray-400 hover:text-white">
                  <svg width="20" height="20" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg" className="h-5 w-5">
                    <path d="M7.5 11.5L7.5 7M7.5 4.5L7.5 4M14 7.5C14 11.0899 11.0899 14 7.5 14C3.91015 14 1 11.0899 1 7.5C1 3.91015 3.91015 1 7.5 1C11.0899 1 14 3.91015 14 7.5Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </button>
              </div>
              
              <Button variant="outline" className="w-full py-6 border-dashed border-gray-700 text-gray-400 bg-transparent flex items-center justify-center">
                <Plus className="h-5 w-5 mr-2" />
                添加笔记
              </Button>
              
              <div className="grid grid-cols-2 gap-4 mt-4">
                <Button variant="outline" className="flex items-center justify-center py-6 bg-gray-800/40 border-gray-700">
                  <svg width="20" height="20" viewBox="0 0 24 24" className="mr-2" stroke="currentColor" fill="none">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                  </svg>
                  学习指南
                </Button>
                <Button variant="outline" className="flex items-center justify-center py-6 bg-gray-800/40 border-gray-700">
                  <svg width="20" height="20" viewBox="0 0 24 24" className="mr-2" stroke="currentColor" fill="none">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  简报文档
                </Button>
              </div>
              
              <div className="grid grid-cols-2 gap-4 mt-4">
                <Button variant="outline" className="flex items-center justify-center py-6 bg-gray-800/40 border-gray-700">
                  <svg width="20" height="20" viewBox="0 0 24 24" className="mr-2" stroke="currentColor" fill="none">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                  常见问题解答
                </Button>
                <Button variant="outline" className="flex items-center justify-center py-6 bg-gray-800/40 border-gray-700">
                  <svg width="20" height="20" viewBox="0 0 24 24" className="mr-2" stroke="currentColor" fill="none">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  时间轴
                </Button>
              </div>
              
              <div className="mt-16 flex flex-col items-center justify-center text-center">
                <svg viewBox="0 0 24 24" width="40" height="40" className="mb-4">
                  <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" strokeWidth="2" fill="none" />
                  <path d="M7 9L17 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  <path d="M7 13L17 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                  <path d="M7 17L13 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
                <p className="text-gray-400 text-sm px-8">
                  已保存的笔记将显示在此处
                </p>
                <p className="text-gray-500 text-xs px-8 mt-2">
                  可通过保存聊天消息创建新笔记，也可点击上面的"添加笔记"。
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* 添加论文对话框 */}
      <Dialog open={isAddPaperOpen} onOpenChange={setIsAddPaperOpen}>
        <DialogContent className="sm:max-w-md bg-[#1e1e1e] text-white">
          <DialogHeader>
            <DialogTitle>添加文章</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="title" className="text-right">标题</Label>
              <Input
                id="title"
                name="title"
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="authors" className="text-right">作者</Label>
              <Input
                id="authors"
                name="authors"
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
                placeholder="用逗号分隔多个作者"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="publishDate" className="text-right">发表日期</Label>
              <Input
                id="publishDate"
                name="publishDate"
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
                placeholder="YYYY-MM-DD"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="doi" className="text-right">DOI</Label>
              <Input
                id="doi"
                name="doi"
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="url" className="text-right">URL</Label>
              <Input
                id="url"
                name="url"
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
              />
            </div>
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="abstract" className="text-right pt-2">摘要</Label>
              <Textarea
                id="abstract"
                name="abstract"
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
                rows={4}
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddPaperOpen(false)} className="border-gray-700 text-white">
              取消
            </Button>
            <Button onClick={handleAddPaper} className="bg-primary text-white">
              添加
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      
      {/* 删除确认对话框 */}
      <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
        <DialogContent className="sm:max-w-md bg-[#1e1e1e] text-white">
          <DialogHeader>
            <DialogTitle>确认删除</DialogTitle>
          </DialogHeader>
          <div className="py-4">
            <p className="text-gray-300">
              你确定要删除这个项目吗？此操作无法撤销。
            </p>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsDeleteDialogOpen(false)} className="border-gray-700 text-white">
              取消
            </Button>
            <Button onClick={handleDeleteItem} variant="destructive" className="bg-red-600 hover:bg-red-700">
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default KnowledgeBaseManage; 