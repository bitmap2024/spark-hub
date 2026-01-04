import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Header from "@/components/Header";
import { useKnowledgeBase, useUser, useCurrentUser } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { 
  ChevronLeft,
  Settings,
  AlertCircle,
  Info
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle, 
  DialogFooter,
  DialogDescription
} from "@/components/ui/dialog";

const KnowledgeBaseSettings: React.FC = () => {
  console.log("渲染 KnowledgeBaseSettings 组件");
  const { kbId } = useParams<{ kbId: string }>();
  const parsedKbId = kbId ? parseInt(kbId) : 0;
  console.log("知识库ID:", kbId, "解析后ID:", parsedKbId);
  
  const { toast } = useToast();
  const navigate = useNavigate();
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);
  const [selectedTab, setSelectedTab] = useState<"basic" | "privacy" | "danger">("basic");
  const [confirmDeleteText, setConfirmDeleteText] = useState("");
  const [error, setError] = useState<string | null>(null);
  
  // 获取知识库数据
  const { data: kbData, isLoading: isKbLoading, error: kbError } = useKnowledgeBase(parsedKbId);
  console.log("知识库数据:", kbData, "加载中:", isKbLoading, "错误:", kbError);
  
  // 获取当前用户数据
  const { data: currentUser, isLoading: isCurrentUserLoading } = useCurrentUser();
  console.log("当前用户:", currentUser, "加载中:", isCurrentUserLoading);
  
  // 获取知识库所有者数据
  const { data: ownerData } = useUser(kbData?.userId || 0);
  console.log("所有者数据:", ownerData);
  
  // 判断当前用户是否是知识库所有者
  const isCurrentUserOwner = currentUser?.id === kbData?.userId;
  console.log("当前用户是否是所有者:", isCurrentUserOwner);
  
  // 表单状态
  const [formState, setFormState] = useState({
    title: "",
    description: "",
    tags: "",
    visibility: "public"
  });
  
  // 当知识库数据加载完成后，初始化表单
  useEffect(() => {
    console.log("useEffect触发 - kbData变化:", kbData);
    if (kbData) {
      try {
        setFormState({
          title: kbData.title || "",
          description: kbData.description || "",
          tags: Array.isArray(kbData.tags) ? kbData.tags.join(", ") : "",
          visibility: "public" // 假设默认为公开
        });
        console.log("表单已初始化:", {
          title: kbData.title,
          description: kbData.description,
          tags: Array.isArray(kbData.tags) ? kbData.tags.join(", ") : "",
          visibility: "public"
        });
      } catch (error) {
        console.error("初始化表单失败:", error);
        setError("初始化表单失败，请刷新页面重试");
      }
    }
  }, [kbData]);
  
  // 简化的搜索处理函数
  const handleSearch = (query: string) => {
    if (query && query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };
  
  // 处理返回按钮点击
  const handleBackClick = () => {
    try {
      navigate(`/knowledge-base/${kbId}`);
    } catch (error) {
      console.error("导航返回失败:", error);
      toast({
        title: "导航失败",
        description: "无法返回知识库页面",
        variant: "destructive"
      });
    }
  };
  
  // 处理输入变化
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    setFormState(prev => ({ ...prev, [name]: value }));
  };
  
  // 处理可见性变化
  const handleVisibilityChange = (value: string) => {
    setFormState(prev => ({ ...prev, visibility: value }));
  };
  
  // 处理保存设置
  const handleSaveSettings = () => {
    // 在实际应用中，这里应该调用API保存设置
    toast({
      title: "保存成功",
      description: "知识库设置已更新",
    });
  };
  
  // 处理删除知识库
  const handleDeleteKnowledgeBase = () => {
    // 验证输入的确认文本是否匹配知识库标题
    if (!kbData || confirmDeleteText !== kbData.title) {
      toast({
        title: "删除失败",
        description: "请正确输入知识库名称以确认删除",
        variant: "destructive"
      });
      return;
    }
    
    // 在实际应用中，这里应该调用API删除知识库
    setIsDeleteDialogOpen(false);
    
    toast({
      title: "删除成功",
      description: "知识库已成功删除",
    });
    
    try {
      // 重定向到首页或用户页面
      navigate("/");
    } catch (error) {
      console.error("导航到首页失败:", error);
    }
  };
  
  // 如果当前用户不是所有者，重定向回知识库详情页
  if (currentUser && !isCurrentUserOwner) {
    console.log("用户无权限，重定向回知识库详情页", {
      currentUserId: currentUser?.id,
      ownerId: kbData?.userId
    });
    navigate(`/knowledge-base/${kbId}`);
    return null;
  }
  
  // 处理加载错误
  if (kbError) {
    console.error("知识库数据加载错误:", kbError);
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white text-center">
          <h2 className="text-xl mb-2">加载失败</h2>
          <p className="text-gray-400">无法加载知识库数据，请稍后再试</p>
          <p className="text-gray-500 text-sm mt-2">错误信息: {kbError.message}</p>
          <Button 
            onClick={handleBackClick}
            className="mt-4 bg-primary text-white"
          >
            返回
          </Button>
        </div>
      </div>
    );
  }
  
  // 处理数据加载中
  if (isKbLoading || isCurrentUserLoading || !kbData) {
    console.log("显示加载中状态...", {isKbLoading, isCurrentUserLoading, kbData});
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white text-center">
          <div className="mb-4">加载中...</div>
          <div className="text-sm text-gray-500">
            知识库ID: {kbId}, 加载状态: {isKbLoading ? '加载中' : '完成'}
          </div>
        </div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onSearch={handleSearch} />
      <div className="pt-16 px-4 max-w-6xl mx-auto">
        {/* 返回按钮 */}
        <button 
          onClick={handleBackClick}
          className="inline-flex items-center text-gray-400 hover:text-white mt-4"
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          返回知识库
        </button>
        
        {/* 标题 */}
        <div className="mt-4 flex flex-col md:flex-row md:items-center md:justify-between">
          <div className="flex items-center">
            <Settings className="h-6 w-6 mr-2 text-primary" />
            <h1 className="text-2xl font-bold text-white">
              {kbData.title} - 设置
            </h1>
          </div>
        </div>
        
        {/* 错误提示 */}
        {error && (
          <div className="mt-4 p-3 bg-red-900/50 border border-red-800 rounded-md text-white">
            {error}
          </div>
        )}
        
        {/* 主要内容 */}
        <div className="mt-8 bg-gray-800 rounded-lg overflow-hidden">
          {/* 标签导航 */}
          <div className="bg-gray-900 p-4 border-b border-gray-700">
            <Tabs 
              value={selectedTab} 
              onValueChange={(value: string) => setSelectedTab(value as any)}
              className="border-b-0"
            >
              <TabsList className="bg-gray-700/50 h-10">
                <TabsTrigger value="basic" className="data-[state=active]:bg-primary data-[state=active]:text-white">
                  <Info className="h-4 w-4 mr-1" />
                  基本信息
                </TabsTrigger>
                <TabsTrigger value="privacy" className="data-[state=active]:bg-primary data-[state=active]:text-white">
                  隐私设置
                </TabsTrigger>
                <TabsTrigger value="danger" className="data-[state=active]:bg-primary data-[state=active]:text-white text-red-400 data-[state=active]:text-white">
                  <AlertCircle className="h-4 w-4 mr-1" />
                  危险区域
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
          
          {/* 内容区域 */}
          <div className="p-6">
            <TabsContent value="basic" className="m-0">
              <div className="space-y-6">
                <div>
                  <Label htmlFor="title" className="text-white font-medium block mb-2">
                    知识库标题
                  </Label>
                  <Input
                    id="title"
                    name="title"
                    value={formState.title}
                    onChange={handleInputChange}
                    className="bg-gray-700 border-gray-600 focus-visible:ring-primary text-white"
                  />
                  <p className="text-gray-400 text-sm mt-1">
                    为您的知识库提供一个简洁明了的标题。
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="description" className="text-white font-medium block mb-2">
                    知识库描述
                  </Label>
                  <Textarea
                    id="description"
                    name="description"
                    value={formState.description}
                    onChange={handleInputChange}
                    rows={4}
                    className="bg-gray-700 border-gray-600 focus-visible:ring-primary text-white"
                  />
                  <p className="text-gray-400 text-sm mt-1">
                    简要描述这个知识库的内容和用途。
                  </p>
                </div>
                
                <div>
                  <Label htmlFor="tags" className="text-white font-medium block mb-2">
                    标签
                  </Label>
                  <Input
                    id="tags"
                    name="tags"
                    value={formState.tags}
                    onChange={handleInputChange}
                    className="bg-gray-700 border-gray-600 focus-visible:ring-primary text-white"
                    placeholder="用逗号分隔多个标签"
                  />
                  <p className="text-gray-400 text-sm mt-1">
                    添加标签以便其他用户更容易找到您的知识库。
                  </p>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="privacy" className="m-0">
              <div className="space-y-6">
                <div>
                  <Label className="text-white font-medium block mb-3">
                    可见性
                  </Label>
                  <RadioGroup 
                    value={formState.visibility}
                    onValueChange={handleVisibilityChange}
                    className="space-y-4"
                  >
                    <div className="flex items-start space-x-3">
                      <RadioGroupItem value="public" id="public" className="mt-1" />
                      <div>
                        <Label htmlFor="public" className="text-white font-medium">公开</Label>
                        <p className="text-gray-400 text-sm mt-1">
                          所有人都可以查看这个知识库的内容。
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <RadioGroupItem value="friends" id="friends" className="mt-1" />
                      <div>
                        <Label htmlFor="friends" className="text-white font-medium">朋友可见</Label>
                        <p className="text-gray-400 text-sm mt-1">
                          只有您的关注者和好友可以查看这个知识库的内容。
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start space-x-3">
                      <RadioGroupItem value="private" id="private" className="mt-1" />
                      <div>
                        <Label htmlFor="private" className="text-white font-medium">私有</Label>
                        <p className="text-gray-400 text-sm mt-1">
                          只有您可以查看这个知识库的内容。
                        </p>
                      </div>
                    </div>
                  </RadioGroup>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="danger" className="m-0">
              <div className="bg-red-950/20 border border-red-900/50 rounded-lg p-6">
                <h3 className="text-xl font-semibold text-red-400 mb-2">危险操作</h3>
                <p className="text-gray-300 mb-6">
                  以下操作是不可逆的，请谨慎操作。
                </p>
                
                <div className="space-y-6">
                  <div>
                    <h4 className="text-lg font-medium text-white mb-2">删除知识库</h4>
                    <p className="text-gray-400 mb-4">
                      删除此知识库将永久移除所有相关数据，包括论文、议题和评论等。此操作无法撤销。
                    </p>
                    <Button 
                      onClick={() => setIsDeleteDialogOpen(true)}
                      variant="destructive" 
                      className="bg-red-600 hover:bg-red-700"
                    >
                      删除知识库
                    </Button>
                  </div>
                </div>
              </div>
            </TabsContent>
          </div>
          
          {/* 保存按钮 */}
          {selectedTab !== "danger" && (
            <div className="px-6 py-4 bg-gray-900 border-t border-gray-700 flex justify-end">
              <Button 
                onClick={handleSaveSettings}
                className="bg-primary text-white"
              >
                保存设置
              </Button>
            </div>
          )}
        </div>
      </div>
      
      {/* 删除确认对话框 */}
      {kbData && (
        <Dialog open={isDeleteDialogOpen} onOpenChange={setIsDeleteDialogOpen}>
          <DialogContent className="sm:max-w-md bg-[#1e1e1e] text-white">
            <DialogHeader>
              <DialogTitle className="text-red-400">确认删除知识库</DialogTitle>
              <DialogDescription className="text-gray-400">
                此操作无法撤销，所有数据将被永久删除。
              </DialogDescription>
            </DialogHeader>
            <div className="py-4">
              <div className="bg-red-950/20 border border-red-900/50 rounded-lg p-4 mb-4">
                <p className="text-gray-300">
                  请输入知识库名称 <span className="font-medium text-white">"{kbData.title}"</span> 以确认删除。
                </p>
              </div>
              <Input 
                placeholder={kbData.title}
                value={confirmDeleteText}
                onChange={(e) => setConfirmDeleteText(e.target.value)}
                className="bg-gray-700 border-gray-600 focus-visible:ring-primary text-white"
              />
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setIsDeleteDialogOpen(false)} className="border-gray-700 text-white">
                取消
              </Button>
              <Button 
                onClick={handleDeleteKnowledgeBase} 
                variant="destructive" 
                className="bg-red-600 hover:bg-red-700"
                disabled={confirmDeleteText !== kbData.title}
              >
                确认删除
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
};

export default KnowledgeBaseSettings; 