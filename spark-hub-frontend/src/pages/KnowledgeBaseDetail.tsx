import React, { useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import Header from "@/components/Header";
import { useKnowledgeBase, useUser, useCurrentUser } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { 
  Star, 
  GitFork, 
  Clock, 
  Tag, 
  FileText, 
  ExternalLink, 
  ChevronLeft,
  Plus,
  MessageCircle,
  Send,
  ArrowLeft,
  Settings,
  Database
} from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { useToast } from "@/hooks/use-toast";
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
import EmailLoginForm from "@/components/EmailLoginForm";
import UserAvatar from "@/components/UserAvatar";
import { useSearchHandler } from "@/lib/navigation";

const KnowledgeBaseDetail: React.FC = () => {
  const { kbId } = useParams<{ kbId: string }>();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [isAddPaperOpen, setIsAddPaperOpen] = useState(false);
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [isAddIssueOpen, setIsAddIssueOpen] = useState(false);
  const [issueView, setIssueView] = useState<"list" | "detail">("list");
  const [currentIssue, setCurrentIssue] = useState<{
    id: number;
    title: string;
    content: string;
    status: string;
    author: {
      name: string;
      avatar: string;
    };
    createdAt: string;
    comments: Array<{
      id: number;
      content: string;
      author: {
        name: string;
        avatar: string;
      };
      createdAt: string;
    }>;
  } | null>(null);
  const [newIssue, setNewIssue] = useState({
    title: "",
    content: ""
  });
  const [newPaper, setNewPaper] = useState({
    title: "",
    authors: "",
    abstract: "",
    publishDate: "",
    doi: "",
    url: ""
  });
  const [newComment, setNewComment] = useState("");
  
  // 获取知识库数据
  const { data: kbData, isLoading: isKbLoading } = useKnowledgeBase(parseInt(kbId || "0"));
  
  // 获取知识库所有者数据
  const { data: ownerData } = useUser(kbData?.userId || 0);
  
  // 获取当前用户数据
  const { data: currentUser } = useCurrentUser();
  
  // 判断当前用户是否是知识库所有者
  const isCurrentUserOwner = currentUser?.id === kbData?.userId;
  
  // 示例议题数据
  const sampleIssues = [
    {
      id: 1,
      title: "如何更好地理解这篇论文的方法部分？",
      content: "我在理解论文的方法部分时遇到了一些困难，特别是关于他们使用的数学模型。论文中提到的公式(3)和公式(5)之间的联系不是很清楚，是否有人可以解释一下这两个公式的推导过程？另外，实验部分使用的参数设置似乎与理论分析不太一致，这是为什么？",
      status: "进行中",
      author: {
        name: "JohnDoe",
        avatar: "https://github.com/shadcn.png"
      },
      createdAt: "2023-05-15",
      comments: [
        {
          id: 1,
          content: "我也遇到了同样的问题，特别是公式(3)中的参数λ的选择依据不是很明确。",
          author: {
            name: "Alice",
            avatar: ""
          },
          createdAt: "2023-05-16"
        },
        {
          id: 2,
          content: "关于公式(3)和(5)的联系，其实是通过泰勒展开推导的，论文的附录A中有详细的推导过程，但确实比较简略。我可以分享一下我的理解...",
          author: {
            name: "Professor",
            avatar: ""
          },
          createdAt: "2023-05-17"
        },
        {
          id: 3,
          content: "实验部分的参数设置是基于作者的初步测试结果，他们在后续的消融实验中解释了不同参数设置的影响，可以参考论文的Figure 4。",
          author: {
            name: "Researcher",
            avatar: ""
          },
          createdAt: "2023-05-18"
        }
      ]
    },
    {
      id: 2,
      title: "这篇论文的实验结果是否可复现？",
      content: "我尝试复现这篇论文的实验结果，但是发现有些细节作者没有提供。特别是他们使用的数据预处理步骤，以及一些超参数的设置。有谁成功复现过这篇论文的结果吗？能否分享一下你们的经验？",
      status: "已解决",
      author: {
        name: "WangLei",
        avatar: ""
      },
      createdAt: "2023-05-10",
      comments: [
        {
          id: 1,
          content: "我成功复现了这篇论文的结果，关键是数据归一化的方法需要用z-score而不是min-max。",
          author: {
            name: "Reproducer",
            avatar: ""
          },
          createdAt: "2023-05-11"
        },
        {
          id: 2,
          content: "我联系了论文作者，他们已经在GitHub上开源了完整代码：https://github.com/author/paper-code",
          author: {
            name: "Helper",
            avatar: ""
          },
          createdAt: "2023-05-12"
        },
        {
          id: 3,
          content: "感谢分享！我使用作者提供的代码成功复现了结果，确实如前面所说，数据预处理是关键。",
          author: {
            name: "WangLei",
            avatar: ""
          },
          createdAt: "2023-05-13"
        },
        {
          id: 4,
          content: "我也复现成功了，另外注意论文中Table 1的结果是运行5次的平均值，单次运行可能会有波动。",
          author: {
            name: "AnotherReproducer",
            avatar: ""
          },
          createdAt: "2023-05-14"
        },
        {
          id: 5,
          content: "对了，还需要注意随机种子的设置，我用了作者代码中的种子才得到接近的结果。",
          author: {
            name: "DetailPerson",
            avatar: ""
          },
          createdAt: "2023-05-15"
        },
        {
          id: 6,
          content: "我已经将完整的复现步骤整理成了一个文档，有需要的可以参考：https://myrepo.com/reproduction-guide",
          author: {
            name: "Organizer",
            avatar: ""
          },
          createdAt: "2023-05-16"
        },
        {
          id: 7,
          content: "感谢大家的帮助，我已经成功复现了结果！",
          author: {
            name: "WangLei",
            avatar: ""
          },
          createdAt: "2023-05-17"
        }
      ]
    }
  ];
  
  const handleSearch = useSearchHandler();
  
  // 处理登录点击
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  // 处理添加论文
  const handleAddPaper = () => {
    // 在实际应用中，这里应该调用API添加论文
    setIsAddPaperOpen(false);
    toast({
      title: "添加成功",
      description: "论文已添加到知识库",
    });
  };
  
  // 处理添加议题
  const handleAddIssue = () => {
    // 在实际应用中，这里应该调用API添加议题
    setIsAddIssueOpen(false);
    toast({
      title: "添加成功",
      description: "议题已添加到知识库",
    });
  };
  
  // 处理议题点击
  const handleIssueClick = (issue: any) => {
    setCurrentIssue(issue);
    setIssueView("detail");
  };
  
  // 返回议题列表
  const handleBackToIssues = () => {
    setIssueView("list");
    setCurrentIssue(null);
  };
  
  // 处理评论提交
  const handleCommentSubmit = () => {
    if (!newComment.trim()) return;
    
    // 在实际应用中，这里应该调用API提交评论
    if (currentIssue) {
      const updatedIssue = {
        ...currentIssue,
        comments: [
          ...currentIssue.comments,
          {
            id: currentIssue.comments.length + 1,
            content: newComment,
            author: {
              name: "当前用户",
              avatar: "https://github.com/shadcn.png"
            },
            createdAt: new Date().toISOString().split('T')[0]
          }
        ]
      };
      setCurrentIssue(updatedIssue);
      setNewComment("");
      
      toast({
        title: "评论成功",
        description: "你的回复已添加到议题中",
      });
    }
  };
  
  // 处理输入变化
  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value } = e.target;
    if (name === "title" || name === "content") {
      setNewIssue(prev => ({ ...prev, [name]: value }));
    } else {
      setNewPaper(prev => ({ ...prev, [name]: value }));
    }
  };
  
  // 处理管理按钮点击
  const handleManageClick = () => {
    try {
      navigate(`/knowledge-base/${kbId}/manage`);
    } catch (error) {
      console.error("导航到管理页面失败", error);
      toast({
        title: "导航失败",
        description: "无法打开管理页面",
        variant: "destructive"
      });
    }
  };

  // 处理设置按钮点击
  const handleSettingsClick = () => {
    try {
      navigate(`/knowledge-base/${kbId}/settings`);
    } catch (error) {
      console.error("导航到设置页面失败", error);
      toast({
        title: "导航失败",
        description: "无法打开设置页面",
        variant: "destructive"
      });
    }
  };
  
  if (isKbLoading || !kbData) {
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white">加载中...</div>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-[#121212]">
      <Header 
        onLoginClick={handleLoginClick} 
        onSearch={handleSearch}
      />
      <div className="pt-16 px-4 max-w-6xl mx-auto">
        {/* 返回按钮 */}
        <Link to={`/user/${ownerData?.username || ""}`} className="inline-flex items-center text-gray-400 hover:text-white mt-4">
          <ChevronLeft className="h-4 w-4 mr-1" />
          返回用户主页
        </Link>
        
        {/* 知识库标题和操作 */}
        <div className="mt-4 flex flex-col md:flex-row md:items-center md:justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">{kbData.title}</h1>
            <p className="text-gray-400 mt-1">{kbData.description}</p>
          </div>
          <div className="flex mt-4 md:mt-0">
            {isCurrentUserOwner && (
              <>
                <Button 
                  variant="outline" 
                  className="mr-2 text-white border-gray-700 hover:bg-gray-800"
                  onClick={handleManageClick}
                >
                  <Database className="h-4 w-4 mr-1" />
                  管理
                </Button>
                <Button 
                  variant="outline" 
                  className="mr-2 text-white border-gray-700 hover:bg-gray-800"
                  onClick={handleSettingsClick}
                >
                  <Settings className="h-4 w-4 mr-1" />
                  设置
                </Button>
              </>
            )}
            <Button variant="outline" className="mr-2 text-white border-gray-700 hover:bg-gray-800">
              <Star className="h-4 w-4 mr-1" />
              收藏 ({kbData.stars})
            </Button>
            <Button variant="outline" className="text-white border-gray-700 hover:bg-gray-800">
              <GitFork className="h-4 w-4 mr-1" />
              复制 ({kbData.forks})
            </Button>
          </div>
        </div>
        
        {/* 知识库信息 */}
        <div className="mt-6 flex flex-wrap items-center text-sm text-gray-400">
          <div className="flex items-center mr-4">
            <Avatar className="h-6 w-6 mr-2">
              <AvatarImage src={ownerData?.avatar} />
              <AvatarFallback>{ownerData?.username.charAt(0).toUpperCase()}</AvatarFallback>
            </Avatar>
            <Link to={`/user/${ownerData?.username || ""}`} className="hover:text-white">
              {ownerData?.username}
            </Link>
          </div>
          <div className="flex items-center mr-4">
            <Clock className="h-4 w-4 mr-1" />
            更新于 {kbData.updatedAt}
          </div>
          <div className="flex items-center">
            <Tag className="h-4 w-4 mr-1" />
            {kbData.tags.map(tag => (
              <Badge key={tag} variant="outline" className="mr-1 bg-gray-800 text-gray-300 border-gray-700">
                {tag}
              </Badge>
            ))}
          </div>
        </div>
        
        {/* 标签页 */}
        <Tabs defaultValue="papers" className="mt-8">
          <TabsList className="bg-gray-800 border-gray-700">
            <TabsTrigger value="papers" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">
              <FileText className="h-4 w-4 mr-1" />
              论文 ({kbData.papers.length})
            </TabsTrigger>
            <TabsTrigger value="issues" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">
              <MessageCircle className="h-4 w-4 mr-1" />
              议题
            </TabsTrigger>
            <TabsTrigger value="about" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">
              关于
            </TabsTrigger>
          </TabsList>
          
          <TabsContent value="papers" className="mt-4">
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-xl font-semibold text-white">论文列表</h2>
              <Button 
                onClick={() => setIsAddPaperOpen(true)}
                className="bg-primary text-white"
              >
                <Plus className="h-4 w-4 mr-1" />
                添加论文
              </Button>
            </div>
            
            <div className="space-y-4">
              {kbData.papers.map(paper => (
                <div key={paper.id} className="bg-gray-800 p-4 rounded-lg">
                  <h3 className="text-lg font-medium text-white">{paper.title}</h3>
                  <p className="text-gray-400 text-sm mt-1">
                    作者: {paper.authors.join(", ")}
                  </p>
                  <p className="text-gray-400 text-sm">
                    发表日期: {paper.publishDate}
                  </p>
                  <p className="text-gray-300 mt-2">{paper.abstract}</p>
                  <div className="mt-3 flex items-center">
                    {paper.doi && (
                      <span className="text-gray-400 text-sm mr-4">
                        DOI: {paper.doi}
                      </span>
                    )}
                    {paper.url && (
                      <Link 
                        to={`/knowledge-base/${kbId}/paper/${paper.id}`}
                        className="text-primary hover:underline text-sm flex items-center"
                      >
                        <ExternalLink className="h-3 w-3 mr-1" />
                        查看原文
                      </Link>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </TabsContent>
          
          <TabsContent value="issues" className="mt-4">
            {issueView === "list" ? (
              <>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-semibold text-white">议题讨论</h2>
                  <Button 
                    onClick={() => setIsAddIssueOpen(true)}
                    className="bg-primary text-white"
                  >
                    <Plus className="h-4 w-4 mr-1" />
                    创建议题
                  </Button>
                </div>
                
                <div className="space-y-4">
                  {sampleIssues.map(issue => (
                    <div 
                      key={issue.id} 
                      className="bg-gray-800 p-4 rounded-lg cursor-pointer hover:bg-gray-700 transition-colors"
                      onClick={() => handleIssueClick(issue)}
                    >
                      <div className="flex justify-between">
                        <h3 className="text-lg font-medium text-white">{issue.title}</h3>
                        <Badge variant="outline" className={
                          issue.status === "进行中" 
                            ? "bg-green-900/30 text-green-400 border-green-800"
                            : "bg-blue-900/30 text-blue-400 border-blue-800"
                        }>
                          {issue.status}
                        </Badge>
                      </div>
                      <div className="flex items-center mt-2 text-sm text-gray-400">
                        <UserAvatar 
                          username={issue.author.name}
                          avatarSrc={issue.author.avatar}
                          size="sm"
                          className="h-5 w-5 mr-2"
                        />
                        <span>{issue.author.name}</span>
                        <span className="mx-2">•</span>
                        <span>创建于 {issue.createdAt}</span>
                        <span className="mx-2">•</span>
                        <span>{issue.comments.length}个回复</span>
                      </div>
                      <p className="text-gray-300 mt-2 line-clamp-2">
                        {issue.content}
                      </p>
                    </div>
                  ))}
                </div>
              </>
            ) : (
              currentIssue && (
                <div className="space-y-6">
                  {/* 返回按钮 */}
                  <Button 
                    variant="ghost" 
                    className="text-gray-400 hover:text-white pl-0" 
                    onClick={handleBackToIssues}
                  >
                    <ArrowLeft className="h-4 w-4 mr-1" />
                    返回议题列表
                  </Button>
                  
                  {/* 议题标题和状态 */}
                  <div className="flex justify-between items-start">
                    <h2 className="text-2xl font-semibold text-white">{currentIssue.title}</h2>
                    <Badge variant="outline" className={
                      currentIssue.status === "进行中" 
                        ? "bg-green-900/30 text-green-400 border-green-800"
                        : "bg-blue-900/30 text-blue-400 border-blue-800"
                    }>
                      {currentIssue.status}
                    </Badge>
                  </div>
                  
                  {/* 作者信息 */}
                  <div className="flex items-center text-sm text-gray-400">
                    <UserAvatar 
                      username={currentIssue.author.name}
                      avatarSrc={currentIssue.author.avatar}
                      size="sm"
                      className="h-5 w-5 mr-2"
                    />
                    <span>{currentIssue.author.name}</span>
                    <span className="mx-2">•</span>
                    <span>创建于 {currentIssue.createdAt}</span>
                  </div>
                  
                  {/* 议题内容 */}
                  <div className="bg-gray-800 p-6 rounded-lg">
                    <p className="text-gray-300 whitespace-pre-line">{currentIssue.content}</p>
                  </div>
                  
                  {/* 评论部分 */}
                  <div className="space-y-4 mt-8">
                    <h3 className="text-xl font-medium text-white">评论 ({currentIssue.comments.length})</h3>
                    
                    {/* 评论列表 */}
                    {currentIssue.comments.map(comment => (
                      <div key={comment.id} className="bg-gray-800 p-4 rounded-lg">
                        <div className="flex items-center mb-3">
                          <UserAvatar 
                            username={comment.author.name}
                            avatarSrc={comment.author.avatar}
                            size="sm"
                            className="mr-2"
                          />
                          <span className="text-white font-medium">{comment.author.name}</span>
                          <span className="mx-2 text-gray-400">•</span>
                          <span className="text-gray-400 text-sm">{comment.createdAt}</span>
                        </div>
                        <p className="text-gray-300">{comment.content}</p>
                      </div>
                    ))}
                    
                    {/* 添加评论 */}
                    <div className="bg-gray-800 p-4 rounded-lg mt-6">
                      <h4 className="text-lg font-medium text-white mb-3">添加回复</h4>
                      <div className="flex items-start">
                        <UserAvatar 
                          username="当前用户"
                          avatarSrc="https://github.com/shadcn.png"
                          size="sm"
                          className="h-8 w-8 mr-3 mt-1"
                        />
                        <div className="flex-grow">
                          <Textarea
                            value={newComment}
                            onChange={(e) => setNewComment(e.target.value)}
                            placeholder="添加评论..."
                            className="bg-[#2a2a2a] border-gray-700 w-full min-h-[100px]"
                          />
                          <div className="flex justify-end mt-3">
                            <Button 
                              onClick={handleCommentSubmit}
                              className="bg-primary text-white"
                              disabled={!newComment.trim()}
                            >
                              <Send className="h-4 w-4 mr-1" />
                              发送回复
                            </Button>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )
            )}
          </TabsContent>
          
          <TabsContent value="about" className="mt-4">
            <div className="bg-gray-800 p-6 rounded-lg">
              <h2 className="text-xl font-semibold text-white mb-4">关于此知识库</h2>
              <p className="text-gray-300 mb-4">{kbData.description}</p>
              <div className="text-gray-400 text-sm">
                <p>创建于: {kbData.createdAt}</p>
                <p>最后更新: {kbData.updatedAt}</p>
                <p>收藏数: {kbData.stars}</p>
                <p>复制数: {kbData.forks}</p>
              </div>
            </div>
          </TabsContent>
          
        </Tabs>
      </div>
      
      {/* 添加论文对话框 */}
      <Dialog open={isAddPaperOpen} onOpenChange={setIsAddPaperOpen}>
        <DialogContent className="sm:max-w-md bg-[#1e1e1e] text-white">
          <DialogHeader>
            <DialogTitle>添加论文</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="title" className="text-right">标题</Label>
              <Input
                id="title"
                name="title"
                value={newPaper.title}
                onChange={handleInputChange}
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="authors" className="text-right">作者</Label>
              <Input
                id="authors"
                name="authors"
                value={newPaper.authors}
                onChange={handleInputChange}
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
                placeholder="用逗号分隔多个作者"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="publishDate" className="text-right">发表日期</Label>
              <Input
                id="publishDate"
                name="publishDate"
                value={newPaper.publishDate}
                onChange={handleInputChange}
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
                placeholder="YYYY-MM-DD"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="doi" className="text-right">DOI</Label>
              <Input
                id="doi"
                name="doi"
                value={newPaper.doi}
                onChange={handleInputChange}
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
              />
            </div>
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="url" className="text-right">URL</Label>
              <Input
                id="url"
                name="url"
                value={newPaper.url}
                onChange={handleInputChange}
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
              />
            </div>
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="abstract" className="text-right pt-2">摘要</Label>
              <Textarea
                id="abstract"
                name="abstract"
                value={newPaper.abstract}
                onChange={handleInputChange}
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
      
      {/* 登录对话框 */}
      <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
        <DialogContent className="sm:max-w-md">
          <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
        </DialogContent>
      </Dialog>
      
      {/* 添加议题对话框 */}
      <Dialog open={isAddIssueOpen} onOpenChange={setIsAddIssueOpen}>
        <DialogContent className="sm:max-w-md bg-[#1e1e1e] text-white">
          <DialogHeader>
            <DialogTitle>创建议题</DialogTitle>
          </DialogHeader>
          <div className="grid gap-4 py-4">
            <div className="grid grid-cols-4 items-center gap-4">
              <Label htmlFor="title" className="text-right">标题</Label>
              <Input
                id="title"
                name="title"
                value={newIssue.title}
                onChange={handleInputChange}
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
                placeholder="议题标题"
              />
            </div>
            <div className="grid grid-cols-4 items-start gap-4">
              <Label htmlFor="content" className="text-right pt-2">内容</Label>
              <Textarea
                id="content"
                name="content"
                value={newIssue.content}
                onChange={handleInputChange}
                className="col-span-3 bg-[#2a2a2a] border-gray-700"
                rows={6}
                placeholder="详细描述你的问题或讨论主题..."
              />
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setIsAddIssueOpen(false)} className="border-gray-700 text-white">
              取消
            </Button>
            <Button onClick={handleAddIssue} className="bg-primary text-white">
              发布
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default KnowledgeBaseDetail; 