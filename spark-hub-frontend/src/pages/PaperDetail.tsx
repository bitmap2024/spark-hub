import React, { useState } from "react";
import { useParams, Link, useNavigate } from "react-router-dom";
import Header from "@/components/Header";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { 
  ChevronLeft, 
  ExternalLink, 
  BookOpen, 
  Code, 
  Download, 
  MessageSquare, 
  Users,
  Star,
  Share2
} from "lucide-react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import { useSearchHandler } from "@/lib/navigation";
import { usePaper } from "@/lib/api";
import { Skeleton } from "@/components/ui/skeleton";
import { Paper } from "@/lib/types";

const PaperDetail: React.FC = () => {
  const { kbId, paperId } = useParams<{ kbId: string; paperId: string }>();
  const { toast } = useToast();
  const navigate = useNavigate();
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [activeTab, setActiveTab] = useState("overview");
  
  // 获取论文详情
  const { data, isLoading } = usePaper(parseInt(paperId || "0"));
  // 类型断言，确保数据类型为Paper
  const paperData = data as Paper | undefined;
  
  const handleSearch = useSearchHandler();
  
  // 处理登录点击
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  // 处理返回到知识库详情
  const handleBackToKb = () => {
    navigate(`/knowledge-base/${kbId}`);
  };
  
  // 处理分享论文
  const handleShare = () => {
    navigator.clipboard.writeText(window.location.href);
    toast({
      title: "链接已复制",
      description: "论文链接已复制到剪贴板",
    });
  };
  
  // 处理下载论文
  const handleDownload = () => {
    toast({
      title: "下载开始",
      description: "论文下载已开始",
    });
  };
  
  // 处理收藏论文
  const handleFavorite = () => {
    toast({
      title: "收藏成功",
      description: "论文已添加到收藏",
    });
  };
  
  if (isLoading || !paperData) {
    return (
      <div className="min-h-screen bg-[#121212]">
        <Header 
          onLoginClick={handleLoginClick} 
          onSearch={handleSearch}
        />
        <div className="pt-16 px-4 max-w-6xl mx-auto">
          <div className="mt-8">
            <Skeleton className="h-10 w-2/3 mb-4 bg-gray-800" />
            <Skeleton className="h-6 w-1/2 mb-8 bg-gray-800" />
            <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
              <div className="md:col-span-3">
                <Skeleton className="h-32 mb-4 bg-gray-800" />
                <Skeleton className="h-64 bg-gray-800" />
              </div>
              <div>
                <Skeleton className="h-48 mb-4 bg-gray-800" />
                <Skeleton className="h-20 bg-gray-800" />
              </div>
            </div>
          </div>
        </div>
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
        <Button 
          variant="ghost" 
          className="text-gray-400 hover:text-white pl-0 mt-4" 
          onClick={handleBackToKb}
        >
          <ChevronLeft className="h-4 w-4 mr-1" />
          返回知识库
        </Button>
        
        {/* 论文标题和基本信息 */}
        <div className="mt-4">
          <h1 className="text-2xl font-bold text-white">{paperData.title}</h1>
          <div className="flex flex-wrap mt-2 text-gray-400 text-sm">
            <span className="mr-2">发表于: {paperData.publishDate}</span>
            {paperData.doi && (
              <span className="mr-2">DOI: {paperData.doi}</span>
            )}
          </div>
          <div className="flex flex-wrap mt-2">
            {paperData.authors.map((author, index) => (
              <Badge key={index} variant="outline" className="mr-2 mb-2 bg-gray-800 text-gray-300 border-gray-700">
                {author}
              </Badge>
            ))}
          </div>
        </div>
        
        {/* 操作按钮 */}
        <div className="mt-6 flex flex-wrap gap-2">
          {paperData.url && (
            <Button variant="outline" className="text-white border-gray-700 hover:bg-gray-800" onClick={() => window.open(paperData.url, "_blank")}>
              <ExternalLink className="h-4 w-4 mr-2" />
              原文链接
            </Button>
          )}
          <Button variant="outline" className="text-white border-gray-700 hover:bg-gray-800" onClick={handleDownload}>
            <Download className="h-4 w-4 mr-2" />
            下载 PDF
          </Button>
          <Button variant="outline" className="text-white border-gray-700 hover:bg-gray-800" onClick={handleFavorite}>
            <Star className="h-4 w-4 mr-2" />
            收藏
          </Button>
          <Button variant="outline" className="text-white border-gray-700 hover:bg-gray-800" onClick={handleShare}>
            <Share2 className="h-4 w-4 mr-2" />
            分享
          </Button>
        </div>
        
        {/* 内容区域 */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="md:col-span-3">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="bg-gray-800 border-gray-700">
                <TabsTrigger value="overview" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">
                  <BookOpen className="h-4 w-4 mr-1" />
                  概览
                </TabsTrigger>
                <TabsTrigger value="code" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">
                  <Code className="h-4 w-4 mr-1" />
                  代码
                </TabsTrigger>
                <TabsTrigger value="discussions" className="data-[state=active]:bg-gray-700 data-[state=active]:text-white">
                  <MessageSquare className="h-4 w-4 mr-1" />
                  讨论
                </TabsTrigger>
              </TabsList>
              
              <TabsContent value="overview" className="mt-4">
                <div className="bg-gray-800 p-6 rounded-lg">
                  <h2 className="text-xl font-semibold text-white mb-4">摘要</h2>
                  <p className="text-gray-300 whitespace-pre-line">{paperData.abstract}</p>
                  
                  <h2 className="text-xl font-semibold text-white mt-8 mb-4">关键亮点</h2>
                  <ul className="list-disc list-inside text-gray-300 space-y-2">
                    <li>开源的大型语言模型，专为 Lean 4 形式化定理证明设计</li>
                    <li>通过递归定理证明管道收集初始化数据</li>
                    <li>使用子目标分解技术进行强化学习</li>
                    <li>在 MiniF2F 测试集上达到 88.9% 的通过率</li>
                    <li>解决了 PutnamBench 658 个问题中的 49 个</li>
                    <li>在 AIME 竞赛问题上取得了良好的效果</li>
                  </ul>
                </div>
              </TabsContent>
              
              <TabsContent value="code" className="mt-4">
                <div className="bg-gray-800 p-6 rounded-lg">
                  <div className="flex justify-between items-center mb-4">
                    <h2 className="text-xl font-semibold text-white">代码仓库</h2>
                    {paperData.url && (
                      <a 
                        href={paperData.url} 
                        target="_blank" 
                        rel="noopener noreferrer" 
                        className="text-primary hover:underline flex items-center"
                      >
                        <ExternalLink className="h-4 w-4 mr-1" />
                        访问仓库
                      </a>
                    )}
                  </div>
                  
                  <div className="bg-gray-900 p-4 rounded-md text-gray-300 font-mono text-sm">
                    <p># DeepSeek-Prover-V2</p>
                    <p className="mt-2">官方实现: DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition</p>
                    <p className="mt-4">## 安装</p>
                    <p className="bg-gray-950 p-2 rounded mt-2">pip install deepseek-prover</p>
                    <p className="mt-4">## 使用示例</p>
                    <p className="bg-gray-950 p-2 rounded mt-2 whitespace-pre">
{`from deepseek_prover import DeepSeekProver

prover = DeepSeekProver()
theorem = "For all natural numbers n, n^2 + n + 41 is prime."
proof = prover.prove(theorem)
print(proof)`}
                    </p>
                  </div>
                </div>
              </TabsContent>
              
              <TabsContent value="discussions" className="mt-4">
                <div className="bg-gray-800 p-6 rounded-lg">
                  <h2 className="text-xl font-semibold text-white mb-4">相关讨论</h2>
                  <p className="text-gray-300">暂无讨论内容。成为第一个发起讨论的人！</p>
                  <Button className="mt-4 bg-primary text-white">
                    <MessageSquare className="h-4 w-4 mr-2" />
                    发起讨论
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </div>
          
          {/* 侧边栏 */}
          <div>
            <div className="bg-gray-800 p-4 rounded-lg mb-4">
              <h3 className="text-lg font-semibold text-white mb-3">引用</h3>
              <div className="bg-gray-900 p-3 rounded-md text-gray-300 text-sm font-mono">
                <pre className="whitespace-pre-wrap break-all">
                  {`@article{ren2023deepseekproverv2,
  title={DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition},
  author={Ren, Z.Z. and Shao, Zhihong and Song, Junxiao and others},
  journal={arXiv preprint},
  year={2023}
}`}
                </pre>
              </div>
              <Button variant="outline" className="w-full mt-3 text-white border-gray-700 hover:bg-gray-800">
                复制引用
              </Button>
            </div>
            
            <div className="bg-gray-800 p-4 rounded-lg">
              <h3 className="text-lg font-semibold text-white mb-3">相关论文</h3>
              <div className="space-y-3">
                <div className="border-b border-gray-700 pb-2">
                  <a href="#" className="text-primary hover:underline">DeepSeek: Extending Large Language Models to Theorem Proving</a>
                  <p className="text-gray-400 text-sm mt-1">Z.Z. Ren, et al. (2022)</p>
                </div>
                <div className="border-b border-gray-700 pb-2">
                  <a href="#" className="text-primary hover:underline">Formalized Mathematical Reasoning with LLMs</a>
                  <p className="text-gray-400 text-sm mt-1">Yang, L., et al. (2023)</p>
                </div>
                <div>
                  <a href="#" className="text-primary hover:underline">The Future of Formal Verification</a>
                  <p className="text-gray-400 text-sm mt-1">Chen, K., et al. (2023)</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      {/* 登录对话框 */}
      <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
        <DialogContent className="sm:max-w-md">
          <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default PaperDetail; 