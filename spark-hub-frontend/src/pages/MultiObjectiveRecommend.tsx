
import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import EmailLoginForm from "@/components/EmailLoginForm";
import KnowledgeBaseVideoFeed from "@/components/KnowledgeBaseVideoFeed";
import { AlertCircle, BookOpen, Clock, Heart, TrendingUp, UserCheck, BarChart3 } from "lucide-react";
import UserAvatar from "@/components/UserAvatar";

// 推荐目标选项
const objectiveOptions = [
  { id: "ctr", name: "点击率", icon: <TrendingUp className="h-4 w-4 mr-2" />, description: "提高内容的点击概率" },
  { id: "cvr", name: "转化率", icon: <BookOpen className="h-4 w-4 mr-2" />, description: "提高用户行动转化率" },
  { id: "satisfaction", name: "满意度", icon: <Heart className="h-4 w-4 mr-2" />, description: "提高用户满意度体验" },
  { id: "diversity", name: "多样性", icon: <BarChart3 className="h-4 w-4 mr-2" />, description: "增加推荐内容的多样性" }
];

// 模拟的推荐设置
const defaultWeights = {
  ctr: 1.0,
  cvr: 1.0,
  satisfaction: 1.0,
  diversity: 0.5
};

const MultiObjectiveRecommend: React.FC = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<string>("feed");
  const [weights, setWeights] = useState(defaultWeights);
  const [diversityFactor, setDiversityFactor] = useState(0.3);
  const [unexpectednessFactor, setUnexpectednessFactor] = useState(0.2);
  const [objectiveVisibility, setObjectiveVisibility] = useState<Record<string, boolean>>({
    ctr: true,
    cvr: true,
    satisfaction: true,
    diversity: true
  });
  
  // 模拟推荐数据
  const [recommendations, setRecommendations] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  
  const navigate = useNavigate();
  
  // 模拟获取推荐
  useEffect(() => {
    setIsLoading(true);
    
    // 模拟API调用延迟
    const timer = setTimeout(() => {
      // 这里应该是从后端获取的推荐数据
      const mockRecommendations = [
        {
          id: "kb1",
          title: "深度学习基础",
          description: "深度学习的基本原理和应用场景",
          tags: ["AI", "机器学习", "神经网络"],
          userId: 1,
          scores: { ctr: 0.92, cvr: 0.75, satisfaction: 0.88, diversity: 0.65 }
        },
        {
          id: "kb2",
          title: "推荐系统实践",
          description: "现代推荐算法的实际应用案例",
          tags: ["推荐系统", "机器学习", "个性化"],
          userId: 2,
          scores: { ctr: 0.85, cvr: 0.90, satisfaction: 0.82, diversity: 0.70 }
        },
        {
          id: "kb3",
          title: "自然语言处理入门",
          description: "NLP基础知识与实践技巧",
          tags: ["NLP", "文本分析", "语言模型"],
          userId: 3,
          scores: { ctr: 0.78, cvr: 0.72, satisfaction: 0.93, diversity: 0.80 }
        },
        {
          id: "kb4",
          title: "计算机视觉技术",
          description: "视觉识别与图像处理的核心技术",
          tags: ["计算机视觉", "图像处理", "目标检测"],
          userId: 4,
          scores: { ctr: 0.81, cvr: 0.68, satisfaction: 0.79, diversity: 0.90 }
        }
      ];
      
      setRecommendations(mockRecommendations);
      setIsLoading(false);
    }, 1000);
    
    return () => clearTimeout(timer);
  }, [weights, diversityFactor, unexpectednessFactor]);
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  const handleWeightChange = (objective: string, value: number[]) => {
    setWeights({
      ...weights,
      [objective]: value[0]
    });
  };
  
  const toggleObjective = (objective: string) => {
    setObjectiveVisibility({
      ...objectiveVisibility,
      [objective]: !objectiveVisibility[objective]
    });
  };
  
  const resetWeights = () => {
    setWeights(defaultWeights);
    setDiversityFactor(0.3);
    setUnexpectednessFactor(0.2);
  };
  
  const calculateOverallScore = (scores: Record<string, number>) => {
    let totalWeight = 0;
    let weightedSum = 0;
    
    for (const objective in weights) {
      if (objectiveVisibility[objective] && scores[objective]) {
        weightedSum += weights[objective] * scores[objective];
        totalWeight += weights[objective];
      }
    }
    
    return totalWeight > 0 ? weightedSum / totalWeight : 0;
  };
  
  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onLoginClick={handleLoginClick} />
      <LeftSidebar />
      
      {/* 主体内容区域 */}
      <div className="ml-64 pt-16 px-6">
        <Tabs defaultValue="feed" value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold text-white mb-4">多目标推荐</h1>
            <TabsList className="grid grid-cols-2 w-[300px]">
              <TabsTrigger value="feed">视频推荐</TabsTrigger>
              <TabsTrigger value="settings">偏好设置</TabsTrigger>
            </TabsList>
          </div>
          
          <TabsContent value="feed" className="mt-0">
            {isLoading ? (
              <div className="flex justify-center items-center h-[70vh]">
                <div className="text-white">加载推荐内容中...</div>
              </div>
            ) : (
              <div className="grid grid-cols-1">
                <KnowledgeBaseVideoFeed sourceType="recommend" displayStyle="douyin" />
                
                {/* 推荐指标面板（浮动在右上角） */}
                <div className="fixed top-20 right-6 w-64 bg-gray-900/90 rounded-lg p-4 shadow-lg">
                  <h3 className="text-white font-medium mb-2 flex items-center">
                    <BarChart3 className="w-4 h-4 mr-2" />
                    推荐指标
                  </h3>
                  <div className="space-y-1">
                    {objectiveOptions.map(objective => 
                      objectiveVisibility[objective.id] && (
                        <div key={objective.id} className="flex justify-between items-center text-sm">
                          <div className="flex items-center text-gray-300">
                            {objective.icon}
                            {objective.name}
                          </div>
                          <div>
                            <Badge variant={weights[objective.id] > 0.8 ? "default" : "secondary"} className="text-xs">
                              权重: {weights[objective.id].toFixed(1)}
                            </Badge>
                          </div>
                        </div>
                      )
                    )}
                    <div className="pt-2 mt-2 border-t border-gray-700">
                      <div className="flex justify-between items-center text-sm text-gray-300">
                        <span>多样性因子:</span>
                        <span>{diversityFactor.toFixed(1)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="settings" className="mt-0">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="md:col-span-2">
                <Card className="bg-gray-900 border-gray-800">
                  <CardHeader>
                    <CardTitle className="text-white">推荐目标权重设置</CardTitle>
                    <CardDescription className="text-gray-400">
                      调整各个目标的权重来自定义您的推荐体验
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {objectiveOptions.map(objective => (
                      <div key={objective.id} className="space-y-2">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center text-white">
                            {objective.icon}
                            <span>{objective.name}</span>
                          </div>
                          <Button 
                            variant={objectiveVisibility[objective.id] ? "default" : "outline"}
                            size="sm"
                            onClick={() => toggleObjective(objective.id)}
                          >
                            {objectiveVisibility[objective.id] ? "启用" : "禁用"}
                          </Button>
                        </div>
                        <div className="flex items-center gap-4">
                          <Slider
                            disabled={!objectiveVisibility[objective.id]}
                            value={[weights[objective.id]]}
                            min={0}
                            max={2}
                            step={0.1}
                            onValueChange={(value) => handleWeightChange(objective.id, value)}
                            className="flex-1"
                          />
                          <span className="text-white min-w-[40px] text-right">
                            {weights[objective.id].toFixed(1)}
                          </span>
                        </div>
                        <p className="text-gray-400 text-sm">{objective.description}</p>
                      </div>
                    ))}
                    
                    <div className="pt-4 border-t border-gray-800">
                      <div className="space-y-4">
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="text-white flex items-center">
                              <BarChart3 className="h-4 w-4 mr-2" />
                              多样性因子
                            </label>
                            <span className="text-white">{diversityFactor.toFixed(1)}</span>
                          </div>
                          <Slider
                            value={[diversityFactor]}
                            min={0}
                            max={1}
                            step={0.1}
                            onValueChange={(value) => setDiversityFactor(value[0])}
                          />
                          <p className="text-gray-400 text-sm mt-1">控制推荐结果的多样性程度，值越高结果越多样</p>
                        </div>
                        
                        <div>
                          <div className="flex items-center justify-between mb-2">
                            <label className="text-white flex items-center">
                              <AlertCircle className="h-4 w-4 mr-2" />
                              意外因子
                            </label>
                            <span className="text-white">{unexpectednessFactor.toFixed(1)}</span>
                          </div>
                          <Slider
                            value={[unexpectednessFactor]}
                            min={0}
                            max={1}
                            step={0.1}
                            onValueChange={(value) => setUnexpectednessFactor(value[0])}
                          />
                          <p className="text-gray-400 text-sm mt-1">控制推荐结果的惊喜程度，值越高可能出现更多意外发现</p>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                  <CardFooter className="border-t border-gray-800 pt-4">
                    <Button onClick={resetWeights} variant="outline" className="mr-2">
                      重置为默认
                    </Button>
                    <Button onClick={() => setActiveTab("feed")}>
                      应用设置
                    </Button>
                  </CardFooter>
                </Card>
              </div>
              
              <div>
                <Card className="bg-gray-900 border-gray-800">
                  <CardHeader>
                    <CardTitle className="text-white">推荐示例</CardTitle>
                    <CardDescription className="text-gray-400">
                      根据当前设置的推荐内容预览
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {isLoading ? (
                      <div className="flex justify-center items-center h-40">
                        <span className="text-white">加载中...</span>
                      </div>
                    ) : (
                      recommendations.slice(0, 3).map(item => {
                        const overallScore = calculateOverallScore(item.scores);
                        
                        return (
                          <div key={item.id} className="bg-gray-800 rounded-lg overflow-hidden">
                            <div className="p-3">
                              <h3 className="text-white font-medium">{item.title}</h3>
                              <p className="text-gray-400 text-sm mt-1 line-clamp-2">{item.description}</p>
                              
                              <div className="flex flex-wrap gap-1 mt-2">
                                {item.tags.map((tag: string, idx: number) => (
                                  <Badge key={idx} variant="secondary" className="bg-gray-700 text-xs">
                                    {tag}
                                  </Badge>
                                ))}
                              </div>
                              
                              <div className="flex items-center justify-between mt-3">
                                <div className="flex items-center">
                                  <UserAvatar 
                                    username={`用户${item.userId}`}
                                    size="sm"
                                    avatarSrc={`https://api.dicebear.com/7.x/avataaars/svg?seed=${item.userId}`}
                                    className="w-6 h-6 mr-2" 
                                  />
                                  <span className="text-gray-300 text-xs">用户{item.userId}</span>
                                </div>
                                <div className="bg-gray-700 rounded-full px-2 py-1">
                                  <div className="text-xs text-white flex items-center">
                                    <BarChart3 className="w-3 h-3 mr-1" />
                                    <span>{(overallScore * 100).toFixed(0)}%</span>
                                  </div>
                                </div>
                              </div>
                            </div>
                            
                            <div className="bg-gray-700/50 px-3 py-2">
                              <div className="grid grid-cols-2 gap-2 text-xs">
                                {Object.entries(item.scores)
                                  .filter(([key]) => objectiveVisibility[key])
                                  .map(([key, score]) => (
                                    <div key={key} className="flex items-center justify-between">
                                      <span className="text-gray-300">
                                        {objectiveOptions.find(o => o.id === key)?.name}:
                                      </span>
                                      <span className="text-white font-medium">
                                        {(score as number * 100).toFixed(0)}%
                                      </span>
                                    </div>
                                  ))
                                }
                              </div>
                            </div>
                          </div>
                        );
                      })
                    )}
                  </CardContent>
                </Card>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
      
      <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
        <DialogContent className="sm:max-w-md">
          <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default MultiObjectiveRecommend;