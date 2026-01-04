import React from "react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { useCurrentUser } from "@/lib/api";
import { useState, useEffect } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { format } from "date-fns";
import axios from "axios";
import { config } from "@/lib/config";

// 定义浏览历史记录的类型
interface BrowsingHistoryItem {
  id: number;
  title: string;
  type: string;
  timestamp: Date;
  imageUrl: string;
}

// 模拟浏览记录数据
const mockHistoryData = [
  {
    id: 1,
    title: "人工智能入门知识",
    type: "知识库",
    timestamp: new Date(2023, 10, 15, 14, 30),
    imageUrl: "/mockImages/ai-intro.jpg",
  },
  {
    id: 2,
    title: "机器学习算法详解",
    type: "视频",
    timestamp: new Date(2023, 10, 14, 10, 15),
    imageUrl: "/mockImages/ml-algorithms.jpg",
  },
  {
    id: 3,
    title: "深度学习与神经网络",
    type: "知识库",
    timestamp: new Date(2023, 10, 12, 9, 45),
    imageUrl: "/mockImages/deep-learning.jpg",
  },
  {
    id: 4,
    title: "Python编程基础",
    type: "视频",
    timestamp: new Date(2023, 10, 10, 16, 20),
    imageUrl: "/mockImages/python-basics.jpg",
  },
  {
    id: 5,
    title: "数据结构与算法",
    type: "知识库",
    timestamp: new Date(2023, 10, 8, 11, 10),
    imageUrl: "/mockImages/data-structures.jpg",
  },
];

// API函数：获取用户浏览历史
export const getUserBrowsingHistory = async (): Promise<BrowsingHistoryItem[]> => {
  if (config.useMockData) {
    // 使用模拟数据
    return mockHistoryData;
  } else {
    // 从后端API获取实际数据
    try {
      const response = await axios.get(`${config.apiBaseUrl}/api/user/browsing-history`);
      // 转换时间戳为Date对象
      return response.data.map((item: any) => ({
        ...item,
        timestamp: new Date(item.timestamp)
      }));
    } catch (error) {
      console.error("获取浏览历史失败:", error);
      throw error;
    }
  }
};

// React Query Hook：获取浏览历史
export const useUserBrowsingHistory = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [historyData, setHistoryData] = useState<BrowsingHistoryItem[]>([]);

  const fetchHistory = async () => {
    setIsLoading(true);
    try {
      const data = await getUserBrowsingHistory();
      setHistoryData(data);
      setError(null);
    } catch (err) {
      setError(err as Error);
      // 发生错误时，可以选择使用mock数据作为备用
      setHistoryData(mockHistoryData);
    } finally {
      setIsLoading(false);
    }
  };

  // 组件加载时获取数据
  useEffect(() => {
    fetchHistory();
  }, []);

  return { historyData, isLoading, error, refetch: fetchHistory };
};

const BrowsingHistory: React.FC = () => {
  const { data: currentUser, isLoading: isUserLoading } = useCurrentUser();
  // 使用新的hook替换原来的useState
  const { historyData, isLoading: isHistoryLoading, error } = useUserBrowsingHistory();
  const [isLoginOpen, setIsLoginOpen] = useState(false);

  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };

  // 加载状态处理
  if (isUserLoading || isHistoryLoading) {
    return (
      <div className="min-h-screen bg-[#121212] flex items-center justify-center">
        <div className="text-white">加载中...</div>
      </div>
    );
  }

  // 错误处理
  if (error) {
    console.error("浏览历史加载错误:", error);
    // 可以选择显示错误信息或继续使用可能已经加载的mock数据
  }

  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onLoginClick={handleLoginClick} />
      <LeftSidebar />
      <div className="ml-64 mt-16 p-8">
        <h1 className="text-2xl font-bold text-white mb-6">浏览记录</h1>
        
        {historyData.length > 0 ? (
          <div className="space-y-4">
            {historyData.map((item) => (
              <Card key={item.id} className="bg-[#1E1E1E] border-0 overflow-hidden hover:bg-[#2A2A2A] transition-colors">
                <CardContent className="p-4 flex items-center">
                  <div className="w-20 h-20 rounded bg-gray-700 overflow-hidden mr-4 flex-shrink-0">
                    <img 
                      src={item.imageUrl} 
                      alt={item.title}
                      className="w-full h-full object-cover"
                      onError={(e) => {
                        (e.target as HTMLImageElement).src = "/placeholder.jpg";
                      }}
                    />
                  </div>
                  <div className="flex-grow">
                    <h3 className="text-lg font-medium text-white">{item.title}</h3>
                    <div className="flex items-center mt-1">
                      <span className="text-sm text-gray-400 mr-4">{item.type}</span>
                      <span className="text-sm text-gray-400">
                        {format(item.timestamp, "yyyy-MM-dd HH:mm")}
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        ) : (
          <div className="text-center py-16">
            <p className="text-gray-400">暂无浏览记录</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default BrowsingHistory; 