import { useNavigate } from "react-router-dom";

/**
 * 返回搜索处理函数，用于导航到搜索结果页面
 * @returns 搜索处理函数
 */
export const useSearchHandler = () => {
  const navigate = useNavigate();
  
  return (query: string) => {
    if (query && query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };
}; 