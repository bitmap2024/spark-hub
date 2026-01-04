import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import KnowledgeBaseVideoFeed from "@/components/KnowledgeBaseVideoFeed";

const Recommend: React.FC = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const navigate = useNavigate();
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  const handleSearch = (query: string) => {
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };
  
  return (
    <div className="min-h-screen bg-[#121212]">
      <Header onLoginClick={handleLoginClick} onSearch={handleSearch} />
      <LeftSidebar />
      
      {/* 主体内容区域，右侧主区域布局 */}
      <div className="ml-64 mt-16">
        <KnowledgeBaseVideoFeed sourceType="recommend" />
      </div>
      
      <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
        <DialogContent className="sm:max-w-md">
          <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Recommend; 