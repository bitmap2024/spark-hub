import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";

interface IndexProps {
  openLogin: () => void;
}

const Index: React.FC<IndexProps> = ({ openLogin }) => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  
  useEffect(() => {
    // 自动重定向到个人主页（展示知识库）
    navigate("/user/me");
  }, [navigate]);
  
  return (
    <div className="min-h-screen bg-[#121212] flex items-center justify-center">
      <div className="text-white">{t('common.loading')}</div>
    </div>
  );
};

export default Index;
