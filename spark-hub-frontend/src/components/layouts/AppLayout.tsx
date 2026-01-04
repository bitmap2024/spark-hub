import React, { ReactNode } from "react";
import Header from "../Header";
import { useNavigate } from "react-router-dom";

interface AppLayoutProps {
  children: ReactNode;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const navigate = useNavigate();

  const handleSearch = (query: string) => {
    if (query.trim()) {
      navigate(`/search?q=${encodeURIComponent(query.trim())}`);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#121212] text-white">
      <Header onSearch={handleSearch} />
      <main className="flex-1 pt-16">
        {children}
      </main>
    </div>
  );
};

export default AppLayout; 