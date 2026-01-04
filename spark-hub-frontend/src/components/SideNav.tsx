
import React from "react";
import { Home, Search, Plus, User, MessageCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItemProps {
  icon: React.ReactNode;
  label: string;
  isActive?: boolean;
  onClick: () => void;
}

const NavItem: React.FC<NavItemProps> = ({ icon, label, isActive, onClick }) => {
  return (
    <button
      className={cn(
        "flex flex-col items-center justify-center py-2 w-full",
        isActive ? "text-[#fe2c55]" : "text-white"
      )}
      onClick={onClick}
    >
      {icon}
      <span className="text-xs mt-1">{label}</span>
    </button>
  );
};

const SideNav: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState("home");

  return (
    <div className="fixed bottom-0 left-0 right-0 bg-black border-t border-gray-800 z-50 pb-safe">
      <div className="flex justify-around max-w-screen-md mx-auto">
        <NavItem
          icon={<Home className="h-5 w-5" />}
          label="首页"
          isActive={activeTab === "home"}
          onClick={() => setActiveTab("home")}
        />
        <NavItem
          icon={<Search className="h-5 w-5" />}
          label="发现"
          isActive={activeTab === "discover"}
          onClick={() => setActiveTab("discover")}
        />
        <NavItem
          icon={
            <div className="bg-[#fe2c55] rounded-lg p-2 -mt-5 mb-1">
              <Plus className="h-5 w-5 text-white" />
            </div>
          }
          label=""
          onClick={() => {}}
        />
        <NavItem
          icon={<MessageCircle className="h-5 w-5" />}
          label="消息"
          isActive={activeTab === "inbox"}
          onClick={() => setActiveTab("inbox")}
        />
        <NavItem
          icon={<User className="h-5 w-5" />}
          label="我"
          isActive={activeTab === "profile"}
          onClick={() => setActiveTab("profile")}
        />
      </div>
    </div>
  );
};

export default SideNav;
