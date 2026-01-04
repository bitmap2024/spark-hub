import React, { useState, useRef, useEffect } from "react";
import { Search, MessageSquare, Bell, ChevronUp, ChevronDown } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { useTranslation } from "react-i18next";
import LanguageSwitcher from "./LanguageSwitcher";

interface HeaderProps {
  onLoginClick?: () => void;
  onSearch?: (query: string) => void;
}

const Header: React.FC<HeaderProps> = ({ 
  onLoginClick = () => {}, 
  onSearch 
}) => {
  const { t } = useTranslation();
  const [searchQuery, setSearchQuery] = useState("");
  const [notificationCount, setNotificationCount] = useState(3);
  const [showNotifications, setShowNotifications] = useState(false);
  const [showAllMessages, setShowAllMessages] = useState(false);
  const notificationContainerRef = useRef<HTMLDivElement>(null);
  const notificationContentRef = useRef<HTMLDivElement>(null);
  const allMessagesRef = useRef<HTMLDivElement>(null);

  // 示例通知数据
  const notifications = [
    {
      id: 1,
      avatar: "https://via.placeholder.com/40",
      username: "笨笨",
      type: "赞了你的评论",
      date: "04-06",
      isLike: true
    },
    {
      id: 2,
      avatar: "https://via.placeholder.com/40",
      username: "丁",
      type: "赞了你的评论",
      date: "04-04",
      isLike: true
    },
    {
      id: 3,
      avatar: "https://via.placeholder.com/40",
      username: "Z",
      type: "等2人赞了你的评论",
      date: "04-03",
      isLike: true,
      supporters: ["https://via.placeholder.com/30"]
    },
    {
      id: 4,
      avatar: "https://via.placeholder.com/40",
      username: "松果体",
      type: "赞了你的评论",
      date: "04-01",
      isLike: true
    },
    {
      id: 5, 
      avatar: "https://via.placeholder.com/40",
      username: "开心的小红牛",
      type: "等3人赞了你的评论",
      date: "03-30",
      isLike: true,
      supporters: ["https://via.placeholder.com/30", "https://via.placeholder.com/30"]
    }
  ];

  // 处理鼠标悬停显示通知
  const handleNotificationMouseEnter = () => {
    setShowNotifications(true);
  };

  // 处理鼠标移出隐藏通知
  const handleNotificationMouseLeave = (e: React.MouseEvent) => {
    // 检查鼠标是否真的离开了整个通知容器
    // 这里需要使用原生事件的relatedTarget来检查鼠标去向
    const relatedTarget = e.relatedTarget as Node;
    const container = notificationContainerRef.current;
    
    // 如果鼠标移动到的元素是容器内部的元素，不隐藏下拉框
    if (container && container.contains(relatedTarget)) {
      return;
    }
    
    setShowNotifications(false);
  };

  // 处理点击全部消息
  const toggleAllMessages = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setShowAllMessages(!showAllMessages);
  };

  // 点击外部关闭全部消息下拉栏
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (allMessagesRef.current && !allMessagesRef.current.contains(event.target as Node)) {
        setShowAllMessages(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, []);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (onSearch && searchQuery.trim()) {
      onSearch(searchQuery);
    }
  };

  return (
    <header className="fixed top-0 left-0 right-0 h-16 bg-[#121212] z-50 flex items-center justify-between px-4 border-b border-gray-800">
      <Link to="/" className="flex items-center space-x-2">
        <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 48 48" fill="none">
          <path d="M24.8 20.43C25.74 19.55 26.26 18.34 26.26 17C26.26 14.52 24.24 12.5 21.76 12.5C19.28 12.5 17.26 14.52 17.26 17H20.26C20.26 16.17 20.93 15.5 21.76 15.5C22.59 15.5 23.26 16.17 23.26 17C23.26 17.83 22.59 18.5 21.76 18.5H21V21.5H21.76C22.59 21.5 23.26 22.17 23.26 23C23.26 23.83 22.59 24.5 21.76 24.5C20.93 24.5 20.26 23.83 20.26 23H17.26C17.26 25.48 19.28 27.5 21.76 27.5C24.24 27.5 26.26 25.48 26.26 23C26.26 21.86 25.67 20.62 24.8 20.43Z" fill="white"/>
          <rect x="17" y="29" width="10" height="3" fill="white"/>
          <path d="M37.5 28V19C37.5 11.56 31.44 5.5 24 5.5C16.56 5.5 10.5 11.56 10.5 19V28C10.5 35.44 16.56 41.5 24 41.5C31.44 41.5 37.5 35.44 37.5 28ZM33.5 19.17V27.83C33.5 33.23 29.4 37.5 24 37.5C18.6 37.5 14.5 33.23 14.5 27.83V19.17C14.5 13.77 18.6 9.5 24 9.5C29.4 9.5 33.5 13.77 33.5 19.17Z" fill="#00F2EA"/>
        </svg>
        <span className="text-white text-xl font-bold">Spark Hub</span>
      </Link>
      
      <div className="flex-1 max-w-md mx-auto">
        <form onSubmit={handleSearchSubmit} className="relative">
          <input 
            type="text"
            placeholder={t('common.search')}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-[#2a2a2a] rounded-full py-2 pl-4 pr-10 text-white focus:outline-none"
          />
          <button type="submit" className="absolute right-3 top-1/2 transform -translate-y-1/2 text-white">
            <Search className="h-5 w-5" />
          </button>
        </form>
      </div>
      
      <div className="flex items-center space-x-4">
        <div className="flex items-center space-x-5">
          <LanguageSwitcher />
          {/* <Link to="/pricing" className="text-gray-300 text-sm hover:text-white cursor-pointer">{t('pricing.title')}</Link> */}
          <div 
            className="relative"
            ref={notificationContainerRef}
          >
            <a 
              href="#" 
              className="text-gray-300 hover:text-white flex items-center space-x-1 cursor-pointer relative"
              onMouseEnter={handleNotificationMouseEnter}
            >
              <Bell className="h-5 w-5" />
              {notificationCount > 0 && (
                <span className="absolute -top-2 -right-2 bg-[#fe2c55] text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                  {notificationCount}
                </span>
              )}
              <span className="text-sm">{t('common.notifications', '通知')}</span>
            </a>
            
            {showNotifications && (
              <div 
                className="absolute right-0 mt-2 w-80 bg-[#1a1a1a] rounded-lg shadow-xl overflow-hidden z-50"
                ref={notificationContentRef}
                onMouseLeave={handleNotificationMouseLeave}
              >
                <div className="flex justify-between items-center px-4 py-3 border-b border-gray-800">
                  <h3 className="text-white font-medium">{t('message.messages')}</h3>
                  <div className="relative" ref={allMessagesRef}>
                    <a 
                      href="#" 
                      className="text-white text-sm flex items-center space-x-1"
                      onClick={toggleAllMessages}
                    >
                      <span>{t('common.allMessages', '全部消息')}</span>
                      {showAllMessages ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
                    </a>
                    
                    {showAllMessages && (
                      <div className="absolute right-0 top-full mt-1 w-32 bg-[#252525] rounded shadow-lg overflow-hidden z-50">
                        <div className="py-2 flex flex-col text-sm">
                          <a href="#" className="text-white px-4 py-2 hover:bg-[#333333] flex items-center">
                            {t('common.allMessages', '全部消息')}
                          </a>
                          <a href="#" className="text-white px-4 py-2 hover:bg-[#333333] flex items-center">
                            {t('user.followers')}
                          </a>
                          <a href="#" className="text-white px-4 py-2 hover:bg-[#333333] flex items-center">
                            {t('common.mentions', '@我的')}
                          </a>
                          <a href="#" className="text-white px-4 py-2 hover:bg-[#333333] flex items-center">
                            {t('community.comments')}
                          </a>
                          <a href="#" className="text-white px-4 py-2 hover:bg-[#333333] flex items-center">
                            {t('user.likes')}
                          </a>
                          <a href="#" className="text-white px-4 py-2 hover:bg-[#333333] flex items-center">
                            {t('common.danmaku', '弹幕')}
                          </a>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
                
                <div className="max-h-96 overflow-y-auto">
                  {notifications.map(notification => (
                    <div key={notification.id} className="flex items-start p-4 hover:bg-[#252525] border-b border-gray-800">
                      <div className="relative">
                        <img src={notification.avatar} alt={notification.username} className="w-10 h-10 rounded-full object-cover" />
                        {notification.isLike && (
                          <div className="absolute -bottom-1 -right-1 bg-[#fe2c55] rounded-full p-1">
                            <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 text-white" viewBox="0 0 20 20" fill="currentColor">
                              <path d="M2 10.5a1.5 1.5 0 113 0v6a1.5 1.5 0 01-3 0v-6zM6 10.333v5.43a2 2 0 001.106 1.79l.05.025A4 4 0 008.943 18h5.416a2 2 0 001.962-1.608l1.2-6A2 2 0 0015.56 8H12V4a2 2 0 00-2-2 1 1 0 00-1 1v.667a4 4 0 01-.8 2.4L6.8 7.933a4 4 0 00-.8 2.4z" />
                            </svg>
                          </div>
                        )}
                      </div>
                      <div className="ml-3 flex-1">
                        <div className="flex justify-between">
                          <p className="text-white font-medium">{notification.username}</p>
                          <span className="text-gray-400 text-xs">{notification.date}</span>
                        </div>
                        <p className="text-gray-300 text-sm">{notification.type}</p>
                        {notification.supporters && (
                          <div className="mt-1 flex">
                            {notification.supporters.map((supporter, index) => (
                              <img key={index} src={supporter} alt="supporter" className="w-6 h-6 rounded-full -ml-1 first:ml-0 border border-[#1a1a1a]" />
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
          <a href="https://discord.gg/sparkhub" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-white flex items-center space-x-1 cursor-pointer">
            <MessageSquare className="h-5 w-5" />
            <span className="text-sm">Discord</span>
          </a>
        </div>
        <Button 
          onClick={onLoginClick}
          className="bg-[#fe2c55] hover:bg-[#fe2c55]/90 text-white rounded-full px-6"
        >
          {t('auth.login')}
        </Button>
      </div>
    </header>
  );
};

export default Header;