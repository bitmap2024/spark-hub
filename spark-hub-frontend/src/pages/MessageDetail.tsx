import React, { useState, useEffect, useRef } from "react";
import { useParams, useNavigate } from "react-router-dom";
import Header from "@/components/Header";
import { useCurrentUser, useMessages, useSendMessage, useMarkMessagesAsRead, users } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { formatDistanceToNow } from "date-fns";
import { zhCN } from "date-fns/locale";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import { Send } from "lucide-react";
import UserAvatar from "@/components/UserAvatar";

const MessageDetail: React.FC = () => {
  const { userId } = useParams<{ userId: string }>();
  const navigate = useNavigate();
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [message, setMessage] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const { data: currentUser } = useCurrentUser();
  const otherUserId = parseInt(userId || "0");
  const otherUser = users.find(user => user.id === otherUserId);
  
  const { data: messages, isLoading } = useMessages(currentUser?.id || 0, otherUserId);
  const sendMessageMutation = useSendMessage();
  const markAsReadMutation = useMarkMessagesAsRead();
  
  const handleLoginClick = () => {
    setIsLoginOpen(true);
  };
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  useEffect(() => {
    if (currentUser && otherUserId) {
      markAsReadMutation.mutate({ userId1: currentUser.id, userId2: otherUserId });
    }
  }, [currentUser, otherUserId, markAsReadMutation]);
  
  const handleSendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!message.trim() || !currentUser) return;
    
    sendMessageMutation.mutate({
      senderId: currentUser.id,
      receiverId: otherUserId,
      content: message.trim()
    });
    
    setMessage("");
  };
  
  if (isLoading) {
    return (
      <div className="min-h-screen bg-background">
        <Header onLoginClick={handleLoginClick} />
        <div className="container mx-auto py-8">
          <div className="flex justify-center items-center h-64">
            <p>加载中...</p>
          </div>
        </div>
        <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
          <DialogContent className="sm:max-w-md">
            <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
          </DialogContent>
        </Dialog>
      </div>
    );
  }
  
  if (!otherUser) {
    return (
      <div className="min-h-screen bg-background">
        <Header onLoginClick={handleLoginClick} />
        <div className="container mx-auto py-8">
          <div className="flex justify-center items-center h-64">
            <p>用户不存在</p>
          </div>
        </div>
        <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
          <DialogContent className="sm:max-w-md">
            <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
          </DialogContent>
        </Dialog>
      </div>
    );
  }
  
  return (
    <div className="min-h-screen bg-background flex flex-col">
      <Header onLoginClick={handleLoginClick} />
      
      <div className="flex-1 container mx-auto py-8 flex flex-col h-[calc(100vh-4rem)]">
        <div className="flex items-center gap-3 mb-6">
          <Button variant="ghost" size="icon" onClick={() => navigate("/messages")}>
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="m15 18-6-6 6-6"/>
            </svg>
          </Button>
          <UserAvatar 
            username={otherUser.username}
            avatarSrc={otherUser.avatar}
            size="md"
          />
          <h1 className="text-xl font-bold">{otherUser.username}</h1>
        </div>
        
        <div className="flex-1 overflow-y-auto mb-4 space-y-4 p-4 bg-muted rounded-lg">
          {messages && messages.length > 0 ? (
            messages.map((msg) => {
              const isCurrentUser = msg.senderId === currentUser?.id;
              const timeAgo = formatDistanceToNow(new Date(msg.timestamp), { addSuffix: true, locale: zhCN });
              
              return (
                <div 
                  key={msg.id} 
                  className={`flex ${isCurrentUser ? "justify-end" : "justify-start"}`}
                >
                  <div className={`max-w-[70%] ${isCurrentUser ? "bg-primary text-primary-foreground" : "bg-background"} rounded-lg p-3`}>
                    <p>{msg.content}</p>
                    <p className="text-xs opacity-70 mt-1">{timeAgo}</p>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="flex justify-center items-center h-32">
              <p className="text-muted-foreground">暂无消息</p>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
        
        <form onSubmit={handleSendMessage} className="flex gap-2">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="输入消息..."
            className="flex-1"
          />
          <Button type="submit" size="icon" disabled={!message.trim()}>
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </div>
      
      <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
        <DialogContent className="sm:max-w-md">
          <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default MessageDetail; 