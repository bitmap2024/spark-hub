import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Header from "@/components/Header";
import { useCurrentUser, useConversations, users } from "@/lib/api";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { formatDistanceToNow } from "date-fns";
import { zhCN } from "date-fns/locale";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "@/components/EmailLoginForm";
import UserAvatar from "@/components/UserAvatar";

const Messages: React.FC = () => {
  const navigate = useNavigate();
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const { data: currentUser } = useCurrentUser();
  const { data: conversations, isLoading } = useConversations(currentUser?.id || 0);

  const handleLoginClick = () => {
    setIsLoginOpen(true);
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

  if (!conversations || conversations.length === 0) {
    return (
      <div className="min-h-screen bg-background">
        <Header onLoginClick={handleLoginClick} />
        <div className="container mx-auto py-8">
          <h1 className="text-2xl font-bold mb-6">我的私信</h1>
          <div className="flex justify-center items-center h-64 bg-muted rounded-lg">
            <p className="text-muted-foreground">暂无私信</p>
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
    <div className="min-h-screen bg-background">
      <Header onLoginClick={handleLoginClick} />
      <div className="container mx-auto py-8">
        <h1 className="text-2xl font-bold mb-6">我的私信</h1>
        <div className="grid gap-4">
          {conversations.map((conversation) => {
            const otherUserId = conversation.participants.find(id => id !== currentUser?.id) || 0;
            const otherUser = users.find(user => user.id === otherUserId);
            
            if (!otherUser) return null;
            
            const lastMessageTime = new Date(conversation.lastMessage.timestamp);
            const timeAgo = formatDistanceToNow(lastMessageTime, { addSuffix: true, locale: zhCN });
            
            return (
              <Card 
                key={conversation.id} 
                className="cursor-pointer hover:bg-muted/50 transition-colors"
                onClick={() => navigate(`/messages/${otherUserId}`)}
              >
                <CardContent className="p-4 flex items-center gap-4">
                  <UserAvatar 
                    username={otherUser.username}
                    avatarSrc={otherUser.avatar}
                    size="md"
                    onClick={(e) => {
                      e.stopPropagation();
                      navigate(`/user/${otherUser.username}`);
                    }}
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex justify-between items-center">
                      <h3 className="font-medium truncate">{otherUser.username}</h3>
                      <span className="text-xs text-muted-foreground">{timeAgo}</span>
                    </div>
                    <p className="text-sm text-muted-foreground truncate">
                      {conversation.lastMessage.content}
                    </p>
                  </div>
                  {conversation.unreadCount > 0 && (
                    <Badge variant="default" className="ml-2">
                      {conversation.unreadCount}
                    </Badge>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
      <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
        <DialogContent className="sm:max-w-md">
          <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
        </DialogContent>
      </Dialog>
    </div>
  );
};

export default Messages; 