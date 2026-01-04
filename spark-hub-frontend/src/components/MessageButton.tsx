import React from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { MessageSquare } from "lucide-react";
import { useCurrentUser } from "@/lib/api";

interface MessageButtonProps {
  userId: number;
  onLoginClick: () => void;
}

const MessageButton: React.FC<MessageButtonProps> = ({ userId, onLoginClick }) => {
  const navigate = useNavigate();
  const { data: currentUser } = useCurrentUser();
  
  const handleSendMessage = () => {
    if (!currentUser) {
      onLoginClick();
      return;
    }
    
    navigate(`/messages/${userId}`);
  };
  
  return (
    <Button 
      variant="ghost" 
      className="rounded-full text-white ml-2"
      onClick={handleSendMessage}
    >
      <MessageSquare className="h-4 w-4 mr-1" />
      发送私信
    </Button>
  );
};

export default MessageButton; 