import React from "react";
import { useNavigate } from "react-router-dom";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

interface UserAvatarProps {
  username: string;
  avatarSrc: string;
  size?: "sm" | "md" | "lg";
  className?: string;
  onClick?: (e: React.MouseEvent<HTMLDivElement>) => void;
}

const UserAvatar: React.FC<UserAvatarProps> = ({
  username,
  avatarSrc,
  size = "md",
  className = "",
  onClick,
}) => {
  const navigate = useNavigate();

  const handleAvatarClick = (e: React.MouseEvent<HTMLDivElement>) => {
    if (onClick) {
      onClick(e);
    } else {
      navigate(`/user/${username}`);
    }
  };

  // 根据size设置尺寸
  const sizeClasses = {
    sm: "h-6 w-6",
    md: "h-10 w-10",
    lg: "h-14 w-14",
  };

  return (
    <Avatar 
      className={`${sizeClasses[size]} cursor-pointer ${className}`} 
      onClick={handleAvatarClick}
    >
      <AvatarImage src={avatarSrc} alt={username} />
      <AvatarFallback>{username.charAt(0).toUpperCase()}</AvatarFallback>
    </Avatar>
  );
};

export default UserAvatar; 