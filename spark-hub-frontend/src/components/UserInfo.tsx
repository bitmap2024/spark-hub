import React from "react";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Music } from "lucide-react";
import { useNavigate } from "react-router-dom";

interface UserInfoProps {
  username: string;
  nickname: string;
  caption: string;
  avatarSrc: string;
  musicName: string;
}

const UserInfo: React.FC<UserInfoProps> = ({
  username,
  nickname,
  caption,
  avatarSrc,
  musicName,
}) => {
  const navigate = useNavigate();

  const handleAvatarClick = () => {
    navigate(`/user/${username}`);
  };

  return (
    <div className="absolute bottom-24 left-4 z-20 w-[calc(100%-100px)]">
      <div className="flex items-center space-x-2 mb-3">
        <Avatar 
          className="h-12 w-12 border-2 border-white ring-1 ring-primary cursor-pointer" 
          onClick={handleAvatarClick}
        >
          <AvatarImage src={avatarSrc} />
          <AvatarFallback>{username.charAt(0).toUpperCase()}</AvatarFallback>
        </Avatar>
        <div className="flex-1">
          <p className="font-bold text-white text-base">
            {username}
          </p>
          <p className="font-normal text-white/80 text-xs">
            {nickname}
          </p>
        </div>
        <Button variant="outline" size="sm" className="rounded-full border border-primary text-primary hover:bg-primary/10">
          关注
        </Button>
      </div>
      <p className="text-white text-sm mb-3">{caption}</p>
      <div className="flex items-center text-white/90 text-xs">
        <Music className="h-3 w-3 mr-2" />
        <div className="marquee-container overflow-hidden w-64">
          <p className="whitespace-nowrap animate-marquee">{musicName}</p>
        </div>
      </div>
    </div>
  );
};

export default UserInfo;
