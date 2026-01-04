import React from "react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

interface VideoInfoProps {
  video: {
    username: string;
    avatar: string;
    date: string;
    title: string;
    hashtags: string[];
    episode?: string;
    series?: string;
    nextEpisode?: string;
  };
}

const VideoInfo: React.FC<VideoInfoProps> = ({ video }) => {
  return (
    <div className="absolute bottom-16 left-4 right-[120px] z-20">
      <div className="flex items-center space-x-3">
        <Link to={`/user/${video.username}`}>
          <Avatar className="h-10 w-10 border-2 border-white cursor-pointer">
            <AvatarImage src={video.avatar} alt={video.username} />
            <AvatarFallback>{video.username.charAt(0).toUpperCase()}</AvatarFallback>
          </Avatar>
        </Link>
        <div className="flex items-center space-x-2">
          <Link to={`/user/${video.username}`} className="text-white font-medium">@{video.username}</Link>
          <span className="text-gray-300 text-sm">· {video.date}</span>
        </div>
      </div>

      {video.episode && (
        <div className="mt-2 text-white text-sm">
          {video.episode}：
          {video.hashtags.map((tag, index) => (
            <span key={index} className="text-[#2F80ED]">#{tag} </span>
          ))}
        </div>
      )}

      {video.series && (
        <div className="mt-4 flex items-center space-x-2">
          <div className="bg-white/20 text-white px-3 py-1 rounded-full text-sm flex items-center space-x-1">
            <span>{video.series}</span>
            <span>|</span>
            <span className="text-[#fe2c55]">更新至第6集</span>
          </div>
          
          <Button className="bg-white/20 hover:bg-white/30 text-white rounded-full px-3 py-1 h-auto text-sm">
            {video.nextEpisode} &gt;
          </Button>
        </div>
      )}

      <div className="fixed bottom-16 left-0 right-0 h-12 bg-gradient-to-b from-transparent to-black/80" />
    </div>
  );
};

export default VideoInfo;
