import React from "react";
import { Link } from "react-router-dom";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Search } from "lucide-react";

interface UserProfileProps {
  username: string;
  followers: number;
  following: number;
  likes: number;
  userId: string;
  avatarSrc: string;
  description: string;
  isCurrentUser?: boolean;
  location?: string;
}

const UserProfile: React.FC<UserProfileProps> = ({
  username,
  followers,
  following,
  likes,
  userId,
  avatarSrc,
  description,
  isCurrentUser = false,
  location = "",
}) => {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* 顶部导航栏 - 固定不变 */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div className="text-xl font-bold">{username}</div>
        <div className="flex items-center space-x-4">
          <Button variant="ghost" className="text-white">
            分享主页
          </Button>
          <Button variant="ghost" className="text-white font-bold">
            ...
          </Button>
        </div>
      </div>

      {/* 用户资料区域 */}
      <div className="px-4 py-6">
        <div className="flex items-center mb-4">
          <Avatar className="h-20 w-20 mr-4 border-2 border-white">
            <AvatarImage src={avatarSrc} alt={username} />
            <AvatarFallback>{username.charAt(0).toUpperCase()}</AvatarFallback>
          </Avatar>
          <div>
            <div className="text-xl font-semibold">{username}</div>
            <div className="text-gray-400">抖音号：{userId}</div>
            <div className="text-gray-400 mt-1">{location}</div>
          </div>
        </div>

        {/* 关注信息 */}
        <div className="flex items-center space-x-6 mb-4">
          <div className="flex flex-col items-center">
            <span className="font-bold">{following}</span>
            <span className="text-sm text-gray-400">关注</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="font-bold">{followers / 10000}万</span>
            <span className="text-sm text-gray-400">粉丝</span>
          </div>
          <div className="flex flex-col items-center">
            <span className="font-bold">{likes / 10000}万</span>
            <span className="text-sm text-gray-400">获赞</span>
          </div>
        </div>

        {/* 个人描述 */}
        <div className="mb-6">
          <p className="text-sm">{description}</p>
        </div>

        {/* 操作按钮 */}
        {isCurrentUser ? (
          <Button className="w-full" variant="outline">
            编辑资料
          </Button>
        ) : (
          <div className="flex gap-3">
            <Button className="flex-1 bg-[#fe2c55] hover:bg-[#d9264a] border-none">
              关注
            </Button>
            <Button variant="outline" className="flex-1">
              私信
            </Button>
          </div>
        )}
      </div>

      {/* 作品/喜欢 切换区域 */}
      <div className="flex border-b border-gray-800 mt-4">
        <button className="flex-1 py-3 text-center font-medium border-b-2 border-white">
          作品 239
        </button>
        <button className="flex-1 py-3 text-center font-medium text-gray-400 border-b-2 border-transparent">
          喜欢 3.9万
        </button>
      </div>

      {/* 搜索栏 */}
      <div className="p-4">
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-4 w-4 text-gray-400" />
          </div>
          <input
            type="text"
            placeholder="搜索 Ta 的作品"
            className="w-full bg-gray-900 border border-gray-700 rounded-full py-2 pl-10 pr-4 text-sm"
          />
        </div>
      </div>

      {/* 作品展示区 - 这里只是示例，实际上会根据API返回数据渲染 */}
      <div className="p-4">
        <div className="grid grid-cols-3 gap-1">
          {/* 这里放作品缩略图，根据实际数据渲染 */}
        </div>
      </div>
    </div>
  );
};

export default UserProfile; 