import React, { useState } from "react";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Heart } from "lucide-react";
import UserAvatar from "@/components/UserAvatar";

interface CommentProps {
  id: string;
  username: string;
  avatar: string;
  content: string;
  likes: number;
  timestamp: string;
}

interface CommentSectionProps {
  isOpen: boolean;
  onClose: () => void;
  videoId: string;
}

const DUMMY_COMMENTS: CommentProps[] = [
  {
    id: "1",
    username: "抖音用户123456",
    avatar: "",
    content: "这个视频太棒了！",
    likes: 1245,
    timestamp: "3小时前",
  },
  {
    id: "2",
    username: "旅行爱好者",
    avatar: "",
    content: "哪里拍的？想去",
    likes: 872,
    timestamp: "2小时前",
  },
  {
    id: "3",
    username: "创作者666",
    avatar: "",
    content: "拍的真好看，请问用的什么滤镜？",
    likes: 543,
    timestamp: "1小时前",
  },
  {
    id: "4",
    username: "音乐达人",
    avatar: "",
    content: "BGM是什么歌？好听",
    likes: 267,
    timestamp: "45分钟前",
  },
];

const CommentSection: React.FC<CommentSectionProps> = ({
  isOpen,
  onClose,
  videoId,
}) => {
  const [comments, setComments] = useState<CommentProps[]>(DUMMY_COMMENTS);
  const [newComment, setNewComment] = useState("");
  const [likedComments, setLikedComments] = useState<Set<string>>(new Set());

  const handleSendComment = () => {
    if (newComment.trim()) {
      const comment: CommentProps = {
        id: Date.now().toString(),
        username: "你",
        avatar: "",
        content: newComment,
        likes: 0,
        timestamp: "刚刚",
      };
      setComments([comment, ...comments]);
      setNewComment("");
    }
  };

  const handleLikeComment = (commentId: string) => {
    setComments(
      comments.map((comment) => {
        if (comment.id === commentId) {
          const alreadyLiked = likedComments.has(commentId);
          const newLikes = alreadyLiked ? comment.likes - 1 : comment.likes + 1;

          // Update the set of liked comments
          const newLikedComments = new Set(likedComments);
          if (alreadyLiked) {
            newLikedComments.delete(commentId);
          } else {
            newLikedComments.add(commentId);
          }
          setLikedComments(newLikedComments);

          return { ...comment, likes: newLikes };
        }
        return comment;
      })
    );
  };

  return (
    <Sheet open={isOpen} onOpenChange={onClose}>
      <SheetContent side="bottom" className="h-[80vh] p-0 rounded-t-xl bg-white">
        <SheetHeader className="px-4 py-3 border-b border-gray-100">
          <SheetTitle className="text-center text-black">
            {comments.length} 条评论
          </SheetTitle>
        </SheetHeader>
        <div className="flex-1 overflow-y-auto h-[calc(80vh-130px)] px-4 py-2">
          {comments.map((comment) => (
            <div key={comment.id} className="flex py-3 border-b border-gray-100">
              <UserAvatar 
                username={comment.username}
                avatarSrc={comment.avatar || `https://api.dicebear.com/7.x/avataaars/svg?seed=${comment.username}`}
                size="md"
                className="h-10 w-10 mr-3"
              />
              <div className="flex-1">
                <p className="text-sm text-gray-500">{comment.username}</p>
                <p className="text-sm py-1 text-black">{comment.content}</p>
                <div className="flex items-center text-xs text-gray-400">
                  <span>{comment.timestamp}</span>
                  <button
                    className="ml-4 flex items-center"
                    onClick={() => handleLikeComment(comment.id)}
                  >
                    回复
                  </button>
                </div>
              </div>
              <button
                className="flex flex-col items-center justify-center ml-2"
                onClick={() => handleLikeComment(comment.id)}
              >
                <Heart
                  size={16}
                  fill={likedComments.has(comment.id) ? "#fe2c55" : "transparent"}
                  className={
                    likedComments.has(comment.id) ? "text-[#fe2c55]" : "text-gray-400"
                  }
                />
                <span className="text-xs mt-1 text-gray-400">
                  {comment.likes}
                </span>
              </button>
            </div>
          ))}
        </div>
        <div className="absolute bottom-0 left-0 right-0 border-t bg-white px-4 py-3">
          <div className="flex space-x-2">
            <Input
              placeholder="添加评论..."
              value={newComment}
              onChange={(e) => setNewComment(e.target.value)}
              className="flex-1 rounded-full bg-gray-100 border-none"
            />
            <Button 
              onClick={handleSendComment}
              className={`${newComment.trim() ? 'bg-[#fe2c55]' : 'bg-gray-200 text-gray-400'} hover:bg-[#fe2c55]/90`}
              disabled={!newComment.trim()}
            >
              发送
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  );
};

export default CommentSection;
