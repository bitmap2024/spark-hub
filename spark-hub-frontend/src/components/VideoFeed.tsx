import React, { useState, useRef, useEffect } from "react";
import VideoPlayer from "./VideoPlayer";
import ActionBar from "./ActionBar";
import VideoInfo from "./VideoInfo";

interface Video {
  id: string;
  src: string;
  username: string;
  avatar: string;
  date: string;
  title: string;
  hashtags: string[];
  likes: number;
  comments: number;
  favorites: number;
  shares: number;
  episode?: string;
  series?: string;
  nextEpisode?: string;
}

const DUMMY_VIDEOS: Video[] = [
  {
    id: "1",
    src: "https://assets.mixkit.co/videos/preview/mixkit-young-woman-practicing-yoga-at-sunset-39760-large.mp4",
    username: "月下宅女",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=月下宅女",
    date: "3月6日",
    title: "没后续，快跑",
    hashtags: ["因为一个片段看了整部剧", "好剧推荐", "超好看的剧强烈推荐"],
    likes: 49000,
    comments: 904,
    favorites: 7554,
    shares: 1446,
    episode: "第1集",
    series: "超好看有上头的",
    nextEpisode: "下一集"
  },
  {
    id: "2",
    src: "https://assets.mixkit.co/videos/preview/mixkit-waves-coming-to-the-beach-5016-large.mp4",
    username: "travel_world",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=travel_world",
    date: "3月5日",
    title: "一剪梅，快跑",
    hashtags: ["旅行", "海滩", "夏天"],
    likes: 45600,
    comments: 234,
    favorites: 5432,
    shares: 456
  },
  {
    id: "3",
    src: "https://assets.mixkit.co/videos/preview/mixkit-top-aerial-shot-of-seashore-with-rocks-1090-large.mp4",
    username: "drone_master",
    avatar: "https://api.dicebear.com/7.x/avataaars/svg?seed=drone_master",
    date: "3月4日",
    title: "做人要听劝，快走",
    hashtags: ["航拍", "海岸", "自然"],
    likes: 78900,
    comments: 543,
    favorites: 8765,
    shares: 876
  },
];

const VideoFeed: React.FC = () => {
  const [activeVideoIndex, setActiveVideoIndex] = useState(0);
  const feedRef = useRef<HTMLDivElement>(null);

  const handleScroll = () => {
    if (feedRef.current) {
      const scrollTop = feedRef.current.scrollTop;
      const videoHeight = window.innerHeight - 64; // Subtract header height
      const index = Math.round(scrollTop / videoHeight);
      setActiveVideoIndex(index);
    }
  };

  useEffect(() => {
    const feedElement = feedRef.current;
    if (feedElement) {
      feedElement.addEventListener("scroll", handleScroll);
      return () => feedElement.removeEventListener("scroll", handleScroll);
    }
  }, []);

  return (
    <div
      ref={feedRef}
      className="h-[calc(100vh-64px)] mt-16 ml-64 w-[calc(100vw-256px)] overflow-y-scroll snap-mandatory snap-y hide-scrollbar"
    >
      {DUMMY_VIDEOS.map((video, index) => (
        <div
          key={video.id}
          className="h-[calc(100vh-64px)] w-full snap-start relative"
        >
          <div className="relative h-full">
            <VideoPlayer
              src={video.src}
              isActive={index === activeVideoIndex}
            />
            <div className="absolute top-0 left-0 right-0 z-10 p-4">
              <div className="flex flex-wrap">
                {index === 0 && (
                  <>
                    <div className="bg-black/50 text-white px-4 py-1 rounded-full text-sm mr-2 mb-2">一剪梅，快跑</div>
                    <div className="bg-black/50 text-white px-4 py-1 rounded-full text-sm mr-2 mb-2">没有后续</div>
                    <div className="bg-black/50 text-white px-4 py-1 rounded-full text-sm mr-2 mb-2">没后续，快跑</div>
                    <div className="bg-black/50 text-white px-4 py-1 rounded-full text-sm mr-2 mb-2">后续女主才是强者，穿到男主梦里解救男主</div>
                    <div className="bg-black/50 text-white px-4 py-1 rounded-full text-sm mr-2 mb-2">看完感觉剧情就是霸道</div>
                  </>
                )}
              </div>
            </div>
            <VideoInfo video={video} />
            <ActionBar video={video} />
          </div>
        </div>
      ))}
    </div>
  );
};

export default VideoFeed;
