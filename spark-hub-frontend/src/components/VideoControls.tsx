
import React from "react";
import { Play, Pause, Volume2, VolumeX } from "lucide-react";

const VideoControls: React.FC = () => {
  const [playing, setPlaying] = React.useState(true);
  const [muted, setMuted] = React.useState(false);

  const togglePlay = () => {
    setPlaying(!playing);
  };

  const toggleMute = () => {
    setMuted(!muted);
  };

  return (
    <div className="flex items-center justify-between h-12 px-4 bg-gradient-to-t from-black/80 to-transparent">
      <div className="flex items-center space-x-4">
        <button onClick={togglePlay} className="text-white">
          {playing ? (
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="6" y="4" width="4" height="16"></rect>
              <rect x="14" y="4" width="4" height="16"></rect>
            </svg>
          ) : (
            <Play className="h-6 w-6" />
          )}
        </button>
        <div className="text-white text-xs">
          <span>00:12</span> <span className="text-gray-400">/ 05:36</span>
        </div>
        <button onClick={toggleMute} className="text-white">
          {muted ? (
            <VolumeX className="h-5 w-5" />
          ) : (
            <Volume2 className="h-5 w-5" />
          )}
        </button>
      </div>
      
      <div className="flex items-center space-x-4 text-white">
        <button className="text-white text-xs">连播</button>
        <button className="text-white text-xs">清屏</button>
        <button className="text-white text-xs">智能</button>
        <button className="text-white text-xs">倍速</button>
        <button className="text-white">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 2h12a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2z"></path>
            <path d="M12 18v-1"></path>
          </svg>
        </button>
        <button className="text-white">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
          </svg>
        </button>
        <button className="text-white">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M15 3h6v6"></path>
            <path d="M10 14 21 3"></path>
            <path d="M9 21h6"></path>
            <path d="M3 9v6"></path>
          </svg>
        </button>
      </div>
    </div>
  );
};

export default VideoControls;
