
import React, { useRef, useEffect } from "react";

interface VideoPlayerProps {
  src: string;
  isActive: boolean;
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({ src, isActive }) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  // Play/pause based on visibility
  useEffect(() => {
    if (videoRef.current) {
      if (isActive) {
        videoRef.current.play().catch(() => {
          // Autoplay was prevented, do nothing
        });
      } else {
        videoRef.current.pause();
      }
    }
  }, [isActive]);

  return (
    <div className="relative h-full w-full">
      <video
        ref={videoRef}
        className="h-full w-full object-cover"
        loop
        playsInline
        autoPlay={isActive}
        src={src}
      />
      <div className="absolute inset-0 bg-gradient-to-b from-black/20 via-transparent to-black/40" />
    </div>
  );
};

export default VideoPlayer;
