// 用户类型
export interface User {
  id: number;
  username: string;
  avatar: string;
  followers: number;
  following: number;
  followingList?: number[];
  likedKnowledgeBases?: number[]; // 用户喜欢的知识库ID
  location?: string;
  experience?: string;
  gender?: string;
  age?: number;
  school?: string;
}

// 浏览历史记录类型
export interface BrowsingHistoryItem {
  id: number;
  userId: number;
  contentId: number;
  contentType: string; // 'knowledge-base', 'paper', 'video' 等
  title: string;
  timestamp: string;
  imageUrl?: string;
}

// 消息类型
export interface Message {
  id: number;
  senderId: number;
  receiverId: number;
  content: string;
  timestamp: string;
  isRead: boolean;
}

// 会话类型
export interface Conversation {
  id: number;
  participants: number[];
  lastMessage: Message;
  unreadCount: number;
}

// 论文类型
export interface Paper {
  id: number;
  title: string;
  authors: string[];
  abstract: string;
  publishDate: string;
  doi?: string;
  url?: string;
}

// 知识库类型
export interface KnowledgeBase {
  id: number;
  title: string;
  description: string;
  userId: number;
  createdAt: string;
  updatedAt: string;
  papers: Paper[];
  tags: string[];
  stars: number;
  forks: number;
}
