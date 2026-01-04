import React, { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogFooter } from "@/components/ui/dialog";
import UserAvatar from "@/components/UserAvatar";
import { Heart, MessageSquare, Send, Edit, ArrowLeft } from "lucide-react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import { Post, Comment, getStoredPosts, savePosts } from "@/lib/postUtils";
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';
import { useSearchHandler } from "@/lib/navigation";

const Community: React.FC = () => {
  const { postId } = useParams<{ postId: string }>();
  const navigate = useNavigate();
  const [posts, setPosts] = useState<Post[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showNewPostForm, setShowNewPostForm] = useState(false);
  const [viewType, setViewType] = useState<"all" | "following">("all");
  const [selectedTag, setSelectedTag] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<"latest" | "popular">("latest");
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);
  const [isViewing, setIsViewing] = useState(false);
  const [likedPosts, setLikedPosts] = useState<Set<string>>(new Set());
  const [likedComments, setLikedComments] = useState<Set<string>>(new Set());
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  const [newComment, setNewComment] = useState("");
  const handleSearch = useSearchHandler();

  // 初始化加载帖子
  useEffect(() => {
    const storedPosts = getStoredPosts();
    setPosts(storedPosts);
  }, []);

  // 监听新帖子事件
  useEffect(() => {
    const handlePostAdded = (event: Event) => {
      const customEvent = event as CustomEvent<Post>;
      setPosts(prevPosts => [customEvent.detail, ...prevPosts]);
    };

    window.addEventListener('post-added', handlePostAdded);
    return () => {
      window.removeEventListener('post-added', handlePostAdded);
    };
  }, []);

  // 当帖子状态变化时，保存到本地存储
  useEffect(() => {
    if (posts.length > 0) {
      savePosts(posts);
    }
  }, [posts]);

  // 当路由参数变化时，查找对应的帖子
  useEffect(() => {
    if (postId) {
      const post = posts.find(p => p.id === postId);
      if (post) {
        setSelectedPost(post);
      } else {
        // 如果找不到对应的帖子，重定向到社区首页
        navigate("/community");
      }
    } else {
      setSelectedPost(null);
    }
  }, [postId, posts, navigate]);

  // 处理点赞帖子
  const handleLikePost = (postId: string) => {
    setPosts(
      posts.map((post) => {
        if (post.id === postId) {
          const alreadyLiked = likedPosts.has(postId);
          const newLikes = alreadyLiked ? post.likes - 1 : post.likes + 1;

          // 更新点赞集合
          const newLikedPosts = new Set(likedPosts);
          if (alreadyLiked) {
            newLikedPosts.delete(postId);
          } else {
            newLikedPosts.add(postId);
          }
          setLikedPosts(newLikedPosts);

          return { ...post, likes: newLikes };
        }
        return post;
      })
    );
  };

  // 处理点赞评论
  const handleLikeComment = (postId: string, commentId: string) => {
    setPosts(
      posts.map((post) => {
        if (post.id === postId) {
          const updatedComments = post.comments.map((comment) => {
            if (comment.id === commentId) {
              const alreadyLiked = likedComments.has(commentId);
              const newLikes = alreadyLiked ? comment.likes - 1 : comment.likes + 1;

              // 更新点赞集合
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
          });
          return { ...post, comments: updatedComments };
        }
        return post;
      })
    );
  };

  // 添加评论
  const handleAddComment = (postId: string) => {
    if (newComment.trim()) {
      const comment: Comment = {
        id: `c${Date.now()}`,
        content: newComment,
        author: {
          id: "currentUser",
          username: "我",
          avatar: "",
        },
        createdAt: "刚刚",
        likes: 0,
      };

      setPosts(
        posts.map((post) => {
          if (post.id === postId) {
            return {
              ...post,
              comments: [comment, ...post.comments],
            };
          }
          return post;
        })
      );
      setNewComment("");
    }
  };

  // 查看帖子详情
  const handleViewPost = (postId: string) => {
    navigate(`/community/${postId}`);
  };

  // 返回帖子列表
  const handleBackToList = () => {
    navigate("/community");
  };

  // 处理创建新帖子页面跳转
  const handleCreatePostNavigation = () => {
    navigate("/community/create");
  };

  return (
    <div className="min-h-screen bg-[#121212]">
      <Header 
        onLoginClick={() => setIsLoginOpen(true)} 
        onSearch={handleSearch}
      />
      <LeftSidebar />
      <div className="pt-16 pl-64">
        <div className="container mx-auto py-6 px-4">
          <div className="flex justify-between items-center mb-6">
            {selectedPost ? (
              <Button 
                variant="ghost" 
                className="flex items-center text-gray-600"
                onClick={handleBackToList}
              >
                <ArrowLeft className="mr-2 h-4 w-4" />
                返回社区
              </Button>
            ) : (
              <h1 className="text-2xl font-bold text-gray-800">社区论坛</h1>
            )}
            
            {!selectedPost && (
              <Button 
                className="bg-[#fe2c55] hover:bg-[#fe2c55]/90"
                onClick={handleCreatePostNavigation}
              >
                <Edit className="mr-2 h-4 w-4" />
                发布帖子
              </Button>
            )}
          </div>

          {selectedPost ? (
            <div className="space-y-4">
              <Card className="bg-white shadow-sm">
                <CardHeader>
                  <div className="flex items-center space-x-3">
                    <UserAvatar
                      username={selectedPost.author.username}
                      avatarSrc={selectedPost.author.avatar || `https://api.dicebear.com/7.x/avataaars/svg?seed=${selectedPost.author.username}`}
                      size="md"
                    />
                    <div>
                      <p className="font-medium">{selectedPost.author.username}</p>
                      <p className="text-xs text-gray-500">{selectedPost.createdAt}</p>
                    </div>
                  </div>
                  <CardTitle className="text-xl mt-3">{selectedPost.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="markdown-content text-gray-700">
                    <ReactMarkdown
                      remarkPlugins={[remarkMath, remarkGfm]}
                      rehypePlugins={[rehypeKatex, rehypeRaw]}
                      components={{
                        // 标题样式
                        h1: ({node, ...props}) => <h1 style={{fontSize: '2em', fontWeight: 'bold', margin: '0.67em 0'}} {...props} />,
                        h2: ({node, ...props}) => <h2 style={{fontSize: '1.5em', fontWeight: 'bold', margin: '0.83em 0'}} {...props} />,
                        h3: ({node, ...props}) => <h3 style={{fontSize: '1.17em', fontWeight: 'bold', margin: '1em 0'}} {...props} />,
                        // 文本样式
                        p: ({node, ...props}) => <p style={{margin: '1em 0'}} {...props} />,
                        strong: ({node, ...props}) => <strong style={{fontWeight: 'bold'}} {...props} />,
                        em: ({node, ...props}) => <em style={{fontStyle: 'italic'}} {...props} />,
                        // 链接样式
                        a: ({node, ...props}) => <a style={{color: '#0366d6', textDecoration: 'none'}} {...props} />,
                        // 代码样式
                        code: ({node, inline, className, children, ...props}: any) => {
                          const match = /language-(\w+)/.exec(className || '');
                          return !inline ? (
                            <pre style={{background: '#f6f8fa', padding: '1em', borderRadius: '3px', overflowX: 'auto'}}>
                              <code style={{fontFamily: 'monospace'}} {...props}>
                                {children}
                              </code>
                            </pre>
                          ) : (
                            <code style={{background: '#f6f8fa', padding: '0.2em 0.4em', borderRadius: '3px', fontFamily: 'monospace'}} {...props}>
                              {children}
                            </code>
                          );
                        },
                        // 列表样式
                        ul: ({node, ...props}) => <ul style={{paddingLeft: '2em', margin: '1em 0'}} {...props} />,
                        ol: ({node, ...props}) => <ol style={{paddingLeft: '2em', margin: '1em 0'}} {...props} />,
                        li: ({node, ...props}) => <li style={{margin: '0.5em 0'}} {...props} />,
                        // 引用样式
                        blockquote: ({node, ...props}) => <blockquote style={{borderLeft: '4px solid #dfe2e5', paddingLeft: '1em', color: '#6a737d', margin: '1em 0'}} {...props} />,
                        // 图片样式
                        img: ({node, ...props}) => <img style={{maxWidth: '100%', height: 'auto'}} {...props} />,
                      }}
                    >
                      {selectedPost.content}
                    </ReactMarkdown>
                  </div>
                </CardContent>
                <CardFooter className="border-t pt-4 flex justify-between">
                  <Button
                    variant="ghost"
                    className="flex items-center"
                    onClick={() => handleLikePost(selectedPost.id)}
                  >
                    <Heart
                      size={16}
                      fill={likedPosts.has(selectedPost.id) ? "#fe2c55" : "transparent"}
                      className={
                        likedPosts.has(selectedPost.id) ? "text-[#fe2c55] mr-2" : "text-gray-500 mr-2"
                      }
                    />
                    <span className={likedPosts.has(selectedPost.id) ? "text-[#fe2c55]" : "text-gray-500"}>
                      {selectedPost.likes}
                    </span>
                  </Button>
                  <div className="flex items-center text-gray-500">
                    <MessageSquare size={16} className="mr-2" />
                    <span>{selectedPost.comments.length} 条评论</span>
                  </div>
                </CardFooter>
              </Card>

              <div className="bg-white p-4 rounded-lg shadow-sm">
                <h2 className="text-lg font-semibold mb-4">评论 ({selectedPost.comments.length})</h2>
                <div className="flex space-x-2 mb-6">
                  <UserAvatar
                    username="我"
                    avatarSrc={`https://api.dicebear.com/7.x/avataaars/svg?seed=currentUser`}
                    size="sm"
                  />
                  <div className="flex-1 flex">
                    <Input
                      placeholder="添加评论..."
                      value={newComment}
                      onChange={(e) => setNewComment(e.target.value)}
                      className="flex-1 rounded-l-md border-r-0"
                    />
                    <Button 
                      onClick={() => handleAddComment(selectedPost.id)}
                      className="bg-[#fe2c55] hover:bg-[#fe2c55]/90 rounded-l-none"
                      disabled={!newComment.trim()}
                    >
                      <Send size={16} />
                    </Button>
                  </div>
                </div>

                <div className="space-y-4">
                  {selectedPost.comments.map((comment) => (
                    <div key={comment.id} className="flex space-x-3 pb-3 border-b border-gray-100">
                      <UserAvatar
                        username={comment.author.username}
                        avatarSrc={comment.author.avatar || `https://api.dicebear.com/7.x/avataaars/svg?seed=${comment.author.username}`}
                        size="sm"
                      />
                      <div className="flex-1">
                        <div className="flex justify-between">
                          <p className="font-medium">{comment.author.username}</p>
                          <p className="text-xs text-gray-500">{comment.createdAt}</p>
                        </div>
                        <p className="text-gray-700 mt-1">{comment.content}</p>
                        <div className="flex items-center mt-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleLikeComment(selectedPost.id, comment.id)}
                            className="flex items-center h-6 px-2"
                          >
                            <Heart
                              size={14}
                              fill={likedComments.has(comment.id) ? "#fe2c55" : "transparent"}
                              className={
                                likedComments.has(comment.id) ? "text-[#fe2c55] mr-1" : "text-gray-500 mr-1"
                              }
                            />
                            <span className={likedComments.has(comment.id) ? "text-[#fe2c55] text-xs" : "text-gray-500 text-xs"}>
                              {comment.likes}
                            </span>
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="flex items-center h-6 px-2 text-gray-500 text-xs"
                          >
                            回复
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 gap-4">
              {posts.map((post) => (
                <Card key={post.id} className="bg-white shadow-sm hover:shadow-md transition-shadow cursor-pointer" onClick={() => handleViewPost(post.id)}>
                  <CardHeader>
                    <div className="flex items-center space-x-3">
                      <UserAvatar
                        username={post.author.username}
                        avatarSrc={post.author.avatar || `https://api.dicebear.com/7.x/avataaars/svg?seed=${post.author.username}`}
                        size="sm"
                      />
                      <div>
                        <p className="font-medium">{post.author.username}</p>
                        <p className="text-xs text-gray-500">{post.createdAt}</p>
                      </div>
                    </div>
                    <CardTitle className="text-lg mt-2">{post.title}</CardTitle>
                  </CardHeader>
                  <CardContent className="pb-2">
                    <div className="text-gray-700 line-clamp-2 markdown-preview-list">
                      {/* 在列表视图中，我们使用简化版的 Markdown 显示，主要用于文本预览 */}
                      {post.content.length > 150 ? (
                        // 显示简化的纯文本预览，避免截断导致的Markdown渲染问题
                        <p>{post.content.replace(/[#*`~_>]+/g, ' ').substring(0, 150)}...</p>
                      ) : (
                        <ReactMarkdown 
                          remarkPlugins={[remarkMath, remarkGfm]}
                          rehypePlugins={[rehypeKatex, rehypeRaw]}
                        >
                          {post.content}
                        </ReactMarkdown>
                      )}
                    </div>
                  </CardContent>
                  <CardFooter className="border-t pt-3 flex justify-between">
                    <div className="flex items-center text-gray-500">
                      <Heart
                        size={16}
                        fill={likedPosts.has(post.id) ? "#fe2c55" : "transparent"}
                        className={
                          likedPosts.has(post.id) ? "text-[#fe2c55] mr-1" : "text-gray-500 mr-1"
                        }
                      />
                      <span>{post.likes}</span>
                    </div>
                    <div className="flex items-center text-gray-500">
                      <MessageSquare size={16} className="mr-1" />
                      <span>{post.comments.length} 条评论</span>
                    </div>
                  </CardFooter>
                </Card>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Community; 