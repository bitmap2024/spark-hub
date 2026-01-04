import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { ArrowLeft, ImagePlus, X, FileVideo, Eye, Edit, Code, Calculator } from "lucide-react";
import Header from "@/components/Header";
import LeftSidebar from "@/components/LeftSidebar";
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';
import { Post, addPost, getTimeAgo } from "@/lib/postUtils";

interface MediaFile {
  id: string;
  file: File;
  type: 'image' | 'video';
  previewUrl: string;
}

const CreatePost: React.FC = () => {
  const navigate = useNavigate();
  const [title, setTitle] = useState("");
  const [content, setContent] = useState("");
  const [media, setMedia] = useState<MediaFile[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [viewMode, setViewMode] = useState<'edit' | 'preview'>('edit');
  const imageInputRef = useRef<HTMLInputElement>(null);
  const videoInputRef = useRef<HTMLInputElement>(null);

  const handleBackToList = () => {
    navigate("/community");
  };

  const handleAddImage = () => {
    imageInputRef.current?.click();
  };

  const handleAddVideo = () => {
    videoInputRef.current?.click();
  };

  const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const newFiles: MediaFile[] = Array.from(e.target.files).map(file => ({
        id: `img-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        file,
        type: 'image',
        previewUrl: URL.createObjectURL(file)
      }));
      setMedia([...media, ...newFiles]);
    }
    // 重置input以便于重复选择相同文件
    if (e.target.value) e.target.value = '';
  };

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const videoFile = e.target.files[0];
      
      // 检查视频时长
      const video = document.createElement('video');
      video.preload = 'metadata';
      
      video.onloadedmetadata = () => {
        window.URL.revokeObjectURL(video.src);
        
        if (video.duration > 30) {
          alert('视频长度不能超过30秒');
          return;
        }
        
        const newFile: MediaFile = {
          id: `vid-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          file: videoFile,
          type: 'video',
          previewUrl: URL.createObjectURL(videoFile)
        };
        
        setMedia([...media, newFile]);
      };
      
      video.src = URL.createObjectURL(videoFile);
    }
    // 重置input以便于重复选择相同文件
    if (e.target.value) e.target.value = '';
  };

  const handleRemoveMedia = (id: string) => {
    setMedia(media.filter(item => item.id !== id));
  };

  const handleSubmit = async () => {
    if (!title.trim()) {
      alert('请输入标题');
      return;
    }

    if (!content.trim()) {
      alert('请输入内容');
      return;
    }

    setIsSubmitting(true);

    try {
      // 构建完整的内容，包括Markdown格式的媒体引用
      let finalContent = content.trim();
      
      // 添加媒体内容到Markdown文本
      if (media.length > 0) {
        finalContent += '\n\n';
        media.forEach(item => {
          if (item.type === 'image') {
            // 在此应该将本地图片上传到服务器并获取URL
            // 这里为了演示，我们直接使用本地预览URL
            finalContent += `![图片](${item.previewUrl})\n\n`;
          } else {
            // 同样，视频也应该上传到服务器
            finalContent += `<video src="${item.previewUrl}" controls style="max-width: 100%; height: auto;"></video>\n\n`;
          }
        });
      }

      // 构建新帖子对象
      const newPost: Post = {
        id: `p${Date.now()}`,
        title: title.trim(),
        content: finalContent,
        author: {
          id: "currentUser",
          username: "我",
          avatar: "",
        },
        createdAt: getTimeAgo(),
        likes: 0,
        comments: [],
      };
      
      // 添加帖子到本地存储并触发事件
      addPost(newPost);
      
      // 模拟服务器延迟
      await new Promise(resolve => setTimeout(resolve, 500));
      
      // 提示发布成功并跳转
      alert('发布成功');
      navigate('/community');
    } catch (error) {
      console.error('发布失败', error);
      alert('发布失败，请重试');
    } finally {
      setIsSubmitting(false);
    }
  };

  // 构建预览内容，包括嵌入媒体文件的预览
  const getPreviewContent = () => {
    // 如果有图片或视频，在Markdown内容的底部添加它们
    let previewContent = content;

    if (media.length > 0) {
      previewContent += '\n\n';
      media.forEach(item => {
        if (item.type === 'image') {
          previewContent += `![图片](${item.previewUrl})\n\n`;
        } else {
          previewContent += `<video src="${item.previewUrl}" controls style="max-width: 100%; height: auto;"></video>\n\n`;
        }
      });
    }

    return previewContent;
  };

  const insertMarkdownExample = () => {
    const example = `## Markdown 示例

**粗体文本** 和 *斜体文本*

### 列表示例
- 项目 1
- 项目 2
  - 子项目 A
  - 子项目 B

### 链接和图片
[链接文本](https://example.com)

### 代码示例
\`\`\`javascript
function hello() {
  console.log("Hello World!");
}
\`\`\`

> 这是一段引用文本
`;
    setContent(prev => prev + example);
  };

  const insertLatexExample = () => {
    const example = `

## LaTeX 公式示例

### 行内公式
Einstein 的质能方程: $E=mc^2$

### 独立公式
$$
\\frac{d}{dx}\\left( \\int_{0}^{x} f(u)\\,du\\right)=f(x)
$$

### 矩阵
$$
\\begin{pmatrix} 
a & b \\\\
c & d
\\end{pmatrix}
$$

### 多行公式
$$
\\begin{align}
f(x) &= (x+a)(x+b) \\\\
&= x^2 + (a+b)x + ab
\\end{align}
$$
`;
    setContent(prev => prev + example);
  };

  return (
    <div className="min-h-screen bg-[#f8f8f8]">
      <Header />
      <LeftSidebar />
      <div className="pt-16 pl-64">
        <div className="container mx-auto py-6 px-4">
          <div className="flex items-center mb-6">
            <button 
              onClick={handleBackToList}
              className="flex items-center text-gray-600 mr-4 bg-transparent border-none cursor-pointer"
            >
              <ArrowLeft className="mr-2 h-4 w-4" />
              返回社区
            </button>
            <h1 className="text-2xl font-bold text-gray-800">发布新帖</h1>
          </div>

          <div className="bg-white shadow-sm rounded-lg overflow-hidden">
            <div className="px-4 py-3 border-b">
              <input
                type="text"
                placeholder="你好明天"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                className="w-full text-xl font-medium outline-none border-none bg-white text-black"
              />
            </div>

            <div className="border-b flex">
              <button
                type="button"
                onClick={() => setViewMode('edit')}
                className={`flex items-center px-4 py-2 ${viewMode === 'edit' ? 'border-b-2 border-[#fe2c55] text-[#fe2c55]' : 'text-gray-600'}`}
              >
                <Edit className="h-4 w-4 mr-2" />
                编辑
              </button>
              <button
                type="button"
                onClick={() => setViewMode('preview')}
                className={`flex items-center px-4 py-2 ${viewMode === 'preview' ? 'border-b-2 border-[#fe2c55] text-[#fe2c55]' : 'text-gray-600'}`}
              >
                <Eye className="h-4 w-4 mr-2" />
                预览
              </button>
            </div>

            {viewMode === 'edit' ? (
              <div>
                <textarea
                  placeholder="支持 Markdown 和 LaTeX 格式编辑，可以添加图片和视频..."
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  className="w-full min-h-[300px] p-4 outline-none border-none resize-none bg-white text-black"
                />
              </div>
            ) : (
              <div className="p-4 min-h-[300px] markdown-preview bg-white text-black overflow-auto">
                {content.trim() ? (
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
                    {getPreviewContent()}
                  </ReactMarkdown>
                ) : (
                  <p className="text-gray-400">预览内容将显示在这里...</p>
                )}
              </div>
            )}

            {viewMode === 'edit' && media.length > 0 && (
              <div className="grid grid-cols-4 gap-4 p-4 border-t">
                {media.map(item => (
                  <div key={item.id} className="relative rounded-md overflow-hidden border">
                    {item.type === 'image' ? (
                      <img
                        src={item.previewUrl}
                        alt="预览图片"
                        className="w-full h-32 object-cover"
                      />
                    ) : (
                      <video
                        src={item.previewUrl}
                        className="w-full h-32 object-cover"
                        controls
                      />
                    )}
                    <button
                      type="button"
                      className="absolute top-1 right-1 bg-black bg-opacity-50 rounded-full p-1 border-none cursor-pointer"
                      onClick={() => handleRemoveMedia(item.id)}
                    >
                      <X className="h-4 w-4 text-white" />
                    </button>
                  </div>
                ))}
              </div>
            )}

            <div className="px-4 py-3 border-t flex items-center">
              <button
                type="button"
                onClick={handleAddImage}
                className="flex items-center mr-4 px-3 py-1 rounded border border-gray-300 bg-white text-gray-700"
              >
                <ImagePlus className="h-4 w-4 mr-2" />
                添加图片
              </button>
              <button
                type="button"
                onClick={handleAddVideo}
                className="flex items-center mr-4 px-3 py-1 rounded border border-gray-300 bg-white text-gray-700"
              >
                <FileVideo className="h-4 w-4 mr-2" />
                添加视频（&lt;30秒）
              </button>
              {viewMode === 'edit' && (
                <>
                  <button
                    type="button"
                    onClick={insertMarkdownExample}
                    className="flex items-center mr-4 px-3 py-1 rounded border border-gray-300 bg-white text-gray-700"
                  >
                    <Code className="h-4 w-4 mr-2" />
                    插入Markdown示例
                  </button>
                  <button
                    type="button"
                    onClick={insertLatexExample}
                    className="flex items-center px-3 py-1 rounded border border-gray-300 bg-white text-gray-700"
                  >
                    <Calculator className="h-4 w-4 mr-2" />
                    插入LaTeX示例
                  </button>
                </>
              )}
              <input
                type="file"
                ref={imageInputRef}
                className="hidden"
                accept="image/*"
                onChange={handleImageChange}
                multiple
              />
              <input
                type="file"
                ref={videoInputRef}
                className="hidden"
                accept="video/*"
                onChange={handleVideoChange}
              />
            </div>

            <div className="px-4 py-3 text-xs text-gray-500">
              <p>支持 Markdown 语法，可以使用 # 标题，**粗体**，*斜体*，[链接](url)，`代码` 等</p>
              <p>支持 LaTeX 公式，行内公式使用 $E=mc^2$，独立公式使用 $$E=mc^2$$</p>
              <p>视频长度不超过30秒，图片支持常见格式</p>
            </div>

            <div className="px-4 py-3 border-t flex justify-end">
              <button
                type="button"
                onClick={handleBackToList}
                className="px-4 py-1 mr-2 rounded border border-gray-300 bg-white text-gray-800"
              >
                取消
              </button>
              <button
                type="button"
                onClick={handleSubmit}
                disabled={isSubmitting || !title.trim() || !content.trim()}
                className="px-4 py-1 rounded bg-[#fe2c55] hover:bg-[#fe2c55]/90 text-white border-none disabled:opacity-50"
              >
                {isSubmitting ? "发布中..." : "发布"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CreatePost; 