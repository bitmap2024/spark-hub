import React, { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle,
  DialogFooter
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Send, X, MessageSquare, Bot, HelpCircle, Settings, Minimize2, Maximize2 } from "lucide-react";

interface Message {
  id: number;
  text: string;
  sender: "user" | "assistant";
  timestamp: Date;
}

interface Position {
  x: number;
  y: number;
}

const AIAssistant: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [wasDragged, setWasDragged] = useState(false);
  const [position, setPosition] = useState<Position>(() => {
    // 从localStorage获取保存的位置，如果没有则使用默认位置
    const savedPosition = localStorage.getItem('aiAssistantPosition');
    return savedPosition 
      ? JSON.parse(savedPosition) 
      : { x: window.innerWidth - 80, y: window.innerHeight - 80 };
  });
  const dragRef = useRef<HTMLDivElement>(null);
  const initialMousePosRef = useRef<Position>({ x: 0, y: 0 });
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "你好！我是你的AI助手小美，有什么我可以帮你的吗？",
      sender: "assistant",
      timestamp: new Date()
    }
  ]);

  const handleSendMessage = () => {
    if (!inputValue.trim()) return;
    
    // 添加用户消息
    const userMessage: Message = {
      id: messages.length + 1,
      text: inputValue,
      sender: "user",
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue("");
    
    // 模拟AI回复
    setTimeout(() => {
      const assistantMessage: Message = {
        id: messages.length + 2,
        text: getAIResponse(inputValue),
        sender: "assistant",
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, assistantMessage]);
    }, 1000);
  };

  const getAIResponse = (userInput: string): string => {
    // 简单的响应逻辑，实际应用中会连接到AI服务
    const lowerInput = userInput.toLowerCase();
    
    if (lowerInput.includes("你好") || lowerInput.includes("嗨") || lowerInput.includes("hi")) {
      return "你好！很高兴见到你！";
    } else if (lowerInput.includes("名字")) {
      return "我叫小美，是你的AI助手！";
    } else if (lowerInput.includes("功能") || lowerInput.includes("能做什么")) {
      return "我可以帮你查找视频、推荐内容、回答问题，还可以陪你聊天哦！";
    } else if (lowerInput.includes("谢谢") || lowerInput.includes("感谢")) {
      return "不客气！随时都可以来找我聊天！";
    } else if (lowerInput.includes("再见") || lowerInput.includes("拜拜")) {
      return "再见！期待下次与你聊天！";
    } else {
      return "我明白你的意思了！作为你的AI助手，我会尽力帮助你。有什么具体的问题我可以回答吗？";
    }
  };

  // 处理拖动开始
  const handleMouseDown = (e: React.MouseEvent) => {
    if (isOpen) return; // 如果对话框已打开，不允许拖动
    
    // 记录鼠标初始位置
    initialMousePosRef.current = { x: e.clientX, y: e.clientY };
    setIsDragging(true);
    setWasDragged(false);
  };

  const handleClick = () => {
    // 只有在没有拖动的情况下才打开聊天窗口
    if (!wasDragged && !isDragging) {
      setIsOpen(true);
    }
  };

  // 处理拖动过程
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isDragging) return;
      
      // 计算移动距离
      const dx = Math.abs(e.clientX - initialMousePosRef.current.x);
      const dy = Math.abs(e.clientY - initialMousePosRef.current.y);
      
      // 如果移动超过5像素，认为是拖拽而非点击
      if (dx > 5 || dy > 5) {
        setWasDragged(true);
      }
      
      // 更新位置，确保不超出屏幕边界
      const newX = Math.max(0, Math.min(e.clientX, window.innerWidth - 80));
      const newY = Math.max(0, Math.min(e.clientY, window.innerHeight - 80));
      
      setPosition({ x: newX, y: newY });
    };

    const handleMouseUp = () => {
      if (isDragging) {
        setIsDragging(false);
        // 保存位置到localStorage
        localStorage.setItem('aiAssistantPosition', JSON.stringify(position));
        
        // 延迟重置wasDragged标志，确保点击事件处理正确
        setTimeout(() => {
          setWasDragged(false);
        }, 100);
      }
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, position]);

  return (
    <>
      {/* 可拖动的悬浮按钮 */}
      {!isOpen && (
        <div 
          ref={dragRef}
          className={`fixed z-50 cursor-${isDragging ? 'grabbing' : 'grab'}`}
          style={{ 
            left: `${position.x}px`, 
            top: `${position.y}px`,
            transition: isDragging ? 'none' : 'all 0.2s ease'
          }}
          onClick={handleClick}
          onMouseDown={handleMouseDown}
        >
          <div className="relative">
            <div className="absolute -top-2 -right-2 bg-red-500 text-white text-xs rounded-full w-5 h-5 flex items-center justify-center">
              新
            </div>
            <Avatar className="h-16 w-16 border-4 border-primary shadow-lg">
              <AvatarImage src="https://placehold.co/200x200/pink/white?text=AI助手" />
              <AvatarFallback className="bg-pink-500 text-white">AI</AvatarFallback>
            </Avatar>
          </div>
        </div>
      )}

      {/* 聊天窗口 */}
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="sm:max-w-md bg-[#1e1e1e] text-white p-0 overflow-hidden">
          <DialogHeader className="p-4 border-b border-gray-800">
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <Avatar className="h-8 w-8 mr-2">
                  <AvatarImage src="https://placehold.co/200x200/pink/white?text=AI助手" />
                  <AvatarFallback className="bg-pink-500 text-white">AI</AvatarFallback>
                </Avatar>
                <DialogTitle>AI助手小美</DialogTitle>
              </div>
              <div className="flex items-center space-x-2">
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-gray-400 hover:text-white"
                  onClick={() => setIsMinimized(!isMinimized)}
                >
                  {isMinimized ? <Maximize2 className="h-4 w-4" /> : <Minimize2 className="h-4 w-4" />}
                </Button>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="text-gray-400 hover:text-white"
                  onClick={() => setIsOpen(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </DialogHeader>
          
          {!isMinimized && (
            <>
              <div className="p-4 h-80 overflow-y-auto">
                {messages.map((message) => (
                  <div 
                    key={message.id} 
                    className={`flex mb-4 ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {message.sender === "assistant" && (
                      <Avatar className="h-8 w-8 mr-2 mt-1">
                        <AvatarImage src="https://placehold.co/200x200/pink/white?text=AI助手" />
                        <AvatarFallback className="bg-pink-500 text-white">AI</AvatarFallback>
                      </Avatar>
                    )}
                    <div 
                      className={`max-w-[80%] rounded-lg p-3 ${
                        message.sender === "user" 
                          ? "bg-primary text-white" 
                          : "bg-gray-800 text-white"
                      }`}
                    >
                      <p>{message.text}</p>
                      <p className="text-xs text-gray-400 mt-1">
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="p-4 border-t border-gray-800">
                <div className="flex items-center">
                  <Input
                    placeholder="输入消息..."
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
                    className="bg-gray-800 border-gray-700 text-white"
                  />
                  <Button 
                    className="ml-2 bg-primary text-white"
                    onClick={handleSendMessage}
                    disabled={!inputValue.trim()}
                  >
                    <Send className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
};

export default AIAssistant; 