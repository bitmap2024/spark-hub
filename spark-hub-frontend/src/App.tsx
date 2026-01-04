import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Index from "./pages/Index";
import NotFound from "./pages/NotFound";
import UserProfile from "./pages/UserProfile";
import Following from "./pages/Following";
import Trending from "./pages/Trending";
import LikedVideos from "./pages/LikedVideos";
import KnowledgeBaseDetail from "./pages/KnowledgeBaseDetail";
import KnowledgeBaseManage from "./pages/KnowledgeBaseManage";
import KnowledgeBaseSettings from "./pages/KnowledgeBaseSettings";
import { useState } from "react";
import { Dialog, DialogContent } from "@/components/ui/dialog";
import EmailLoginForm from "./components/EmailLoginForm";
import Pricing from "./pages/Pricing";
import AIAssistant from "./components/AIAssistant";
import { useCurrentUser } from "./lib/api";
import Messages from "./pages/Messages";
import MessageDetail from "./pages/MessageDetail";
import Recommend from "./pages/Recommend";
import Featured from "./pages/Featured";
import Spark from "./pages/Spark";
import Friends from "./pages/Friends";
import Community from "./pages/Community";
import CreatePost from "./pages/CreatePost";
import SearchResults from "./pages/SearchResults";
import PaperDetail from "./pages/PaperDetail";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: false,
      refetchOnWindowFocus: false,
    },
  },
});

const App = () => {
  const [isLoginOpen, setIsLoginOpen] = useState(false);
  
  // Function to open login dialog - expose this to the header component in a real implementation
  const openLogin = () => setIsLoginOpen(true);
  
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <Dialog open={isLoginOpen} onOpenChange={setIsLoginOpen}>
          <DialogContent className="sm:max-w-md">
            <EmailLoginForm onClose={() => setIsLoginOpen(false)} />
          </DialogContent>
        </Dialog>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Index openLogin={openLogin} />} />
            <Route path="/user/me" element={<UserProfile isCurrentUser={true} />} />
            <Route path="/user/:username" element={<UserProfile />} />
            <Route path="/following" element={<Following />} />
            <Route path="/friends" element={<Friends />} />
            <Route path="/trending" element={<Trending />} />
            <Route path="/recommend" element={<Recommend />} />
            <Route path="/featured" element={<Featured />} />
            <Route path="/spark" element={<Spark />} />
            <Route path="/liked-videos" element={<LikedVideos />} />
            <Route path="/pricing" element={<Pricing />} />
            <Route path="/knowledge-base/:kbId" element={<KnowledgeBaseDetail />} />
            <Route path="/knowledge-base/:kbId/manage" element={<KnowledgeBaseManage />} />
            <Route path="/knowledge-base/:kbId/settings" element={<KnowledgeBaseSettings />} />
            <Route path="/knowledge-base/:kbId/paper/:paperId" element={<PaperDetail />} />
            <Route path="/messages" element={<Messages />} />
            <Route path="/messages/:userId" element={<MessageDetail />} />
            <Route path="/community" element={<Community />} />
            <Route path="/community/create" element={<CreatePost />} />
            <Route path="/community/:postId" element={<Community />} />
            <Route path="/search" element={<SearchResults />} />
            {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
        <AIAssistant />
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
