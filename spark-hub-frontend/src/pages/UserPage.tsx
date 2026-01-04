
import React from "react";
import { useParams } from "react-router-dom";
import UserProfilePage from "@/pages/UserProfilePage";
import UserProfile from "@/components/UserProfile";
import { useUserByUsername } from "@/lib/api";

const UserPage: React.FC = () => {
  const { username } = useParams<{ username: string }>();
  const { data: userData, isLoading } = useUserByUsername(username || "");
  
  // Use the new UserProfilePage for our detailed profile view
  return <UserProfilePage />;
};

export default UserPage;
