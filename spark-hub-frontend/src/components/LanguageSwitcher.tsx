import React from 'react';
import { useTranslation } from 'react-i18next';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Globe } from "lucide-react";

const LanguageSwitcher: React.FC = () => {
  const { i18n } = useTranslation();

  const changeLanguage = (lng: string) => {
    i18n.changeLanguage(lng);
    localStorage.setItem('language', lng);
  };

  return (
    <div className="flex items-center gap-2">
      <Globe className="h-4 w-4 text-gray-400" />
      <Select value={i18n.language} onValueChange={changeLanguage}>
        <SelectTrigger className="w-[120px] bg-[#1E1E1E] border-gray-700">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="zh-CN">中文</SelectItem>
          <SelectItem value="en-US">English</SelectItem>
        </SelectContent>
      </Select>
    </div>
  );
};

export default LanguageSwitcher;

