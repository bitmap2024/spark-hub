# 国际化使用指南 (i18n Guide)

## 已完成的工作

### 1. 配置和基础设施
- ✅ 安装并配置了 `i18next` 和 `react-i18next`
- ✅ 创建了 i18n 配置文件 (`src/i18n/config.ts`)
- ✅ 创建了中英文语言资源文件
  - `src/i18n/locales/zh-CN.json` (中文)
  - `src/i18n/locales/en-US.json` (英文)
- ✅ 在 `main.tsx` 中导入 i18n 配置

### 2. 组件
- ✅ 创建了语言切换组件 (`src/components/LanguageSwitcher.tsx`)
- ✅ 在 Header 组件中集成了语言切换器
- ✅ Header 组件已国际化
- ✅ LeftSidebar 组件已国际化

### 3. 页面
- ✅ UserProfile 页面已完全国际化（主要页面）
- ✅ Index 页面已国际化
- ✅ NotFound 页面已国际化

## 如何使用

### 在组件中使用国际化

```typescript
import { useTranslation } from 'react-i18next';

const MyComponent = () => {
  const { t } = useTranslation();
  
  return (
    <div>
      <h1>{t('common.loading')}</h1>
      <button>{t('common.save')}</button>
    </div>
  );
};
```

### 带参数的翻译

```typescript
// 在语言文件中
{
  "knowledgeBase": {
    "morePapers": "还有 {{count}} 篇论文..."
  }
}

// 在组件中使用
{t('knowledgeBase.morePapers', { count: 5 })}
```

### 带默认值的翻译

```typescript
// 如果翻译键不存在，使用默认值
{t('common.notifications', '通知')}
```

### 切换语言

用户可以通过 Header 中的语言切换器在中英文之间切换。选择的语言会自动保存到 localStorage。

## 需要继续完成的页面

以下页面还需要添加国际化支持：

1. **知识库相关**
   - [ ] KnowledgeBaseDetail.tsx
   - [ ] KnowledgeBaseManage.tsx
   - [ ] KnowledgeBaseSettings.tsx

2. **社区相关**
   - [ ] Community.tsx
   - [ ] CreatePost.tsx

3. **消息相关**
   - [ ] Messages.tsx
   - [ ] MessageDetail.tsx

4. **其他页面**
   - [ ] Following.tsx
   - [ ] Friends.tsx
   - [ ] Trending.tsx
   - [ ] Featured.tsx
   - [ ] Spark.tsx
   - [ ] Recommend.tsx
   - [ ] SearchResults.tsx
   - [ ] PaperDetail.tsx
   - [ ] Pricing.tsx
   - [ ] LikedVideos.tsx

## 添加新翻译的步骤

1. **编辑语言文件**
   - 在 `src/i18n/locales/zh-CN.json` 添加中文翻译
   - 在 `src/i18n/locales/en-US.json` 添加对应的英文翻译

2. **在组件中使用**
   ```typescript
   import { useTranslation } from 'react-i18next';
   
   const { t } = useTranslation();
   // 使用 t('key.path') 获取翻译
   ```

3. **测试**
   - 切换语言，确保翻译正确显示
   - 检查是否有遗漏的硬编码文本

## 语言资源文件结构

```json
{
  "common": { ... },      // 通用文本
  "nav": { ... },         // 导航相关
  "user": { ... },        // 用户相关
  "knowledgeBase": { ... }, // 知识库相关
  "paper": { ... },       // 论文相关
  "message": { ... },     // 消息相关
  "auth": { ... },        // 认证相关
  "tabs": { ... },        // 标签页相关
  "toast": { ... },       // 提示消息相关
  "community": { ... },   // 社区相关
  "search": { ... },      // 搜索相关
  "pricing": { ... }      // 定价相关
}
```

## 最佳实践

1. **一致性**: 相同的文本使用相同的翻译键
2. **组织性**: 按功能模块组织翻译键
3. **默认值**: 为可选的翻译提供默认值
4. **参数化**: 使用参数而不是字符串拼接
5. **上下文**: 为翻译提供清晰的上下文

## 示例：国际化一个新页面

```typescript
// 1. 添加翻译到 zh-CN.json
{
  "myPage": {
    "title": "我的页面",
    "description": "这是描述",
    "button": "点击我"
  }
}

// 2. 添加翻译到 en-US.json
{
  "myPage": {
    "title": "My Page",
    "description": "This is a description",
    "button": "Click Me"
  }
}

// 3. 在组件中使用
import { useTranslation } from 'react-i18next';

const MyPage = () => {
  const { t } = useTranslation();
  
  return (
    <div>
      <h1>{t('myPage.title')}</h1>
      <p>{t('myPage.description')}</p>
      <button>{t('myPage.button')}</button>
    </div>
  );
};
```

## 注意事项

- 语言选择会自动保存到 localStorage
- 页面刷新后会自动加载用户之前选择的语言
- 默认语言为中文 (zh-CN)
- 所有日期格式化可能需要根据语言进行调整

## 测试

运行项目并测试：
```bash
npm run dev
```

1. 打开应用
2. 点击 Header 中的语言切换器
3. 验证所有文本是否正确切换
4. 检查页面刷新后语言是否保持

## 需要帮助？

如果在添加国际化时遇到问题：
1. 检查翻译键是否在两个语言文件中都存在
2. 确保导入了 `useTranslation` hook
3. 检查控制台是否有 i18n 相关的警告或错误

