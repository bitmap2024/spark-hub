# PDF解析工具

这是一个使用常规PDF解析方法和OCR技术来解析PDF文件的工具。

## 功能特点

- 直接从PDF提取文本（适用于文本型PDF）
- 提取带布局信息的文本
- 从PDF提取图像
- 使用OCR技术识别PDF中的文本（适用于扫描型PDF）
- 支持中英文混合识别

## 安装

1. 安装依赖项：

```bash
pip install -r requirements.txt
```

2. 安装Tesseract OCR引擎：

- macOS:
```bash
brew install tesseract
brew install tesseract-lang  # 安装语言包，包括中文
```

- Ubuntu/Debian:
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-chi-sim  # 安装中文语言包
```

- Windows:
  从[这里](https://github.com/UB-Mannheim/tesseract/wiki)下载安装程序，并确保将Tesseract添加到系统PATH中。

## 使用方法

### 基本用法

```bash
python main.py 你的PDF文件路径
```

### 使用OCR

```bash
python main.py 你的PDF文件路径 --use-ocr
```

### 指定输出目录

```bash
python main.py 你的PDF文件路径 --output-dir 输出目录路径
```

### 不保存输出（仅返回解析结果）

```bash
python main.py 你的PDF文件路径 --no-save
```

## 输出

程序会生成以下输出文件：

- `{PDF文件名}_text.txt`：直接从PDF提取的文本
- `{PDF文件名}_ocr.txt`：使用OCR识别的文本（如果使用--use-ocr选项）

## 示例

```bash
# 基本用法
python main.py 文档.pdf

# 使用OCR
python main.py 扫描文档.pdf --use-ocr

# 指定输出目录
python main.py 文档.pdf --output-dir ./输出
```

## 注意事项

- OCR功能需要安装Tesseract OCR引擎
- OCR处理可能需要较长时间，特别是对于大型PDF文件
- 对于文本型PDF，直接提取文本通常比OCR更准确
- 对于扫描型PDF，OCR是提取文本的唯一方法 