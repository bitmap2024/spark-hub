#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF解析工具
使用常规的pdf2txt和OCR技术解析PDF文件
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

# PDF解析相关库
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
import cv2

# 文本处理相关库
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFParser:
    """PDF解析器类，支持多种解析方法"""
    
    def __init__(self, pdf_path: str, output_dir: Optional[str] = None):
        """
        初始化PDF解析器
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录，默认为PDF所在目录
        """
        self.pdf_path = pdf_path
        self.pdf_name = os.path.basename(pdf_path)
        self.output_dir = output_dir or os.path.dirname(pdf_path)
        self.doc = fitz.open(pdf_path)
        self.num_pages = len(self.doc)
        logger.info(f"已加载PDF文件: {pdf_path}, 共 {self.num_pages} 页")
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 下载NLTK资源（如果需要）
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')
    
    def extract_text_direct(self) -> Dict[int, str]:
        """
        直接从PDF提取文本（适用于文本型PDF）
        
        Returns:
            字典，键为页码，值为提取的文本
        """
        text_by_page = {}
        for page_num in range(self.num_pages):
            page = self.doc[page_num]
            text = page.get_text()
            text_by_page[page_num + 1] = text
        
        return text_by_page
    
    def extract_text_with_layout(self) -> Dict[int, Dict[str, Any]]:
        """
        提取带布局信息的文本
        
        Returns:
            字典，键为页码，值为包含文本和布局信息的字典
        """
        layout_by_page = {}
        for page_num in range(self.num_pages):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            layout_by_page[page_num + 1] = blocks
        
        return layout_by_page
    
    def extract_images(self, page_num: int) -> List[Image.Image]:
        """
        从指定页面提取图像
        
        Args:
            page_num: 页码（从1开始）
            
        Returns:
            图像列表
        """
        page = self.doc[page_num - 1]
        image_list = []
        
        # 提取图像
        image_list = page.get_images(full=True)
        
        images = []
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = self.doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # 转换为PIL图像
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
        
        return images
    
    def perform_ocr(self, image: Image.Image) -> str:
        """
        对图像执行OCR
        
        Args:
            image: PIL图像
            
        Returns:
            OCR识别的文本
        """
        # 转换为OpenCV格式进行预处理
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # 图像预处理
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # 转回PIL格式
        pil_image = Image.fromarray(thresh)
        
        # 执行OCR
        text = pytesseract.image_to_string(pil_image, lang='chi_sim+eng')
        return text
    
    def extract_text_with_ocr(self, page_num: int) -> str:
        """
        使用OCR从指定页面提取文本
        
        Args:
            page_num: 页码（从1开始）
            
        Returns:
            OCR识别的文本
        """
        page = self.doc[page_num - 1]
        
        # 将页面渲染为图像
        pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # 执行OCR
        text = self.perform_ocr(img)
        return text
    
    def parse_pdf(self, use_ocr: bool = False, save_output: bool = True) -> Dict[str, Any]:
        """
        解析PDF文件
        
        Args:
            use_ocr: 是否使用OCR
            save_output: 是否保存输出
            
        Returns:
            解析结果
        """
        result = {
            "metadata": self.doc.metadata,
            "num_pages": self.num_pages,
            "text_by_page": {},
            "ocr_text_by_page": {} if use_ocr else None
        }
        
        # 直接提取文本
        result["text_by_page"] = self.extract_text_direct()
        
        # 如果需要OCR
        if use_ocr:
            for page_num in range(1, self.num_pages + 1):
                ocr_text = self.extract_text_with_ocr(page_num)
                result["ocr_text_by_page"][page_num] = ocr_text
        
        # 保存结果
        if save_output:
            self._save_results(result)
        
        return result
    
    def _save_results(self, result: Dict[str, Any]) -> None:
        """
        保存解析结果
        
        Args:
            result: 解析结果
        """
        # 保存直接提取的文本
        text_output_path = os.path.join(self.output_dir, f"{os.path.splitext(self.pdf_name)[0]}_text.txt")
        with open(text_output_path, 'w', encoding='utf-8') as f:
            for page_num, text in result["text_by_page"].items():
                f.write(f"=== 第 {page_num} 页 ===\n\n")
                f.write(text)
                f.write("\n\n")
        
        logger.info(f"已保存文本到: {text_output_path}")
        
        # 如果有OCR结果，也保存
        if result["ocr_text_by_page"]:
            ocr_output_path = os.path.join(self.output_dir, f"{os.path.splitext(self.pdf_name)[0]}_ocr.txt")
            with open(ocr_output_path, 'w', encoding='utf-8') as f:
                for page_num, text in result["ocr_text_by_page"].items():
                    f.write(f"=== 第 {page_num} 页 (OCR) ===\n\n")
                    f.write(text)
                    f.write("\n\n")
            
            logger.info(f"已保存OCR结果到: {ocr_output_path}")
    
    def close(self) -> None:
        """关闭PDF文档"""
        self.doc.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PDF解析工具')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('--output-dir', help='输出目录')
    parser.add_argument('--use-ocr', action='store_true', help='使用OCR')
    parser.add_argument('--no-save', action='store_true', help='不保存输出')
    
    args = parser.parse_args()
    
    try:
        pdf_parser = PDFParser(args.pdf_path, args.output_dir)
        result = pdf_parser.parse_pdf(use_ocr=args.use_ocr, save_output=not args.no_save)
        pdf_parser.close()
        
        logger.info("PDF解析完成")
        
    except Exception as e:
        logger.error(f"解析PDF时出错: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
