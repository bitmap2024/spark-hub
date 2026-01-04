import asyncio
import aiofiles
import aiohttp
from typing import Optional, Dict, Union, BinaryIO
from pathlib import Path
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

class AsyncDownloader:
    """异步文件下载器"""
    
    def __init__(
        self,
        concurrency: int = 5,
        chunk_size: int = 1024 * 1024,  # 1MB
        timeout: int = 300,
        save_dir: Union[str, Path] = "./downloads"
    ):
        self.concurrency = concurrency
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.semaphore = asyncio.Semaphore(concurrency)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def download_file(
        self,
        url: str,
        filename: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Path:
        """
        下载文件到指定位置
        
        Args:
            url: 文件URL
            filename: 保存的文件名（可选）
            headers: 自定义请求头（可选）
            
        Returns:
            保存文件的路径
        """
        async with self.semaphore:
            if not filename:
                filename = url.split("/")[-1]
            
            file_path = self.save_dir / filename
            
            try:
                async with self.session.get(
                    url,
                    headers=headers,
                    timeout=self.timeout
                ) as response:
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded_size = 0
                    
                    async with aiofiles.open(file_path, "wb") as f:
                        async for chunk in response.content.iter_chunked(self.chunk_size):
                            await f.write(chunk)
                            downloaded_size += len(chunk)
                            
                            if total_size:
                                progress = (downloaded_size / total_size) * 100
                                logger.info(f"Downloading {filename}: {progress:.2f}%")
                    
                    logger.success(f"Successfully downloaded {filename}")
                    return file_path
                    
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
                if file_path.exists():
                    file_path.unlink()
                raise
            
    async def download_files(
        self,
        urls: list[str],
        filenames: Optional[list[str]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> list[Path]:
        """批量下载文件"""
        if filenames and len(urls) != len(filenames):
            raise ValueError("URLs and filenames must have the same length")
            
        tasks = []
        for i, url in enumerate(urls):
            filename = filenames[i] if filenames else None
            tasks.append(self.download_file(url, filename, headers))
            
        return await asyncio.gather(*tasks, return_exceptions=True)
        
    async def download_with_progress(
        self,
        url: str,
        filename: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Path:
        """带进度回调的文件下载"""
        if not self.session:
            raise RuntimeError("Downloader session not initialized")
            
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                response.raise_for_status()
                
                if not filename:
                    filename = url.split('/')[-1]
                    
                file_path = self.save_dir / filename
                total_size = int(response.headers.get('content-length', 0))
                
                async with aiofiles.open(file_path, 'wb') as f:
                    downloaded_size = 0
                    async for chunk in response.content.iter_chunked(self.chunk_size):
                        await f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if progress_callback and total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            await progress_callback(filename, progress)
                            
                return file_path
                
        except Exception as e:
            logger.error(f"Error downloading {url}: {str(e)}")
            raise 