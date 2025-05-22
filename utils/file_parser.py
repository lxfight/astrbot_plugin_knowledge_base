from typing import Optional
from astrbot.api import logger
import os
import aiofiles
from markitdown import MarkItDown

class FileParser:
    def __init__(self):
        # 初始化 MarkItDown
        self.md_converter = MarkItDown(enable_plugins=False)
        
        self.text_extensions = {".txt", ".md"}
        
        # MarkItDown 支持的文件扩展名
        self.markitdown_extensions = {
            ".pdf", ".docx", ".doc", 
            ".pptx", ".ppt", 
            ".xlsx", ".xls",
            ".html", ".htm",
            ".json", ".xml", ".csv",
            ".epub",
            ".jpg", ".jpeg", ".png", 
            ".mp3", ".wav",  
        }

    async def parse_file_content(self, file_path: str) -> Optional[str]:
        """
        异步读取并解析文件内容。
        
        Args:
            file_path: 文件路径。
            
        Returns:
            文件文本内容，如果解析失败则返回 None。
        """
        try:
            _, extension = os.path.splitext(file_path)
            extension = extension.lower()

            # 处理普通文本文件
            if extension in self.text_extensions:
                async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                return content
                    
            # 使用 MarkItDown 处理其他支持的文件格式
            elif extension in self.markitdown_extensions:
                try:
                    result = self.md_converter.convert(file_path)
                    content = result.text_content
                    return content
                except Exception as e:
                    logger.error(f"MarkItDown 转换文件失败 {file_path}: {e}")
                    return None
            
            else:
                logger.warning(f"不支持的文件类型: {extension}，文件路径: {file_path}")
                return None

        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return None
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时发生错误: {e}")
            return None

    def get_supported_extensions(self) -> set:
        """
        获取所有支持的文件扩展名
        
        Returns:
            包含所有支持的文件扩展名的集合
        """
        return self.text_extensions.union(self.markitdown_extensions)