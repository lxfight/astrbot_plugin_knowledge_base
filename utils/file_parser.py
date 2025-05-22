from typing import Optional
from astrbot.api import logger
import os
import aiofiles

# TODO 增加更多的文件类型支持


async def parse_file_content(file_path: str) -> Optional[str]:
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

        if extension not in [".txt", ".md"]:
            logger.warning(f"不支持的文件类型: {extension}，文件路径: {file_path}")
            return None

        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
        return content
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        return None
    except Exception as e:
        logger.error(f"解析文件 {file_path} 时发生错误: {e}")
        return None
