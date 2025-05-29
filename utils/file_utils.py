# astrbot_plugin_knowledge_base/utils/file_utils.py
import os
import tempfile
import httpx
import re
from typing import Optional
from urllib.parse import urlparse

from astrbot.api import logger
from ..core.constants import ALLOWED_FILE_EXTENSIONS, MAX_DOWNLOAD_FILE_SIZE_MB


async def download_file(url: str, destination_folder: str) -> Optional[str]:
    """
    异步下载文件到指定文件夹。
    返回下载后的文件路径，如果失败则返回 None。
    """
    max_size_bytes = MAX_DOWNLOAD_FILE_SIZE_MB * 1024 * 1024
    try:
        async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
            response = await client.get(url)
            response.raise_for_status()

            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                content_disposition = response.headers.get("Content-Disposition")
                if content_disposition:
                    match = re.search(r'filename="?([^"]+)"?', content_disposition)
                    if match:
                        filename = match.group(1)
                if not filename:
                    filename = (
                        f"downloaded_file_{tempfile._RandomNameSequence().next()}"
                    )

            filename = "".join(
                c for c in filename if c.isalnum() or c in [".", "_", "-"]
            ).strip()
            if not filename:
                filename = "untitled_download"

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_size_bytes:
                logger.error(
                    f"文件下载失败：文件过大 ({int(content_length) / (1024 * 1024):.2f} MB > {MAX_DOWNLOAD_FILE_SIZE_MB} MB)。URL: {url}"
                )
                return None

            _, extension = os.path.splitext(filename)
            if extension.lower() not in ALLOWED_FILE_EXTENSIONS:
                logger.error(
                    f"文件下载失败：不支持的文件类型 '{extension}'. URL: {url}"
                )
                return None

            temp_file_path = os.path.join(destination_folder, filename)

            with open(temp_file_path, "wb") as f:
                downloaded_size = 0
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if downloaded_size > max_size_bytes:
                        f.close()
                        os.remove(temp_file_path)
                        logger.error(
                            f"文件下载失败：文件在下载过程中超出大小限制。URL: {url}"
                        )
                        return None

            logger.info(f"文件已成功下载到: {temp_file_path} 从 URL: {url}")
            return temp_file_path
    except httpx.HTTPStatusError as e:
        logger.error(
            f"文件下载 HTTP 错误: {e.response.status_code} - {e.response.text}. URL: {url}"
        )
        return None
    except Exception as e:
        logger.error(f"文件下载失败: {e}. URL: {url}", exc_info=True)
        return None
