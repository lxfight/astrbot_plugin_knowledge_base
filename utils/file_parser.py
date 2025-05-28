from typing import Optional
from astrbot.api import logger
from astrbot.api.star import Context
import os
import base64
import aiofiles
from aiofiles.os import stat as aio_stat
import chardet
import asyncio
from ..core.constants import (
    COMMON_ENCODINGS,
    READ_FILE_LIMIT,
    TEXT_EXTENSIONS,
    IMAGE_EXTENSIONS,
    MARKITDOWN_EXTENSIONS,
    AUDIO_EXTENSIONS,
)
from markitdown import MarkItDown
from openai import AsyncOpenAI, OpenAI


class LLM_Config:
    """LLM配置类"""
    def __init__(self, context: Context, status: bool):
        self.context = context
        self.status = status
        if self.status:
            # 获取当前使用的 provider
            provider_config = self.context.get_using_provider()
            if provider_config is None:
                logger.error("未在AstrBot配置LLM服务商，请检查配置")
                raise ValueError("未在AstrBot配置LLM服务商，请检查配置")
            self.api_key = provider_config.get_current_key()
            self.api_url = provider_config.provider_config.get("api_base")
            self.model_name = provider_config.get_model()

            # 初始化 LLM 客户端
            self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.api_url)
            self.sync_client = OpenAI(api_key=self.api_key, base_url=self.api_url)
            # 初始化 MarkItDown
            self.md_converter = MarkItDown(
                enable_plugins=True,
                llm_client=self.async_client,
                llm_model=self.model_name,
            )
            logger.info("配置LLM成功")
        else:
            self.md_converter = MarkItDown(enable_plugins=False)
            logger.warning("未启用LLM大模型解析文件，图片和复杂文档解析可能失败")

    """文本文件解析类"""
    async def _detect_and_read_file(self, file_path: str) -> str:
        """
        检测文件编码并读取文件内容
        """
        content = None
        detected_encoding = None

        # 优化：对于非常大的文件，chardet 读取整个文件可能不理想
        # 可以先读取头部一小部分来检测
        try:
            file_size = (await aio_stat(file_path)).st_size
            read_limit = min(file_size, READ_FILE_LIMIT)

            async with aiofiles.open(file_path, "rb") as f_binary:
                raw_head = await f_binary.read(read_limit)  # 读取头部

            if raw_head:
                result = chardet.detect(raw_head)
                detected_encoding = result["encoding"]
                confidence = result["confidence"]

                if detected_encoding and confidence > 0.7:
                    logger.info(
                        f"Chardet: {file_path} 编码={detected_encoding}, 置信度={confidence:.2f}"
                    )
                    try:
                        # 如果 chardet 成功，用检测到的编码完整读取文件
                        async with aiofiles.open(
                            file_path, "r", encoding=detected_encoding, errors="ignore"
                        ) as f:  # errors='ignore' 或 'replace' 可以增加容错
                            content = await f.read()
                        return content
                    except UnicodeDecodeError:
                        logger.warning(
                            f"使用 Chardet 检测到的编码 {detected_encoding} 无法完整读取 {file_path}。尝试常用编码列表。"
                        )
                        content = None  # 确保回退
                    except Exception as e_read_full:
                        logger.warning(
                            f"读取 {file_path} 时使用 Chardet 检测到的编码 {detected_encoding} 出错: {e_read_full}。尝试常用编码列表。"
                        )
                        content = None
                else:
                    logger.info(
                        f"Chardet 对 {file_path} 的检测结果不确定 (编码: {detected_encoding}, 置信度: {confidence:.2f})。尝试常用编码列表。"
                    )
            else:  # 文件为空或非常小
                logger.info(
                    f"文件 {file_path} 为空或太小，无法进行 Chardet 检测。尝试常用编码列表。"
                )

        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            raise
        except Exception as e_chardet:
            logger.warning(
                f"对 {file_path} 进行 Chardet 检测时出错: {e_chardet}。尝试常用编码列表。"
            )

        # 如果 chardet 失败或未启用，尝试常用编码
        if content is None:
            for enc in COMMON_ENCODINGS:
                try:
                    async with aiofiles.open(file_path, "r", encoding=enc) as f:
                        content = await f.read()
                    logger.info(f"成功使用编码 {enc} 读取文件 {file_path}")
                    return content
                except UnicodeDecodeError:
                    logger.debug(f"使用编码 {enc} 解码文件 {file_path} 失败")
                except FileNotFoundError:  # 应该在 chardet 步骤就被捕获，但再次检查无妨
                    logger.error(f"在尝试常用编码时文件未找到: {file_path}")
                    raise
                except Exception as e:
                    logger.error(f"使用编码 {enc} 读取文件 {file_path} 时发生错误: {e}")
                    # 考虑是否应该 break，如果不是解码错误
                    # break

        if content is None:
            logger.error(f"无法使用任何尝试过的编码解码文件 {file_path}")
            # 最后的尝试：使用 utf-8 并替换无法解码的字符
            try:
                logger.warning(
                    f"最终尝试：以 UTF-8 编码（替换错误字符）方式读取文件 {file_path}"
                )
                async with aiofiles.open(
                    file_path, "r", encoding="utf-8", errors="replace"
                ) as f:
                    content = await f.read()
                return content
            except Exception as e_final:
                logger.error(
                    f"最终尝试使用 UTF-8 编码读取文件 {file_path}（替换模式）也失败: {e_final}"
                )
                raise ValueError(f"无法读取或解码文件: {file_path}")
        return content
        
    """图片解析"""
    def image_converter(self, base64_image: str, image_format: str) -> str:
        if not self.status:
            logger.warning("未启用LLM大模型解析文件，无法解析图片")
            return None
        try:   
            response = self.sync_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "你是图片解析专家，请用当前图片语言提取图片中的文字，只返回纯净的段落文本，不要返回JSON或坐标信息。",
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_format.lstrip('.')};base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"图片解析失败 {e}")
            return None
        
    """音频解析"""
    def audio_converter(self, base64_audio: str, audio_format: str) -> str:
        if not self.status:
            logger.warning("未启用LLM大模型解析文件，无法解析音频")
            return None
        try:
            response = self.sync_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "你是音频解析专家，请用当前音频语言提取音频中的文字(中文则使用简体)，只返回纯净的段落文本，不要返回JSON或坐标信息。",
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": base64_audio,
                                    "format": audio_format.lstrip('.')
                                },
                            },
                        ],
                    }
                ],
            )
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"音频解析失败 {e}")
            return None
        
class TextFileParser:

    def __init__(self, llm_config: LLM_Config):
        self._detect_and_read_file = llm_config._detect_and_read_file
    async def parse(self, file_path: str) -> Optional[str]:
        """
        异步读取并解析文件内容。

        Args:
            file_path: 文件路径。

        Returns:
            文件文本内容，如果解析失败则返回 None。
        """
        try:
            content = await self._detect_and_read_file(file_path=file_path)
            if content is None:
                logger.error(f"无法读取文件 {file_path}，请检查文件编码")
                return None
            return content
        except Exception as e:
            logger.error(f"解析文本文件 {file_path} 时发生错误: {e}")
            return None
        
class MarkdownFileParser:
    """Markdown和复杂文本文件解析器"""
    def __init__(self, llm_config: LLM_Config):
        self.md_converter = llm_config.md_converter

    async def parse(self, file_path: str) -> Optional[str]:
        """解析Markdown文件"""
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: self.md_converter.convert(file_path)
            )
            return result.text_content
        except Exception as e:
            logger.error(f"MarkItDown 转换文件失败 {file_path}: {e}")
            return None

class ImageFileParser:
    """图片文件解析器"""
    def __init__(self, llm_config: LLM_Config):
        self.image_converter = llm_config.image_converter

    def _encode_image(self, file_path: str) -> str:
        """
        将图片转换为base64编码
        """
        try:
            with open(file_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"图片编码失败 {file_path}: {e}")
            raise


    async def parse(self, file_path: str) -> Optional[str]:
        """解析图片文件"""
        try:
            loop = asyncio.get_running_loop()
            base64_image = self._encode_image(file_path)
            image_format = os.path.splitext(file_path)[1]
            result = await loop.run_in_executor(
                None, lambda: self.image_converter(base64_image, image_format)
            )
            logger.info(f"图片转换结果：{result}")
            return result
        
        except Exception as e:
            logger.error(f"图片转换失败 {file_path}: {e}")
            return None

class AudioFileParser:
    """音频文件解析器"""
    def __init__(self, llm_config: LLM_Config):
        self.audio_converter = llm_config.audio_converter

    def _encode_audio(self, file_path: str) -> str:
        """
        将音频转换为base64编码
        """
        try:
            with open(file_path, "rb") as audio_file:
                  return base64.b64encode(audio_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"音频编码失败 {file_path}: {e}")
            raise

    async def parse(self, file_path: str) -> Optional[str]:
        """解析音频文件"""
        try:
            loop = asyncio.get_running_loop()
            base64_audio = self._encode_audio(file_path)
            audio_format = os.path.splitext(file_path)[1]
            result = await loop.run_in_executor(
                None, lambda: self.audio_converter(base64_audio, audio_format)
            )
            logger.info(f"音频转换结果：{result}")
            return result
        
        except Exception as e:
            logger.error(f"音频转换失败 {file_path}: {e}")
            return None


class FileParser:
    """文件解析器主类"""
    def __init__(self, llm_config: LLM_Config):
        self.text_parser = TextFileParser(llm_config)
        self.markdown_parser = MarkdownFileParser(llm_config)
        self.image_parser = ImageFileParser(llm_config)
        self.audio_parser = AudioFileParser(llm_config)

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

            # 根据文件类型选择对应的解析器
            if extension in TEXT_EXTENSIONS:
                return await self.text_parser.parse(file_path)
            elif extension in IMAGE_EXTENSIONS:
                return await self.image_parser.parse(file_path)
            elif extension in MARKITDOWN_EXTENSIONS:
                return await self.markdown_parser.parse(file_path)
            elif extension in AUDIO_EXTENSIONS:
                return await self.audio_parser.parse(file_path)
            else:
                logger.warning(f"不支持的文件类型: {extension}，文件路径: {file_path}")
                return None

        except FileNotFoundError:
            logger.error(f"文件未找到: {file_path}")
            return None
        except Exception as e:
            logger.error(f"解析文件 {file_path} 时发生错误: {e}")
            return None
