# astrbot_plugin_knowledge_base/command_handlers/add_commands.py
import os
from urllib.parse import urlparse
from typing import Optional, TYPE_CHECKING, AsyncGenerator

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from ..vector_store.base import Document
from ..utils import file_utils
from ..core.constants import ALLOWED_FILE_EXTENSIONS

if TYPE_CHECKING:
    from ..main import KnowledgeBasePlugin


async def handle_add_text(
    plugin: "KnowledgeBasePlugin",
    event: AstrMessageEvent,
    content: str,
    collection_name: Optional[str] = None,
) -> AsyncGenerator[AstrMessageEvent, None]:
    if not content.strip():
        yield event.plain_result("添加的内容不能为空。")
        return

    target_collection = (
        collection_name
        if collection_name
        else plugin.user_prefs_handler.get_user_default_collection(event)
    )

    if plugin.config.get(
        "auto_create_collection", True
    ) and not await plugin.vector_db.collection_exists(target_collection):
        try:
            await plugin.vector_db.create_collection(target_collection)
            logger.info(f"知识库 '{target_collection}' 不存在，已自动创建。")
            yield event.plain_result(
                f"知识库 '{target_collection}' 不存在，已自动创建。"
            )
        except Exception as e:
            logger.error(f"自动创建知识库 '{target_collection}' 失败: {e}")
            yield event.plain_result(f"自动创建知识库 '{target_collection}' 失败: {e}")
            return

    chunks = plugin.text_splitter.split_text(content)
    if not chunks:
        yield event.plain_result("文本分割后无有效内容。")
        return

    documents_to_add = [
        Document(
            text_content=chunk,
            metadata={"source": "direct_text", "user": event.get_sender_name()},
        )
        for chunk in chunks
    ]

    try:
        yield event.plain_result(
            f"正在处理 {len(chunks)} 个文本块并添加到知识库 '{target_collection}'..."
        )
        doc_ids = await plugin.vector_db.add_documents(
            target_collection, documents_to_add
        )
        if doc_ids:
            yield event.plain_result(
                f"成功添加 {len(doc_ids)} 条知识到 '{target_collection}'。"
            )
        else:
            yield event.plain_result(
                f"未能添加任何知识到 '{target_collection}'，请检查日志。"
            )
    except Exception as e:
        logger.error(f"添加文本到知识库 '{target_collection}' 失败: {e}", exc_info=True)
        yield event.plain_result(f"添加知识失败: {e}")


async def handle_add_file(
    plugin: "KnowledgeBasePlugin",
    event: AstrMessageEvent,
    path_or_url: str,
    collection_name: Optional[str] = None,
) -> AsyncGenerator[AstrMessageEvent, None]:
    if not path_or_url:
        yield event.plain_result("请输入文件/文件夹路径或 URL。")
        return

    target_collection = (
        collection_name
        if collection_name
        else plugin.user_prefs_handler.get_user_default_collection(event)
    )

    if plugin.config.get(
        "auto_create_collection", True
    ) and not await plugin.vector_db.collection_exists(target_collection):
        try:
            await plugin.vector_db.create_collection(target_collection)
            logger.info(f"知识库 '{target_collection}' 不存在，已自动创建。")
            yield event.plain_result(
                f"知识库 '{target_collection}' 不存在，已自动创建。"
            )
        except Exception as e_create:
            logger.error(f"自动创建知识库 '{target_collection}' 失败: {e_create}")
            yield event.plain_result(
                f"自动创建知识库 '{target_collection}' 失败: {e_create}"
            )
            return

    files_to_process_info = []
    is_temp_dir_created = False
    temp_download_dir_for_cleanup = None

    try:
        parsed_uri = urlparse(path_or_url)
        is_url = all([parsed_uri.scheme, parsed_uri.netloc]) and parsed_uri.scheme in [
            "http",
            "https",
        ]
    except ValueError:
        is_url = False

    if is_url:
        yield event.plain_result(f"检测到 URL，正在尝试下载: {path_or_url} ...")
        temp_download_dir = os.path.join(
            plugin.persistent_data_root_path, "temp_downloads"
        )
        os.makedirs(temp_download_dir, exist_ok=True)
        is_temp_dir_created = True
        temp_download_dir_for_cleanup = temp_download_dir

        downloaded_path = await file_utils.download_file(path_or_url, temp_download_dir)
        if downloaded_path:
            files_to_process_info.append(
                (downloaded_path, os.path.basename(downloaded_path), True)
            )
        else:
            yield event.plain_result(f"无法下载文件: {path_or_url}")
            return
    else:
        logger.info(f"用户提供了本地路径: {path_or_url}。将检查是文件还是文件夹。")
        if not os.path.exists(path_or_url):
            yield event.plain_result(f"本地路径无效或不存在: {path_or_url}")
            return

        if os.path.isfile(path_or_url):
            files_to_process_info.append(
                (path_or_url, os.path.basename(path_or_url), False)
            )
        elif os.path.isdir(path_or_url):
            yield event.plain_result(
                f"检测到文件夹路径，正在遍历支持的文件: {path_or_url} ..."
            )
            supported_extensions = tuple(ALLOWED_FILE_EXTENSIONS)
            found_files_count = 0
            for root, _, files in os.walk(path_or_url):
                for filename in files:
                    if filename.lower().endswith(supported_extensions):
                        full_path = os.path.join(root, filename)
                        files_to_process_info.append((full_path, filename, False))
                        found_files_count += 1
            if not files_to_process_info:
                yield event.plain_result(
                    f"在文件夹 '{path_or_url}' 中未找到支持的文件类型 ({', '.join(supported_extensions)})。"
                )
                return
            yield event.plain_result(
                f"在文件夹中找到 {found_files_count} 个支持的文件，将开始处理。"
            )
        else:
            yield event.plain_result(f"路径 '{path_or_url}' 不是有效的文件或文件夹。")
            return

    if not files_to_process_info:
        yield event.plain_result("未能获取到任何要处理的文件。")
        return

    total_docs_added = 0
    total_chunks_processed = 0
    files_processed_count = 0
    error_files = []

    for file_path, original_filename, is_temp_file in files_to_process_info:
        files_processed_count += 1
        logger.debug(
            f"正在处理第 {files_processed_count}/{len(files_to_process_info)} 个文件: '{original_filename}'..."
        )

        content = await plugin.file_parser.parse_file_content(file_path)
        if content is None:
            message = f"无法解析文件 '{original_filename}' 或文件为空，已跳过。"
            yield event.plain_result(message)
            error_files.append(original_filename)
            if is_temp_file:
                try:
                    os.remove(file_path)
                    logger.info(f"已删除临时文件: {file_path}")
                except OSError as e:
                    logger.error(f"删除临时文件 {file_path} 失败: {e}")
            continue

        chunks = plugin.text_splitter.split_text(content)
        if not chunks:
            message = f"文件 '{original_filename}' 分割后无有效内容，已跳过。"
            yield event.plain_result(message)
            error_files.append(original_filename)
            if is_temp_file:
                try:
                    os.remove(file_path)
                    logger.info(f"已删除临时文件: {file_path}")
                except OSError as e:
                    logger.error(f"删除临时文件 {file_path} 失败: {e}")
            continue

        total_chunks_processed += len(chunks)
        documents_to_add = [
            Document(
                text_content=chunk,
                metadata={"source": original_filename, "user": event.get_sender_name()},
            )
            for chunk in chunks
        ]
        # yield event.plain_result(f"开始添加文件：{original_filename}")
        logger.info(f"开始添加文件：{original_filename}")
        try:
            doc_ids = await plugin.vector_db.add_documents(
                target_collection, documents_to_add
            )
            if doc_ids:
                total_docs_added += len(doc_ids)
        except Exception as e_add:
            logger.error(
                f"从文件 '{original_filename}' 添加知识到知识库 '{target_collection}' 失败: {e_add}",
                exc_info=True,
            )
            yield event.plain_result(f"处理文件 '{original_filename}' 时出错: {e_add}")
            error_files.append(original_filename)
        finally:
            if is_temp_file:
                try:
                    os.remove(file_path)
                    logger.info(f"已删除临时下载文件: {file_path}")
                except OSError as e_rm:
                    logger.error(f"删除临时文件失败 {file_path}: {e_rm}")

    summary_message = f"文件处理完成。\n总计处理文件数: {len(files_to_process_info)}\n"
    summary_message += f"成功添加知识条目数: {total_docs_added} (来自 {total_chunks_processed} 个文本块)\n"
    if error_files:
        summary_message += f"处理失败或跳过的文件 ({len(error_files)} 个): {', '.join(error_files[:5])}"
        if len(error_files) > 5:
            summary_message += "..."
    else:
        summary_message += "所有文件均成功处理完毕！"
    yield event.plain_result(summary_message)

    if is_temp_dir_created and temp_download_dir_for_cleanup:
        try:
            if not os.listdir(temp_download_dir_for_cleanup):
                os.rmdir(temp_download_dir_for_cleanup)
                logger.info(f"已删除空的临时下载目录: {temp_download_dir_for_cleanup}")
            else:
                logger.info(
                    f"临时下载目录 {temp_download_dir_for_cleanup} 非空，未删除。"
                )
        except OSError as e_rmdir:
            logger.error(
                f"删除临时下载目录 {temp_download_dir_for_cleanup} 失败: {e_rmdir}"
            )
