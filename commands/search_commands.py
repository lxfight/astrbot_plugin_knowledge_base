# astrbot_plugin_knowledge_base/command_handlers/search_commands.py
from typing import Optional, TYPE_CHECKING, AsyncGenerator
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import KnowledgeBasePlugin


async def handle_search(
    plugin: "KnowledgeBasePlugin",
    event: AstrMessageEvent,
    query: str,
    collection_name: Optional[str] = None,
    top_k_str: Optional[str] = None,
) -> AsyncGenerator[AstrMessageEvent, None]:
    if not query.strip():
        yield event.plain_result("查询内容不能为空。")
        return

    target_collection = (
        collection_name
        if collection_name
        else plugin.user_prefs_handler.get_user_default_collection(event)
    )

    if not await plugin.vector_db.collection_exists(target_collection):
        yield event.plain_result(f"知识库 '{target_collection}' 不存在。")
        return

    top_k = 1
    if top_k_str is not None:
        if isinstance(top_k_str, int):
            top_k = top_k_str
        elif isinstance(top_k_str, str) and top_k_str.isdigit():
            try:
                top_k = int(top_k_str)
            except ValueError:
                logger.warning(
                    f"无法将 top_k 参数 '{top_k_str}' 转换为整数，将使用默认值 {top_k}。"
                )
        else:
            logger.warning(
                f"top_k 参数 '{top_k_str}' (类型: {type(top_k_str)}) 无效，将使用默认值 {top_k}。"
            )

    top_k = max(1, min(top_k, 30))  # Limit top_k
    logger.info(
        f"搜索知识库 '{target_collection}'，查询: '{query[:30]}...', top_k: {top_k}"
    )

    try:
        yield event.plain_result(
            f"正在知识库 '{target_collection}' 中搜索 '{query[:30]}...' (最多{top_k}条)..."
        )
        search_results = await plugin.vector_db.search(
            target_collection, query, top_k=top_k
        )

        if not search_results:
            yield event.plain_result(
                f"在知识库 '{target_collection}' 中没有找到与 '{query[:30]}...' 相关的内容。"
            )
            return

        response_message = f"知识库 '{target_collection}' 中关于 '{query[:30]}...' 的搜索结果 (相关度从高到低):\n"
        for i, (doc, score) in enumerate(search_results):
            source_info = (
                f" (来源: {doc.metadata.get('source', '未知')})"
                if doc.metadata.get("source")
                else ""
            )
            response_message += f"\n{i + 1}. [相关度: {score:.2f}]{source_info}\n"
            content_preview = (
                doc.text_content[:200] + "..."
                if len(doc.text_content) > 200
                else doc.text_content
            )
            response_message += f"   内容: {content_preview}\n"

        if len(response_message) > 1500:  # Threshold for text_to_image
            yield event.plain_result("搜索结果较长，将尝试转为图片发送。")
            # Assuming self.text_to_image is a method of Star (plugin instance)
            img_url = await plugin.text_to_image(response_message)
            yield event.image_result(img_url)
        else:
            yield event.plain_result(response_message)

    except Exception as e:
        logger.error(f"搜索知识库 '{target_collection}' 失败: {e}", exc_info=True)
        yield event.plain_result(f"搜索失败: {e}")
