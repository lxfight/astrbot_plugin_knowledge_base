# astrbot_plugin_knowledge_base/llm_enhancer.py
from typing import TYPE_CHECKING

from astrbot.api import logger, AstrBotConfig
from astrbot.api.event import AstrMessageEvent
from astrbot.api.provider import ProviderRequest
from .constants import KB_START_MARKER, KB_END_MARKER, USER_PROMPT_DELIMITER_IN_HISTORY

if TYPE_CHECKING:
    from ..vector_store.base import VectorDBBase
    from .user_prefs_handler import UserPrefsHandler


def clean_contexts_from_kb_content(req: ProviderRequest):
    """
    自动删除 req.contexts 里面由知识库补充的历史对话内容。
    """
    if not req.contexts:
        return

    cleaned_contexts = []
    initial_context_count = len(req.contexts)

    for message in req.contexts:
        role = message.get("role")
        content = message.get("content", "")

        if role == "system" and KB_START_MARKER in content:
            logger.debug(
                f"从历史对话中检测到并删除知识库 system 消息: {content[:100]}..."
            )
            continue
        elif role == "user" and KB_START_MARKER in content:
            start_marker_idx = content.find(KB_START_MARKER)
            end_marker_idx = content.find(KB_END_MARKER, start_marker_idx)
            if start_marker_idx != -1 and end_marker_idx != -1:
                original_prompt_delimiter_idx = content.find(
                    USER_PROMPT_DELIMITER_IN_HISTORY,
                    end_marker_idx + len(KB_END_MARKER),
                )
                if original_prompt_delimiter_idx != -1:
                    original_user_prompt = content[
                        original_prompt_delimiter_idx
                        + len(USER_PROMPT_DELIMITER_IN_HISTORY) :
                    ].strip()
                    message["content"] = original_user_prompt
                    cleaned_contexts.append(message)
                    logger.debug(
                        f"从历史对话 user 消息中清理知识库内容，保留原用户问题: {original_user_prompt[:100]}..."
                    )
                else:
                    logger.warning(
                        f"用户消息中检测到知识库标记但缺少原始用户问题分隔符，删除该消息: {content[:100]}..."
                    )
                    continue
            else:
                logger.warning(
                    f"用户消息中检测到知识库起始标记但缺少结束标记，删除该消息: {content[:100]}..."
                )
                continue
        else:
            cleaned_contexts.append(message)

    req.contexts = cleaned_contexts
    if len(req.contexts) < initial_context_count:
        logger.info(
            f"成功从历史对话中删除了 {initial_context_count - len(req.contexts)} 条知识库补充消息。"
        )


async def enhance_request_with_kb(
    event: AstrMessageEvent,
    req: ProviderRequest,
    vector_db: "VectorDBBase",
    user_prefs_handler: "UserPrefsHandler",
    plugin_config: AstrBotConfig,
):
    default_collection_name = user_prefs_handler.get_user_default_collection(event)

    if not default_collection_name:
        logger.debug("未找到当前会话的默认知识库，跳过知识库查询。")
        return

    if not await vector_db.collection_exists(default_collection_name):
        logger.warning(
            f"知识库 '{default_collection_name}' 不存在，跳过知识库查询。"
        )
        return

    # 获取LLM RAG配置
    llm_rag_config = plugin_config.get("llm_rag_settings", {})
    kb_search_top_k = plugin_config.get("search_top_k", 3)  # 复用现有的搜索配置
    kb_insertion_method = llm_rag_config.get("insertion_method", "prepend_prompt")
    kb_context_template = llm_rag_config.get(
        "context_template",
        "这是相关的知识库信息，请参考这些信息来回答用户的问题：\n{retrieved_contexts}",
    )
    min_similarity_score = llm_rag_config.get("min_similarity_score", 0.5)

    user_query = req.prompt
    if not user_query or not user_query.strip():
        logger.debug("用户查询为空，跳过知识库搜索。")
        return

    try:
        logger.info(
            f"为LLM请求在知识库 '{default_collection_name}' 中搜索: '{user_query[:50]}...' (top_k={kb_search_top_k})"
        )
        search_results = await vector_db.search(
            default_collection_name, user_query, top_k=kb_search_top_k
        )
    except Exception as e:
        logger.error(
            f"LLM 请求时从知识库 '{default_collection_name}' 搜索失败: {e}",
            exc_info=True,
        )
        return

    if not search_results:
        logger.info(
            f"在知识库 '{default_collection_name}' 中未找到与查询 '{user_query[:50]}...' 相关的内容。"
        )
        return

    retrieved_contexts_list = []
    for doc, score in search_results:
        if score >= min_similarity_score:
            source_info = doc.metadata.get("source", "未知来源")
            context_item = (
                f"- 内容: {doc.text_content} (来源: {source_info}, 相关度: {score:.2f})"
            )
            retrieved_contexts_list.append(context_item)
        else:
            logger.debug(
                f"文档 '{doc.text_content[:30]}...' 相关度 {score:.2f} 低于阈值 {min_similarity_score}，已忽略。"
            )

    if not retrieved_contexts_list:
        logger.info(
            f"所有检索到的知识库内容相关度均低于阈值 {min_similarity_score}，不进行增强。"
        )
        return

    formatted_contexts = "\n".join(retrieved_contexts_list)
    knowledge_to_insert = kb_context_template.format(
        retrieved_contexts=formatted_contexts
    )

    max_kb_insert_length = llm_rag_config.get("max_insert_length", 200000)
    if len(knowledge_to_insert) > max_kb_insert_length:
        logger.warning(
            f"知识库插入内容过长 ({len(knowledge_to_insert)} chars)，将被截断至 {max_kb_insert_length} chars。"
        )
        knowledge_to_insert = (
            knowledge_to_insert[:max_kb_insert_length] + "\n... [内容已截断]"
        )

    knowledge_to_insert = f"{KB_START_MARKER}\n{knowledge_to_insert}\n{KB_END_MARKER}"

    if kb_insertion_method == "system_prompt":
        if req.system_prompt:
            req.system_prompt = f"{knowledge_to_insert}\n\n{req.system_prompt}"
        else:
            req.system_prompt = knowledge_to_insert
        logger.info(
            f"知识库内容已添加到 system_prompt。长度: {len(knowledge_to_insert)}"
        )
    elif kb_insertion_method == "prepend_prompt":
        req.prompt = (
            f"{knowledge_to_insert}\n\n{USER_PROMPT_DELIMITER_IN_HISTORY}{req.prompt}"
        )
        logger.info(f"知识库内容已前置到用户 prompt。长度: {len(knowledge_to_insert)}")
    else:
        logger.warning(
            f"未知的知识库内容插入方式: {kb_insertion_method}，将默认前置到用户 prompt。"
        )
        req.prompt = (
            f"{knowledge_to_insert}\n\n{USER_PROMPT_DELIMITER_IN_HISTORY}{req.prompt}"
        )

    logger.debug(f"修改后的 ProviderRequest.prompt: {req.prompt[:200]}...")
    if req.system_prompt:
        logger.debug(
            f"修改后的 ProviderRequest.system_prompt: {req.system_prompt[:200]}..."
        )
