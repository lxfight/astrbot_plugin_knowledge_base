# astrbot_plugin_knowledge_base/command/general_commands.py
from typing import TYPE_CHECKING, AsyncGenerator
from astrbot.api.event import AstrMessageEvent

if TYPE_CHECKING:
    from ..main import KnowledgeBasePlugin

async def handle_kb_help(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent
) -> AsyncGenerator[AstrMessageEvent, None]:
    help_text = """
知识库插件帮助：
/kb add text <内容> [知识库名] - 添加文本到知识库
/kb add file <文件路径或者下载链接> [知识库名]
/kb search <查询内容> [知识库名] [数量] - 搜索知识库
/kb list - 列出所有知识库
/kb current - 查看当前会话默认知识库
/kb use <知识库名> - 设置当前会话默认知识库
/kb create <知识库名> - 创建一个新的知识库
/kb delete <知识库名> - 删除一个知识库及其内容 (危险操作!)
/kb count [知识库名] - 查看知识库中文档数量
/kb help - 显示此帮助信息
""".strip()
    yield event.plain_result(help_text)


async def handle_kb_current_collection(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent
) -> AsyncGenerator[AstrMessageEvent, None]:
    current_col = plugin.user_prefs_handler.get_user_default_collection(event)
    yield event.plain_result(f"当前会话默认知识库为: {current_col}")


async def handle_kb_use_collection(
    plugin: "KnowledgeBasePlugin", event: AstrMessageEvent, collection_name: str
) -> AsyncGenerator[AstrMessageEvent, None]:
    if not collection_name:
        yield event.plain_result("请输入要设置的知识库名称。用法: /kb use <知识库名>")
        return

    async for msg_result in plugin.user_prefs_handler.set_user_default_collection(
        event, collection_name
    ):
        yield msg_result
