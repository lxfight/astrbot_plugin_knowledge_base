import os
import time
from astrbot.api.star import Context
from .vector_store.base import VectorDBBase, Document
from quart import request
from astrbot.dashboard.server import Response
from .utils.text_splitter import TextSplitterUtil
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from .utils.file_parser import FileParser, LLM_Config
from astrbot import logger
from astrbot.api import AstrBotConfig
from astrbot.core.config.default import VERSION
from .core.user_prefs_handler import UserPrefsHandler


class KnowledgeBaseWebAPI:
    def __init__(
        self,
        vec_db: VectorDBBase,
        text_splitter: TextSplitterUtil,
        astrbot_context: Context,
        llm_config: LLM_Config,
        user_prefs_handler: UserPrefsHandler = None,
        plugin_config: AstrBotConfig = None,
    ):
        self.vec_db = vec_db
        self.text_splitter = text_splitter
        self.astrbot_context = astrbot_context
        self.user_prefs_handler = user_prefs_handler
        self.plugin_config = plugin_config

        if VERSION < "3.5.13":
            raise RuntimeError(
                "AstrBot 版本过低，无法支持 FAISS 存储，请升级 AstrBot 至 3.5.13 或更高版本。"
            )

        self.astrbot_context.register_web_api(
            "/alkaid/kb/create_collection",
            self.create_collection,
            ["POST"],
            "创建一个新的知识库集合",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collections",
            self.list_collections,
            ["GET"],
            "列出所有知识库集合",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/add_file",
            self.add_documents,
            ["POST"],
            "向指定集合添加文档",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/search",
            self.search_documents,
            ["GET"],
            "搜索指定集合中的文档",
        )
        self.astrbot_context.register_web_api(
            "/alkaid/kb/collection/delete",
            self.delete_collection,
            ["GET"],
            "删除指定集合",
        )
        self.fp = FileParser(llm_config=llm_config)

    async def create_collection(self):
        """
        创建一个新的知识库集合。
        :param collection_name: 集合名称
        :return: 创建结果
        """
        data = await request.get_json()
        collection_name = data.get("collection_name")
        emoji = data.get("emoji", "🙂")
        description = data.get("description", "")
        embedding_provider_id = data.get("embedding_provider_id", None)
        if not collection_name:
            return Response().error("缺少集合名称").__dict__
        if await self.vec_db.collection_exists(collection_name):
            return Response().error("集合已存在").__dict__
        if not embedding_provider_id:
            return Response().error("缺少嵌入提供商 ID").__dict__
        try:
            # 添加集合元数据
            metadata = {
                "version": 1,  # metadata 配置版本
                "emoji": emoji,
                "description": description,
                "created_at": int(time.time()),
                "origin": "astrbot-webui",
                "embedding_provider_id": embedding_provider_id,  # AstrBot 嵌入提供商 ID
            }
            collection_metadata = (
                self.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                )
            )
            collection_metadata[collection_name] = metadata
            self.user_prefs_handler.user_collection_preferences[
                "collection_metadata"
            ] = collection_metadata
            await self.user_prefs_handler.save_user_preferences()
            # 兼容性问题，create_collection 方法放在上一步之后执行。
            await self.vec_db.create_collection(collection_name)
            return Response().ok("集合创建成功").__dict__
        except Exception as e:
            return Response().error(f"创建集合失败: {str(e)}").__dict__

    async def list_collections(self):
        """
        列出所有知识库集合。
        :return: 集合列表
        """
        try:
            collections = await self.vec_db.list_collections()
            result = []
            collections_metadata = (
                self.user_prefs_handler.user_collection_preferences.get(
                    "collection_metadata", {}
                )
            )
            for collection in collections:
                collection_md = collections_metadata.get(collection, {})
                if "embedding_provider_id" in collection_md:
                    p_id = collection_md.get("embedding_provider_id", "")
                    provider = self.astrbot_context.get_provider_by_id(p_id)
                    if provider:
                        collection_md["_embedding_provider_config"] = (
                            provider.provider_config
                        )
                count = await self.vec_db.count_documents(collection)
                result.append(
                    {"collection_name": collection, "count": count, **collection_md}
                )
            return Response().ok(data=result).__dict__
        except Exception as e:
            return Response().error(f"获取集合列表失败: {str(e)}").__dict__

    async def add_documents(self):
        """
        向指定集合添加文档。
        :param collection_name: 集合名称
        :param documents: 文档列表
        :return: 添加结果
        """
        upload_file = (await request.files).get("file")
        collection_name = (await request.form).get("collection_name")
        chunk_size = (await request.form).get("chunk_size", None)
        overlap = (await request.form).get("chunk_overlap", None)
        if not upload_file or not collection_name:
            return Response().error("缺少知识库名称").__dict__
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        try:
            chunk_size = int(chunk_size) if chunk_size else None
            overlap = int(overlap) if overlap else None
            path = os.path.join(get_astrbot_data_path(), "temp", upload_file.filename)
            await upload_file.save(path)
            content = await self.fp.parse_file_content(path)
            if not content:
                raise ValueError("文件内容为空或不支持的格式")

            chunks = self.text_splitter.split_text(
                text=content, chunk_size=chunk_size, overlap=overlap
            )
            if not chunks:
                raise Exception("chunk 内容为空")

            documents_to_add = [
                Document(
                    text_content=chunk,
                    metadata={
                        "source": upload_file.filename,
                        "user": "astrbot_webui",
                    },
                )
                for chunk in chunks
            ]

            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as e:
                logger.warning(f"删除临时文件失败: {str(e)}")

            try:
                doc_ids = await self.vec_db.add_documents(
                    collection_name, documents_to_add
                )
                if not doc_ids:
                    raise Exception("添加文档失败，返回的文档 ID 为空")
                return (
                    Response()
                    .ok(
                        data=doc_ids,
                        message=f"成功从文件 '{upload_file.filename}' 添加 {len(doc_ids)} 条知识到 '{collection_name}'。",
                    )
                    .__dict__
                )
            except Exception as e:
                raise Exception(f"添加文档失败: {str(e)}。")

        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            if os.path.exists(path):
                os.remove(path)
            return Response().error(f"添加文档失败: {str(e)}").__dict__

    async def search_documents(self):
        """
        搜索指定集合中的文档。
        :param collection_name: 集合名称
        :param query: 查询字符串
        :param top_k: 返回结果数量，默认为5
        :return: 搜索结果
        """
        # 从 URL 参数中获取查询参数
        collection_name = request.args.get("collection_name")
        query = request.args.get("query")
        try:
            top_k = int(request.args.get("top_k", 5))
        except ValueError:
            top_k = 5

        # 验证必要参数
        if not collection_name or not query:
            return Response().error("缺少集合名称或查询字符串").__dict__

        # 检查知识库是否存在
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        try:
            # 执行搜索
            results = await self.vec_db.search(collection_name, query, top_k)

            # 格式化结果以便前端展示
            formatted_results = []
            for i, doc in enumerate(results):
                doc, score = doc
                formatted_results.append(
                    {
                        "id": doc.id,
                        "content": doc.text_content,
                        "metadata": doc.metadata,
                        "score": score,
                    }
                )
            return Response().ok(data=formatted_results).__dict__
        except Exception as e:
            logger.error(f"搜索失败: {str(e)}")
            return Response().error(f"搜索失败: {str(e)}").__dict__

    async def delete_collection(self):
        """
        删除指定集合。
        :param collection_name: 集合名称
        """
        # 从 URL 参数中获取查询参数
        collection_name = request.args.get("collection_name")

        # 检查知识库是否存在
        if not await self.vec_db.collection_exists(collection_name):
            return Response().error("目标知识库不存在").__dict__

        try:
            # 执行删除
            await self.vec_db.delete_collection(collection_name)
            return Response().ok(f"删除 {collection_name} 成功").__dict__
        except Exception as e:
            logger.error(f"删除失败: {str(e)}")
            return Response().error(f"删除失败: {str(e)}").__dict__
