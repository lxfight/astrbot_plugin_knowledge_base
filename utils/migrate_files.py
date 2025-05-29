import os
import pickle
from astrbot.api import logger
from astrbot.core.config.default import VERSION


def migrate_docs_to_db(data_path: str):
    """
    迁移函数，将原有的 .docs 文件转换为 .db 文件。

    参数：
    data_path: 存储 Faiss 集合文件的目录路径

    流程：
    1. 遍历 data_path 目录下以 .docs 结尾的文件。
    2. 对于每个 .docs 文件，尝试加载原始数据。
    3. 将数据写入新的 .db 文件，文件名与原 .docs 文件相同，只是后缀由 .docs 改为 .db。
    4. 可选：迁移完成后删除原始的 .docs 文件。
    """
    # data_path 后添加/faiss_data路径
    faiss_data_path = os.path.join(data_path, "faiss_data")
    files = os.listdir(faiss_data_path)
    docs_files = [f for f in files if f.endswith(".docs")]

    if not docs_files:
        logger.info("未找到任何需要迁移的 .docs 文件。")
        return

    for docs_file in docs_files:
        docs_filepath = os.path.join(faiss_data_path, docs_file)
        # 构造新的 db 文件路径，将后缀 .docs 替换为 .db
        db_filename = docs_file[: -len(".docs")] + ".db"
        db_filepath = os.path.join(faiss_data_path, db_filename)
        try:
            with open(docs_filepath, "rb") as f:
                data = pickle.load(f)
            with open(db_filepath, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"成功迁移 {docs_file} 到 {db_filename}")

            # 如果需要，可取消下面代码的注释以删除原来的 .docs 文件
            # os.remove(docs_filepath)
            # logger.info(f"已删除原始文件 {docs_file}")
        except Exception as e:
            logger.error(f"迁移文件 {docs_file} 失败: {e}")


# def migrate_to_astrbot_faiss_store(data_path: str):
#     """将 FAISS 存储迁移到 AstrBot 维护的 FAISS 存储格式。

#     依赖于 migrate_docs_to_db。为了保证最好的迁移兼容性，需要在 migrate_docs_to_db 之后执行。
#     """
#     if VERSION < "3.5.12":
#         # 仅在 AstrBot 版本 >= 3.5.12 时执行迁移
#         return

#     faiss_data_path = os.path.join(data_path, "faiss_data")
#     files = os.listdir(faiss_data_path)
#     candidate_files = [f for f in files if f.endswith(".db")]

#     # 识别是否是 pickle 文件。检查魔术头
#     def filter_pickle_files(files):
#         ret = []
#         for file in files:
#             try:
#                 with open(os.path.join(faiss_data_path, file), "rb") as f:
#                     magic = f.read(4)
#                     if magic == b"\x80\x04":
#                         ret.append(file)
#             except Exception:
#                 pass
#         return ret

#     pickle_files = filter_pickle_files(candidate_files)

#     if not pickle_files:
#         logger.debug("未找到任何需要迁移的 .db 的 Pickle 格式文件。")
#         return

#     import faiss

#     index_files = [f for f in files if f.endswith(".index")]
#     indexes = {}
#     for index_file in index_files:
#         index_path = os.path.join(faiss_data_path, index_file)
#         try:
#             index = faiss.read_index(index_path)
#             collection_name = index_file[: -len(".index")]
#             indexes[collection_name] = index
#         except Exception as e:
#             logger.error(f"读取索引文件 {index_file} 失败: {e}")
#     docs = {}
#     for pickle_file in pickle_files:
#         try:
#             with open(os.path.join(faiss_data_path, pickle_file), "rb") as f:
#                 data = pickle.load(f)
#             collection_name = pickle_file[: -len(".db")]
#             docs[collection_name] = data
#         except Exception as e:
#             logger.error(f"读取文档文件 {pickle_file} 失败: {e}")

#     for collection_name in docs.keys():
#         from ..vector_store.astrbot_faiss_store import FaissStore

#         old_faiss_index = indexes.get(collection_name)
#         dimention = old_faiss_index.d
#         embedding_util = None
#         new_faiss_store = FaissStore(None, dimention, faiss_data_path)
