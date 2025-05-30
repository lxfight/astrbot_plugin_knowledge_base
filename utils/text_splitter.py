from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitterUtil:
    def __init__(
        self, chunk_size: int, chunk_overlap: int
    ):
        """
        初始化文本分割器。
        Args:
            chunk_size: 每个块的目标大小 (字符数或 token 数，取决于分割器实现)
            chunk_overlap: 块之间的重叠大小
        """
        # 使用 Langchain 的 RecursiveCharacterTextSplitter，它按字符分割并尝试保持段落完整性
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,  # 按字符数计算长度
            is_separator_regex=False,
        )
        # logger.info(f"文本分割器初始化：chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def split_text(self, text: str) -> List[str]:
        """
        将文本分割成块。
        Args:
            text: 待分割的文本。
        Returns:
            分割后的文本块列表。
        """
        if not text or not text.strip():
            return []
        return self.splitter.split_text(text)


# 基于 tiktoken 的分割器
# class TokenTextSplitter:
#     def __init__(self, chunk_size: int, chunk_overlap: int, model_name: str = "text-embedding-ada-002"):
#         self.chunk_size = chunk_size
#         self.chunk_overlap = chunk_overlap
#         try:
#             self.encoding = get_encoding(model_name)
#         except: # Fallback for models not directly supported by tiktoken's default list
#             self.encoding = get_encoding("cl100k_base")


#     def split_text(self, text: str) -> List[str]:
#         if not text or not text.strip():
#             return []
#         tokens = self.encoding.encode(text)
#         chunks = []
#         for i in range(0, len(tokens), self.chunk_size - self.chunk_overlap):
#             chunk_tokens = tokens[i:i + self.chunk_size]
#             chunks.append(self.encoding.decode(chunk_tokens))
#         return chunks
