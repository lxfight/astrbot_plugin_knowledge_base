# astrbot_plugin_knowledge_base/constants.py

PLUGIN_REGISTER_NAME = "astrbot_plugin_knowledge_base"

# 定义知识库内容标记
KB_START_MARKER = "###KBDATA_START###"
KB_END_MARKER = "###KBDATA_END###"

# 用于 'prepend_prompt' 方式时，在用户原始问题前添加的标记
USER_PROMPT_DELIMITER_IN_HISTORY = "\n\n用户的原始问题是：\n"

# 文件下载相关常量
ALLOWED_FILE_EXTENSIONS = [
    ".txt",
    ".md",
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".csv",
    ".epub",
    ".jpg",
    ".jpeg",
    ".png",
]
TEXT_EXTENSIONS = {".txt", ".md"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".acc", ".aiff"}
MARKITDOWN_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".doc",
    ".pptx",
    ".ppt",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".json",
    ".xml",
    ".csv",
    ".epub",
}
MAX_DOWNLOAD_FILE_SIZE_MB = 50
COMMON_ENCODINGS = [
    "utf-8",
    "gbk",
    "gb2312",
    "gb18030",
    "utf-16",
    "latin-1",
    "iso-8859-1",
]
READ_FILE_LIMIT = 4096  # 4KB
