import subprocess
import importlib
from typing import Literal
from astrbot.api import logger
# --- UV 工具查找 ---


def _find_uv_executable():
    """
    尝试查找 'uv' 可执行文件。
    在系统的 PATH 环境变量中查找。
    返回 'uv' 的路径，如果找不到则返回 None。
    """
    import shutil

    uv_exe = shutil.which("uv")
    return uv_exe


# 在模块加载时一次性查找 UV 可执行文件，避免重复查找
_UV_EXECUTABLE = _find_uv_executable()

# --- 包安装与检查核心逻辑 (使用 UV) ---


def _install_package_uv(package_name_with_spec: str, friendly_name: str = None) -> bool:
    """
    尝试使用 'uv pip install' 安装一个 Python 包。
    如果安装成功或包已存在，则返回 True；否则返回 False。
    """
    if not _UV_EXECUTABLE:
        logger.error(
            f"错误: 无法安装 {friendly_name or package_name_with_spec}，因为未找到 UV 可执行文件。请安装 UV。"
        )
        return False

    if friendly_name is None:
        friendly_name = (
            package_name_with_spec.split(">=")[0]
            .split("==")[0]
            .split("<=")[0]
            .split("[")[0]
            .strip()
        )

    logger.info(f"信息: [UV] 尝试安装 {friendly_name} ({package_name_with_spec})...")
    try:
        command = [_UV_EXECUTABLE, "pip", "install", package_name_with_spec]
        process = subprocess.run(
            command, check=False, capture_output=True, text=True, encoding="utf-8"
        )

        if process.returncode == 0:
            logger.info(f"信息: [UV] 成功安装/验证 {friendly_name}。")
            importlib.invalidate_caches()  # 使 importlib 缓存失效
            return True
        else:
            logger.error(f"错误: [UV] 安装 {friendly_name} 失败。")
            logger.error(f"       UV 标准输出:\n{process.stdout}")
            logger.error(f"       UV 标准错误:\n{process.stderr}")
            return False
    except FileNotFoundError:
        logger.error(
            f"错误: [UV] 命令 '{_UV_EXECUTABLE}' 未找到。请确保 UV 已安装并位于 PATH 中。"
        )
        return False
    except Exception as e:
        logger.error(f"错误: [UV] 安装 {friendly_name} 时发生未知错误: {e}")
        return False


def _check_and_install_package_uv(
    package_name: str,
    version_spec: str = "",
    friendly_name: str = None,
    auto_install: bool = True,
) -> bool:
    """
    检查一个包是否可导入。如果不可导入且 auto_install 为 True，则尝试使用 UV 安装它。
    返回 True 如果包可用 (已存在或成功安装)，否则返回 False。
    """
    module_name = (
        package_name.split("[")[0].split("==")[0].split(">=")[0].split("<")[0].strip()
    )
    if friendly_name is None:
        friendly_name = module_name

    try:
        importlib.import_module(module_name)
        logger.info(f"信息: {friendly_name} 已存在。")
        return True
    except ImportError:
        logger.info(f"信息: 未找到 {friendly_name}。")
        if auto_install:
            full_package_spec = f"{package_name}{version_spec}"
            if _install_package_uv(full_package_spec, friendly_name):
                try:
                    importlib.invalidate_caches()
                    importlib.import_module(module_name)
                    logger.info(f"信息: {friendly_name} 通过 UV 成功安装，现在可用。")
                    return True
                except ImportError:
                    logger.error(f"错误: {friendly_name} 报告已通过 UV 安装，但仍无法导入。")
                    return False
            else:
                return False
        else:
            logger.error(f"错误: 必需的包 '{friendly_name}' 未安装，且已禁用自动安装。")
            return False


# --- 主要函数：确保向量数据库依赖项 (移除 FAISS-GPU 逻辑) ---


def ensure_vector_db_dependencies(
    db_type: Literal["faiss", "milvus_lite", "milvus"],
) -> bool:
    """
    根据指定的向量数据库类型，检查并自动安装所需的 Python 包。
    此版本只支持 FAISS-CPU，不再安装 FAISS-GPU。

    参数:
        db_type (Literal["faiss", "milvus_lite", "milvus"]):
            要使用的向量数据库类型。

    返回:
        bool: 如果所有必需的依赖项都已满足 (已存在或成功安装)，则返回 True；否则返回 False。
    """
    if not _UV_EXECUTABLE:
        logger.error("严重错误: UV 工具未找到。无法执行自动依赖安装。")
        logger.error("          请确保 'uv' 已安装并可从 PATH 访问。")
        return False

    logger.info(f"\n--- 正在为向量数据库类型 '{db_type}' 准备依赖项 ---")

    # 1. 检查并安装核心依赖项
    logger.info("检查核心依赖项...")
    core_deps_ok = True
    core_deps_ok &= _check_and_install_package_uv(
        "numpy", ">=2.2.0", friendly_name="Numpy"
    )
    core_deps_ok &= _check_and_install_package_uv(
        "langchain_text_splitters", ">=0.3.8", friendly_name="Langchain Text Splitters"
    )

    if not core_deps_ok:
        logger.error("严重错误: 核心依赖项未能满足。插件功能将受限或无法使用。")
        return False

    # 2. 根据 db_type 检查并安装可选依赖项
    optional_deps_ok = True
    if db_type == "faiss":
        logger.info("准备 FAISS 依赖项 (仅限 CPU 版本)...")
        optional_deps_ok &= _check_and_install_package_uv(
            "faiss", ">=1.10.0", friendly_name="faiss"
        )
        if not optional_deps_ok:
            logger.error("错误: FAISS 依赖未能满足。")

    elif db_type == "milvus_lite":
        logger.info("准备 Milvus Lite 依赖项...")
        optional_deps_ok &= _check_and_install_package_uv(
            "milvus-lite", ">=2.4.10", friendly_name="Milvus Lite"
        )
        if not optional_deps_ok:
            logger.error("错误: Milvus Lite 依赖未能满足。")

    elif db_type == "milvus":
        logger.info("准备 Milvus (Pymilvus) 依赖项...")
        optional_deps_ok &= _check_and_install_package_uv(
            "pymilvus", ">=2.5.0", friendly_name="Pymilvus (for Milvus Server)"
        )
        if not optional_deps_ok:
            logger.error("错误: Pymilvus 依赖未能满足。")
    else:
        logger.error(
            f"错误: 不支持的向量数据库类型: {db_type}。请选择 'faiss', 'milvus_lite' 或 'milvus'。"
        )
        optional_deps_ok = False

    if optional_deps_ok:
        logger.info(f"信息: 所有针对 '{db_type}' 的依赖项都已满足。")
    else:
        logger.error(f"错误: 针对 '{db_type}' 的部分或全部可选依赖项未能满足。")

    return optional_deps_ok
