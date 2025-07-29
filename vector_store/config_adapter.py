"""
AstrBot配置适配器
将AstrBot的配置字典转换为API重排序配置
"""

from .api_rerank_config import APIRerankConfig
from .rerank_service import EnhancedHybridReranker


def create_rerank_config_from_astrbot(config: dict) -> dict:
    """
    从AstrBot配置创建重排序配置
    
    Args:
        config: AstrBot配置字典
        
    Returns:
        重排序配置字典
    """
    rerank_config = config.get("rerank_config", {})
    return {
        "strategy": rerank_config.get("strategy", "auto"),
        "api": {
            "provider": rerank_config.get("api_provider", "cohere"),
            "api_key": rerank_config.get("api_key", ""),
            "api_url": rerank_config.get("api_url", ""),
            "timeout": rerank_config.get("timeout", 30),
            "max_retries": rerank_config.get("max_retries", 3),
            "enable_cache": rerank_config.get("enable_cache", True),
            "cache_ttl": rerank_config.get("cache_ttl", 3600)
        },
        "cross_encoder": {
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "batch_size": 32,
            "max_length": 512,
            "use_gpu": False
        }
    }


def create_reranker_from_astrbot_config(config: dict):
    """
    从AstrBot配置创建重排序器
    
    Args:
        config: AstrBot配置字典
        
    Returns:
        EnhancedHybridReranker实例
    """
    rerank_config = create_rerank_config_from_astrbot(config)
    return EnhancedHybridReranker(rerank_config)


def get_api_rerank_config_from_astrbot(config: dict) -> APIRerankConfig:
    """
    从AstrBot配置创建API重排序配置
    
    Args:
        config: AstrBot配置字典
        
    Returns:
        APIRerankConfig实例
    """
    rerank_config = config.get("rerank_config", {})
    return APIRerankConfig(
        provider=rerank_config.get("api_provider", "cohere"),
        api_key=rerank_config.get("api_key", ""),
        api_url=rerank_config.get("api_url", ""),
        timeout=rerank_config.get("timeout", 30),
        max_retries=rerank_config.get("max_retries", 3),
        enable_cache=rerank_config.get("enable_cache", True),
        cache_ttl=rerank_config.get("cache_ttl", 3600)
    )


def validate_rerank_config(config: dict) -> tuple[bool, str]:
    """
    验证重排序配置
    
    Args:
        config: AstrBot配置字典
        
    Returns:
        (是否有效, 错误信息)
    """
    rerank_config = config.get("rerank_config", {})
    strategy = rerank_config.get("strategy", "auto")
    if strategy not in ["auto", "api", "cross_encoder", "simple"]:
        return False, f"无效的重排序策略: {strategy}"
    
    if strategy in ["api", "auto"]:
        provider = rerank_config.get("api_provider", "cohere")
        if provider not in ["cohere", "jina", "custom"]:
            return False, f"无效的API提供商: {provider}"
        
        if provider != "custom" and not rerank_config.get("api_key"):
            # 检查环境变量
            env_keys = {
                "cohere": "COHERE_API_KEY",
                "jina": "JINA_API_KEY"
            }
            if not env_keys.get(provider) or not __import__("os").getenv(env_keys[provider]):
                return False, f"缺少{provider}的API密钥"
    
    return True, ""


def get_rerank_config_summary(config: dict) -> dict:
    """
    获取重排序配置摘要
    
    Args:
        config: AstrBot配置字典
        
    Returns:
        配置摘要字典
    """
    rerank_config = config.get("rerank_config", {})
    strategy = rerank_config.get("strategy", "auto")
    provider = rerank_config.get("api_provider", "cohere")
    
    summary = {
        "strategy": strategy,
        "provider": provider if strategy in ["api", "auto"] else None,
        "cache_enabled": rerank_config.get("enable_cache", True),
        "timeout": rerank_config.get("timeout", 30),
        "max_retries": rerank_config.get("max_retries", 3)
    }
    
    # 检查API密钥状态
    if strategy in ["api", "auto"]:
        api_key = rerank_config.get("api_key")
        if not api_key:
            env_keys = {
                "cohere": "COHERE_API_KEY",
                "jina": "JINA_API_KEY"
            }
            env_key = env_keys.get(provider)
            if env_key and __import__("os").getenv(env_key):
                summary["api_key_status"] = "环境变量"
            else:
                summary["api_key_status"] = "未配置"
        else:
            summary["api_key_status"] = "已配置"
    
    return summary