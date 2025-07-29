"""
API重排序配置模块
支持多种API服务提供商的配置管理
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


@dataclass
class APIRerankConfig:
    """API重排序配置类"""
    
    # 服务提供商
    provider: str = "cohere"  # cohere, jina, azure, custom
    
    # API配置
    api_key: str = ""
    api_url: str = ""
    timeout: int = 30
    max_retries: int = 3
    
    # 缓存配置
    enable_cache: bool = True
    cache_ttl: int = 3600  # 缓存时间(秒)
    max_cache_size: int = 1000  # 最大缓存条目
    
    # 降级策略
    fallback_strategy: str = "simple"  # simple, none, exception
    use_local_if_api_fails: bool = True
    
    # 批处理配置
    batch_size: int = 32
    max_documents: int = 100  # 最大重排序文档数
    
    # 模型配置
    model_name: Optional[str] = None  # 特定模型名称
    
    def __post_init__(self):
        """初始化后处理"""
        # 从环境变量读取API密钥
        if not self.api_key:
            env_keys = {
                "cohere": "COHERE_API_KEY",
                "jina": "JINA_API_KEY",
                "azure": "AZURE_OPENAI_API_KEY",
                "custom": "CUSTOM_RERANK_API_KEY"
            }
            self.api_key = os.getenv(env_keys.get(self.provider, ""), "")
        
        # 设置默认API URL
        if not self.api_url:
            default_urls = {
                "cohere": "https://api.cohere.ai/v1/rerank",
                "jina": "https://api.jina.ai/v1/rerank",
                "azure": "https://{resource}.openai.azure.com/openai/deployments/{deployment}/rerank",
                "custom": ""
            }
            self.api_url = default_urls.get(self.provider, "")
    
    def validate(self) -> bool:
        """验证配置有效性"""
        if self.provider not in ["cohere", "jina", "azure", "custom"]:
            return False
        
        if self.provider != "custom" and not self.api_key:
            return False
        
        if self.timeout <= 0 or self.max_retries <= 0:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "provider": self.provider,
            "api_url": self.api_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "enable_cache": self.enable_cache,
            "cache_ttl": self.cache_ttl,
            "fallback_strategy": self.fallback_strategy,
            "batch_size": self.batch_size,
            "max_documents": self.max_documents,
            "model_name": self.model_name
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "APIRerankConfig":
        """从字典创建配置"""
        return cls(**config_dict)