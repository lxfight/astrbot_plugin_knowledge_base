"""
数据迁移工具
支持从旧格式迁移到新的增强格式
提供零停机迁移和安全回滚
"""

import os
import json
import pickle
import shutil
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

from .base import Document
from .enhanced_faiss_store import EnhancedFaissStore, EnhancedDocument
from .faiss_store import FaissStore as OldFaissStore
from .astrbot_faiss_store import FaissStore as AstrBotFaissStore

class MigrationTool:
    """数据迁移工具"""
    
    def __init__(self, data_path: str, backup_path: Optional[str] = None):
        self.data_path = Path(data_path)
        self.backup_path = Path(backup_path) if backup_path else self.data_path / "backup"
        self.migration_log = self.data_path / "migration.log"
        
        # 确保备份目录存在
        self.backup_path.mkdir(exist_ok=True)
        
    def detect_old_format(self) -> List[str]:
        """检测旧格式集合"""
        old_collections = []
        
        if not self.data_path.exists():
            return old_collections
        
        # 检测旧格式文件
        for file in self.data_path.iterdir():
            if file.suffix == '.index':
                collection_name = file.stem
                old_db = file.with_suffix('.db')
                old_docs = file.with_suffix('.docs')
                
                # 检查是否存在对应的旧格式文件
                if old_db.exists() or old_docs.exists():
                    # 检查是否已经是新格式
                    new_db = file.with_suffix('.enhanced.db')
                    new_faiss = file.with_suffix('.faiss')
                    
                    if not (new_db.exists() and new_faiss.exists()):
                        old_collections.append(collection_name)
        
        return old_collections
    
    def create_backup(self, collection_name: str) -> bool:
        """创建备份"""
        try:
            backup_dir = self.backup_path / f"{collection_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(exist_ok=True)
            
            # 备份旧格式文件
            files_to_backup = [
                f"{collection_name}.index",
                f"{collection_name}.db",
                f"{collection_name}.docs"
            ]
            
            for filename in files_to_backup:
                src = self.data_path / filename
                if src.exists():
                    shutil.copy2(src, backup_dir / filename)
            
            # 记录备份信息
            with open(self.migration_log, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} - 备份集合 {collection_name} 到 {backup_dir}\n")
            
            logging.info(f"成功创建备份: {backup_dir}")
            return True
            
        except Exception as e:
            logging.error(f"创建备份失败: {e}")
            return False
    
    def migrate_collection(
        self, 
        collection_name: str, 
        embedding_util: Any,
        force: bool = False
    ) -> bool:
        """迁移单个集合"""
        try:
            # 检查是否需要迁移
            if not force and collection_name not in self.detect_old_format():
                logging.info(f"集合 {collection_name} 无需迁移")
                return True
            
            # 创建备份
            if not self.create_backup(collection_name):
                return False
            
            # 确定旧格式类型
            old_format_type = self._determine_old_format_type(collection_name)
            logging.info(f"检测到旧格式类型: {old_format_type}")
            
            # 加载旧数据
            old_documents = self._load_old_format(collection_name, old_format_type, embedding_util)
            if not old_documents:
                logging.warning(f"集合 {collection_name} 没有可迁移的数据")
                return True
            
            # 创建新的存储
            new_store = EnhancedFaissStore(embedding_util, str(self.data_path))
            
            # 迁移数据
            asyncio.run(self._migrate_documents(new_store, collection_name, old_documents))
            
            # 验证迁移
            if self._validate_migration(collection_name, len(old_documents)):
                # 标记迁移完成
                self._mark_migration_complete(collection_name)
                logging.info(f"集合 {collection_name} 迁移成功")
                return True
            else:
                # 回滚
                self.rollback(collection_name)
                return False
                
        except Exception as e:
            logging.error(f"迁移集合 {collection_name} 失败: {e}")
            self.rollback(collection_name)
            return False
    
    def _determine_old_format_type(self, collection_name: str) -> str:
        """确定旧格式类型"""
        index_file = self.data_path / f"{collection_name}.index"
        db_file = self.data_path / f"{collection_name}.db"
        docs_file = self.data_path / f"{collection_name}.docs"
        
        if db_file.exists():
            # 检查是否是pickle格式
            try:
                with open(db_file, 'rb') as f:
                    data = f.read(2)
                    if data in [b"\x80\x04", b"\x80\x03", b"\x80\x02"]:
                        return "faiss_store"
            except:
                pass
        
        if docs_file.exists():
            return "astrbot_faiss_store"
        
        return "unknown"
    
    def _load_old_format(
        self, 
        collection_name: str, 
        format_type: str,
        embedding_util: Any
    ) -> List[Document]:
        """加载旧格式数据"""
        documents = []
        
        try:
            if format_type == "faiss_store":
                # 加载旧的FaissStore格式
                old_store = OldFaissStore(embedding_util, str(self.data_path))
                asyncio.run(old_store.initialize())
                
                # 获取文档
                docs = old_store.db.get(collection_name, [])
                for doc in docs:
                    documents.append(Document(
                        text_content=doc.text_content,
                        embedding=doc.embedding,
                        metadata=doc.metadata,
                        id=doc.id
                    ))
                
                asyncio.run(old_store.close())
                
            elif format_type == "astrbot_faiss_store":
                # 加载AstrBot格式
                old_store = AstrBotFaissStore(embedding_util, str(self.data_path))
                asyncio.run(old_store.initialize())
                
                # 检查是否是旧格式
                if collection_name in old_store._old_collections:
                    # 从旧存储获取
                    if old_store._old_faiss_store:
                        docs = old_store._old_faiss_store.db.get(collection_name, [])
                        for doc in docs:
                            documents.append(Document(
                                text_content=doc.text_content,
                                embedding=doc.embedding,
                                metadata=doc.metadata,
                                id=doc.id
                            ))
                
                asyncio.run(old_store.close())
                
        except Exception as e:
            logging.error(f"加载旧格式数据失败: {e}")
        
        return documents
    
    async def _migrate_documents(
        self, 
        new_store: EnhancedFaissStore, 
        collection_name: str, 
        documents: List[Document]
    ):
        """迁移文档到新格式"""
        # 分批迁移以避免内存问题
        batch_size = 100
        total_docs = len(documents)
        
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            await new_store.add_documents(collection_name, batch)
            
            logging.info(f"已迁移 {min(i+batch_size, total_docs)}/{total_docs} 个文档")
    
    def _validate_migration(self, collection_name: str, expected_count: int) -> bool:
        """验证迁移结果"""
        try:
            # 检查新格式文件是否存在
            new_db = self.data_path / f"{collection_name}.enhanced.db"
            new_faiss = self.data_path / f"{collection_name}.faiss"
            
            if not (new_db.exists() and new_faiss.exists()):
                logging.error("新格式文件未创建")
                return False
            
            # 检查文档数量
            new_store = EnhancedFaissStore(None, str(self.data_path))
            actual_count = asyncio.run(new_store.count_documents(collection_name))
            
            if actual_count != expected_count:
                logging.error(f"文档数量不匹配: 期望 {expected_count}, 实际 {actual_count}")
                return False
            
            # 检查是否可以搜索
            # 这里可以添加更详细的验证
            
            return True
            
        except Exception as e:
            logging.error(f"验证失败: {e}")
            return False
    
    def _mark_migration_complete(self, collection_name: str):
        """标记迁移完成"""
        marker_file = self.data_path / f"{collection_name}.migrated"
        with open(marker_file, 'w') as f:
            f.write(datetime.now().isoformat())
    
    def rollback(self, collection_name: str) -> bool:
        """回滚迁移"""
        try:
            # 查找最新的备份
            backup_dirs = [d for d in self.backup_path.iterdir() 
                          if d.name.startswith(f"{collection_name}_")]
            
            if not backup_dirs:
                logging.warning(f"未找到集合 {collection_name} 的备份")
                return False
            
            latest_backup = max(backup_dirs, key=lambda x: x.stat().st_mtime)
            
            # 删除新格式文件
            new_files = [
                f"{collection_name}.enhanced.db",
                f"{collection_name}.faiss",
                f"{collection_name}.migrated"
            ]
            
            for filename in new_files:
                file_path = self.data_path / filename
                if file_path.exists():
                    file_path.unlink()
            
            # 恢复旧格式文件
            for backup_file in latest_backup.iterdir():
                shutil.copy2(backup_file, self.data_path / backup_file.name)
            
            logging.info(f"成功回滚集合 {collection_name}")
            return True
            
        except Exception as e:
            logging.error(f"回滚失败: {e}")
            return False
    
    def migrate_all(self, embedding_util: Any, force: bool = False) -> Dict[str, bool]:
        """迁移所有旧格式集合"""
        old_collections = self.detect_old_format()
        results = {}
        
        logging.info(f"检测到 {len(old_collections)} 个需要迁移的集合")
        
        for collection_name in old_collections:
            logging.info(f"开始迁移集合: {collection_name}")
            success = self.migrate_collection(collection_name, embedding_util, force)
            results[collection_name] = success
            
            if success:
                logging.info(f"集合 {collection_name} 迁移成功")
            else:
                logging.error(f"集合 {collection_name} 迁移失败")
        
        return results
    
    def get_migration_status(self) -> Dict[str, str]:
        """获取迁移状态"""
        status = {}
        
        # 检测旧格式
        old_collections = self.detect_old_format()
        
        # 检测已迁移的集合
        migrated_collections = []
        for file in self.data_path.glob("*.migrated"):
            collection_name = file.stem
            migrated_collections.append(collection_name)
        
        # 构建状态
        for collection in old_collections:
            status[collection] = "pending"
        
        for collection in migrated_collections:
            status[collection] = "completed"
        
        return status
    
    def cleanup_old_files(self, collection_name: str) -> bool:
        """清理旧格式文件（确认迁移成功后）"""
        try:
            # 检查迁移标记
            marker_file = self.data_path / f"{collection_name}.migrated"
            if not marker_file.exists():
                logging.warning(f"集合 {collection_name} 未标记为已迁移")
                return False
            
            # 删除旧格式文件
            old_files = [
                f"{collection_name}.index",
                f"{collection_name}.db",
                f"{collection_name}.docs"
            ]
            
            for filename in old_files:
                file_path = self.data_path / filename
                if file_path.exists():
                    file_path.unlink()
                    logging.info(f"删除旧文件: {filename}")
            
            # 删除迁移标记
            marker_file.unlink()
            
            return True
            
        except Exception as e:
            logging.error(f"清理旧文件失败: {e}")
            return False

# 迁移进度跟踪器
class MigrationProgress:
    """迁移进度跟踪"""
    
    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.failed = 0
        self.errors = []
    
    def update(self, success: bool, error: Optional[str] = None):
        """更新进度"""
        if success:
            self.completed += 1
        else:
            self.failed += 1
            if error:
                self.errors.append(error)
    
    @property
    def progress(self) -> float:
        """获取进度百分比"""
        return (self.completed + self.failed) / self.total * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "progress": self.progress,
            "errors": self.errors
        }