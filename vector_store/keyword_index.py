"""
关键词索引实现
基于SQLite FTS5的全文搜索和倒排索引
支持BM25算法和TF-IDF权重计算
"""

import sqlite3
import math
import re
from typing import List, Tuple, Dict, Set
from collections import defaultdict, Counter
import json

class KeywordIndex:
    """关键词索引管理器"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化关键词索引数据库"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # 创建FTS5全文索引表
        conn.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS keyword_index USING fts5(
                doc_id,
                content,
                keywords,
                metadata,
                tokenize='porter unicode61'
            )
        ''')
        
        # 创建关键词统计表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS keyword_stats (
                keyword TEXT PRIMARY KEY,
                doc_count INTEGER,
                total_freq INTEGER,
                idf REAL
            )
        ''')
        
        # 创建文档关键词频率表
        conn.execute('''
            CREATE TABLE IF NOT EXISTS doc_keywords (
                doc_id TEXT,
                keyword TEXT,
                tf REAL,
                tf_idf REAL,
                PRIMARY KEY (doc_id, keyword)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_document(self, doc_id: str, content: str, keywords: List[str], metadata: Dict = None):
        """添加文档到关键词索引"""
        conn = sqlite3.connect(self.db_path)
        
        # 添加到FTS索引
        conn.execute('''
            INSERT OR REPLACE INTO keyword_index (doc_id, content, keywords, metadata)
            VALUES (?, ?, ?, ?)
        ''', (
            doc_id,
            content,
            ' '.join(keywords),
            json.dumps(metadata or {})
        ))
        
        # 更新关键词统计
        self._update_keyword_stats(conn, doc_id, content, keywords)
        
        conn.commit()
        conn.close()
    
    def _update_keyword_stats(self, conn: sqlite3.Connection, doc_id: str, content: str, keywords: List[str]):
        """更新关键词统计信息"""
        # 计算词频
        word_freq = Counter(self._tokenize(content))
        
        # 计算TF-IDF
        total_docs = self._get_total_docs(conn)
        
        for keyword in set(keywords):
            tf = word_freq.get(keyword, 0) / max(len(word_freq), 1)
            
            # 更新关键词统计
            cursor = conn.execute(
                'SELECT doc_count FROM keyword_stats WHERE keyword = ?', (keyword,)
            )
            row = cursor.fetchone()
            
            if row:
                doc_count = row[0] + 1
                conn.execute('''
                    UPDATE keyword_stats 
                    SET doc_count = ?, total_freq = total_freq + ?
                    WHERE keyword = ?
                ''', (doc_count, word_freq.get(keyword, 0), keyword))
            else:
                conn.execute('''
                    INSERT INTO keyword_stats (keyword, doc_count, total_freq)
                    VALUES (?, 1, ?)
                ''', (keyword, word_freq.get(keyword, 0)))
            
            # 计算IDF
            idf = math.log(total_docs + 1) - math.log(doc_count + 1) + 1
            tf_idf = tf * idf
            
            # 更新文档关键词表
            conn.execute('''
                INSERT OR REPLACE INTO doc_keywords (doc_id, keyword, tf, tf_idf)
                VALUES (?, ?, ?, ?)
            ''', (doc_id, keyword, tf, tf_idf))
    
    def search(self, query: str, limit: int = 100) -> List[Tuple[str, float]]:
        """搜索相关文档"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # FTS5搜索
        cursor = conn.execute('''
            SELECT doc_id, rank FROM keyword_index 
            WHERE keyword_index MATCH ?
            ORDER BY rank
            LIMIT ?
        ''', (query, limit))
        
        results = [(row['doc_id'], row['rank']) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def search_with_bm25(self, query: str, k1: float = 1.2, b: float = 0.75, limit: int = 100) -> List[Tuple[str, float]]:
        """使用BM25算法搜索"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # 获取查询词
        query_terms = self._tokenize(query)
        if not query_terms:
            return []
        
        # 获取相关文档
        doc_scores = defaultdict(float)
        
        # 计算平均文档长度
        avg_doc_len = self._get_avg_doc_length(conn)
        
        for term in query_terms:
            # 获取包含该词的文档
            cursor = conn.execute('''
                SELECT doc_id, tf FROM doc_keywords WHERE keyword = ?
            ''', (term,))
            
            for row in cursor.fetchall():
                doc_id = row['doc_id']
                tf = row['tf']
                
                # 计算BM25分数
                score = self._calculate_bm25_score(
                    tf, self._get_doc_length(conn, doc_id), 
                    avg_doc_len, self._get_df(conn, term), 
                    k1, b
                )
                doc_scores[doc_id] += score
        
        conn.close()
        
        # 排序并返回结果
        sorted_results = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:limit]
    
    def get_keywords(self, doc_id: str) -> List[Tuple[str, float]]:
        """获取文档的关键词及其权重"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT keyword, tf_idf FROM doc_keywords 
            WHERE doc_id = ? 
            ORDER BY tf_idf DESC
        ''', (doc_id,))
        
        keywords = [(row['keyword'], row['tf_idf']) for row in cursor.fetchall()]
        
        conn.close()
        return keywords
    
    def delete_document(self, doc_id: str):
        """从索引中删除文档"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('DELETE FROM keyword_index WHERE doc_id = ?', (doc_id,))
        conn.execute('DELETE FROM doc_keywords WHERE doc_id = ?', (doc_id,))
        
        conn.commit()
        conn.close()
    
    def _tokenize(self, text: str) -> List[str]:
        """分词处理"""
        # 简单的英文分词
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # 过滤停用词
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within',
            'without', 'toward', 'against', 'upon', 'under', 'over', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'can', 'will', 'just', 'don', 'should', 'now'
        }
        return [word for word in words if word not in stop_words and len(word) > 2]
    
    def _get_total_docs(self, conn: sqlite3.Connection) -> int:
        """获取总文档数"""
        cursor = conn.execute('SELECT COUNT(DISTINCT doc_id) as count FROM doc_keywords')
        return cursor.fetchone()['count'] or 1
    
    def _get_df(self, conn: sqlite3.Connection, term: str) -> int:
        """获取文档频率"""
        cursor = conn.execute(
            'SELECT doc_count FROM keyword_stats WHERE keyword = ?', (term,)
        )
        row = cursor.fetchone()
        return row['doc_count'] if row else 0
    
    def _get_doc_length(self, conn: sqlite3.Connection, doc_id: str) -> int:
        """获取文档长度"""
        cursor = conn.execute(
            'SELECT LENGTH(content) as length FROM keyword_index WHERE doc_id = ?', 
            (doc_id,)
        )
        row = cursor.fetchone()
        return row['length'] if row else 0
    
    def _get_avg_doc_length(self, conn: sqlite3.Connection) -> float:
        """获取平均文档长度"""
        cursor = conn.execute('SELECT AVG(LENGTH(content)) as avg_len FROM keyword_index')
        result = cursor.fetchone()
        return result['avg_len'] or 100.0
    
    def _calculate_bm25_score(self, tf: float, doc_len: int, avg_doc_len: float, 
                            df: int, k1: float, b: float) -> float:
        """计算BM25分数"""
        total_docs = self._get_total_docs(sqlite3.connect(self.db_path))
        idf = math.log((total_docs - df + 0.5) / (df + 0.5))
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
        
        return idf * numerator / denominator
    
    def get_statistics(self) -> Dict[str, int]:
        """获取索引统计信息"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        cursor = conn.execute('''
            SELECT 
                COUNT(DISTINCT doc_id) as doc_count,
                COUNT(DISTINCT keyword) as keyword_count,
                SUM(total_freq) as total_terms
            FROM keyword_stats
        ''')
        
        stats = dict(cursor.fetchone())
        
        conn.close()
        return stats
    
    def close(self):
        """关闭索引"""
        pass  # SQLite连接是按需创建的