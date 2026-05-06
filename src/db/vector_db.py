"""
================================================================================
 汽车涂装不良品智能分析系统 - ChromaDB 向量数据库操作模块
================================================================================
 功能说明：
   - 管理 ChromaDB HTTP 客户端的创建与复用
   - 提供语义相似度检索（基于向量嵌入）
   - 支持文档的增删改查操作
   - 自动管理 Collection 的创建和获取

 数据说明：
   collection: process_knowledge（工艺知识库）
   存储内容: 不良品分析案例的语义向量，用于相似案例检索

 新手提示：
   - 向量数据库用于存储和检索"语义相似"的历史案例
   - 当 LLM 分析出有效方案后，案例会自动沉淀到知识库
   - ChromaDB 未连接时检索返回空列表，不影响主流程
================================================================================
"""

import logging
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class VectorDBError(Exception):
    """向量数据库操作异常类"""
    pass


_client = None


def get_chroma_client():
    """
    获取 ChromaDB HTTP 客户端实例（懒加载单例模式）

    首次调用时根据配置创建 HTTP 客户端连接，后续调用直接返回已有实例。

    返回:
        chromadb.HttpClient 实例
    """
    global _client
    if _client is None:
        import chromadb
        
        _client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
        )
    return _client


class VectorDB:
    """
    ChromaDB 向量数据库操作封装

    职责：
      1. 管理 ChromaDB 客户端和 Collection 生命周期
      2. 提供语义相似度检索（用于查找历史相似案例）
      3. 支持知识文档的增删改查
      4. 自动创建 Collection（首次使用时）

    使用示例：
        vdb = VectorDB()
        results = await vdb.similarity_search("缩孔缺陷 橘皮", top_k=5)
        await vdb.add_document("doc-001", "案例内容...", {"type": "缩孔"})
        count = await vdb.count_documents()
    """

    def __init__(self) -> None:
        """初始化向量数据库实例，加载 Collection 名称配置"""
        self._collection_name = settings.CHROMA_COLLECTION
        self._collection = None

    def _get_client(self):
        """获取 ChromaDB HTTP 客户端实例"""
        return get_chroma_client()

    def _get_collection(self):
        """
        获取或创建 ChromaDB Collection

        首次调用时尝试获取已存在的 Collection，
        如果不存在则自动创建新的 Collection。
        """
        if self._collection is None:
            client = self._get_client()
            try:
                self._collection = client.get_collection(
                    name=self._collection_name
                )
            except Exception:
                self._collection = client.create_collection(
                    name=self._collection_name,
                    metadata={"description": "Coating process knowledge base"},
                )
        return self._collection

    async def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_criteria: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        语义相似度检索

        根据查询文本在向量知识库中检索最相似的 Top-K 个文档。
        用于查找与当前不良品最相似的历史案例，辅助根因推理。

        参数:
            query: 查询文本（不良品描述或特征）
            top_k: 返回的最相似文档数量（默认5）
            filter_criteria: 元数据过滤条件（可选，如 {"defect_type": "缩孔"}）

        返回:
            List[Dict]: 相似文档列表，每项包含 id, content, distance, metadata
                       按相似度降序排列（distance 越小越相似）
        """
        try:
            collection = self._get_collection()
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=filter_criteria,
            )

            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            formatted: List[Dict[str, Any]] = []
            ids = results["ids"][0]
            distances = results.get("distances", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            documents = results.get("documents", [[]])[0]

            for i, doc_id in enumerate(ids):
                formatted.append({
                    "id": doc_id,
                    "content": documents[i] if i < len(documents) else "",
                    "distance": distances[i] if i < len(distances) else 0.0,
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                })

            logger.debug(
                "VectorDB similarity search | query=%s | top_k=%d | results=%d",
                query[:50],
                top_k,
                len(formatted),
            )
            return formatted

        except Exception as exc:
            logger.error(
                "VectorDB similarity search failed | query=%s | error=%s",
                query[:50],
                str(exc),
            )
            return []

    async def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        向知识库添加文档

        将不良品分析案例写入向量数据库，系统会自动对 content 进行向量嵌入。
        通常由 ClosedLoopAgent 在验证方案有效后自动调用。

        参数:
            doc_id: 文档唯一标识（建议使用 defect_id）
            content: 文档文本内容（不良类型+根因+方案摘要）
            metadata: 附加元数据（如不良类型、严重程度等）

        异常:
            VectorDBError: 写入失败时抛出
        """
        try:
            collection = self._get_collection()
            collection.add(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata or {}],
            )
            logger.info(
                "VectorDB document added | doc_id=%s | content_len=%d",
                doc_id,
                len(content),
            )

        except Exception as exc:
            logger.error(
                "VectorDB add document failed | doc_id=%s | error=%s",
                doc_id,
                str(exc),
            )
            raise VectorDBError(f"Failed to add document: {exc}") from exc

    async def update_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        更新知识库中的文档

        当同一不良品的分析结果需要修正或补充时使用。

        参数:
            doc_id: 文档唯一标识
            content: 更新后的文档内容
            metadata: 更新后的元数据

        异常:
            VectorDBError: 更新失败时抛出
        """
        try:
            collection = self._get_collection()
            collection.update(
                ids=[doc_id],
                documents=[content],
                metadatas=[metadata or {}],
            )
            logger.info("VectorDB document updated | doc_id=%s", doc_id)

        except Exception as exc:
            logger.error(
                "VectorDB update document failed | doc_id=%s | error=%s",
                doc_id,
                str(exc),
            )
            raise VectorDBError(f"Failed to update document: {exc}") from exc

    async def delete_document(self, doc_id: str) -> None:
        """
        从知识库中删除文档

        参数:
            doc_id: 要删除的文档唯一标识

        异常:
            VectorDBError: 删除失败时抛出
        """
        try:
            collection = self._get_collection()
            collection.delete(ids=[doc_id])
            logger.info("VectorDB document deleted | doc_id=%s", doc_id)

        except Exception as exc:
            logger.error(
                "VectorDB delete document failed | doc_id=%s | error=%s",
                doc_id,
                str(exc),
            )
            raise VectorDBError(f"Failed to delete document: {exc}") from exc

    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 ID 获取知识库文档

        参数:
            doc_id: 文档唯一标识

        返回:
            Optional[Dict]: 文档信息（id, content, metadata），不存在时返回 None
        """
        try:
            collection = self._get_collection()
            results = collection.get(ids=[doc_id])

            if not results or not results.get("ids"):
                return None

            return {
                "id": results["ids"][0],
                "content": results["documents"][0] if results.get("documents") else "",
                "metadata": results["metadatas"][0] if results.get("metadatas") else {},
            }

        except Exception as exc:
            logger.error(
                "VectorDB get document failed | doc_id=%s | error=%s",
                doc_id,
                str(exc),
            )
            return None

    async def count_documents(self) -> int:
        """
        获取知识库文档总数

        返回:
            int: Collection 中的文档数量，失败时返回 0
        """
        try:
            collection = self._get_collection()
            return collection.count()
        except Exception as exc:
            logger.error("VectorDB count failed: %s", str(exc))
            return 0
