"""
================================================================================
 汽车涂装不良品智能分析系统 - 不良现象语义解析 Agent
================================================================================
 功能说明：
   - 接收不良品文字描述和缺陷图片，解析出标准化的语义标签
   - 支持 OCR 识别缺陷图片中的文字信息
   - 从向量知识库检索相似历史案例辅助解析
   - 调用 LLM 进行语义理解和分类

 不良分类体系（四级分类）：
   外观缺陷 → 缩孔、橘皮、流挂、颗粒、针孔、气泡、毛刺、划痕
   色差缺陷 → 色差、发花、露底、发雾、光泽不良
   附着力缺陷 → 脱落、起皮、附着力不足
   膜厚缺陷 → 膜厚偏薄、膜厚偏厚、膜厚不均

 严重程度分级：
   critical（致命）: 缩孔、脱落 — 直接影响产品功能
   high（严重）:     流挂、色差 — 严重影响外观质量
   medium（中等）:   橘皮、颗粒、针孔、气泡 — 需返工处理
   low（轻微）:      划痕、光泽不良 — 可让步接收

 容错设计：
   - LLM 调用失败时，使用关键词匹配进行降级解析
   - 降级解析的置信度标记为 0.3，便于后续人工复核

 新手提示：
   - 本 Agent 的输出（SemanticLabel）是后续根因推理的关键输入
   - 不良类型标签的准确性直接影响根因定位的精度
================================================================================
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.models.schemas import PreprocessedData, SemanticLabel
from src.utils.llm_client import llm_client
from src.utils.ocr_tool import recognize_defect_image

logger = logging.getLogger(__name__)

# ======================== 不良分类体系 ========================
DEFECT_TAXONOMY: Dict[str, List[str]] = {
    "外观缺陷": ["缩孔", "橘皮", "流挂", "颗粒", "针孔", "气泡", "毛刺", "划痕"],
    "色差缺陷": ["色差", "发花", "露底", "发雾", "光泽不良"],
    "附着力缺陷": ["脱落", "起皮", "附着力不足"],
    "膜厚缺陷": ["膜厚偏薄", "膜厚偏厚", "膜厚不均"],
}
"""不良品四级分类字典：大类 → 具体类型列表"""

# ======================== 严重程度映射表 ========================
SEVERITY_MAP: Dict[str, str] = {
    "缩孔": "critical",
    "脱落": "critical",
    "流挂": "high",
    "色差": "high",
    "橘皮": "medium",
    "颗粒": "medium",
    "针孔": "medium",
    "气泡": "medium",
    "划痕": "low",
    "光泽不良": "low",
}
"""不良类型 → 严重程度映射，用于降级解析时的默认严重程度"""

# ======================== LLM System Prompt ========================
SYSTEM_PROMPT = """你是一名汽车涂装不良品语义解析专家。你的任务是根据输入的不良品描述、缺陷图片OCR文本和工艺数据，准确解析出不良现象的语义标签。

请严格按照以下JSON格式输出：
{
    "defect_type": "具体不良类型(如:缩孔/橘皮/流挂/色差/颗粒等)",
    "defect_category": "不良大类(外观缺陷/色差缺陷/附着力缺陷/膜厚缺陷)",
    "severity": "严重程度(low/medium/high/critical)",
    "description": "用自然语言详细描述该不良现象的特征",
    "confidence": 0.95
}

注意事项：
1. defect_type 必须从已知分类中选择最匹配的
2. severity 需根据缺陷对产品功能的影响程度判断
3. description 需包含缺陷的视觉特征、位置、范围等关键信息
4. confidence 反映解析结果的可信度(0-1)
"""


class SemanticAgent:
    """
    不良现象语义解析 Agent

    职责：
      1. 接收不良品描述文本和缺陷图片路径
      2. 调用 OCR 引擎识别图片中的文字
      3. 从向量知识库检索相似案例
      4. 调用 LLM 进行语义解析，输出标准化标签

    依赖注入：
      - vector_db: 向量数据库实例（可选，用于检索历史案例）

    使用示例：
        agent = SemanticAgent(vector_db)
        label = await agent.execute(preprocessed_data, "表面出现橘皮", "defect.jpg")
    """

    def __init__(self, vector_db: Any = None) -> None:
        """
        初始化语义解析 Agent

        参数:
            vector_db: 向量数据库实例，用于检索相似历史案例
        """
        self._vector_db = vector_db

    async def execute(
        self,
        preprocessed_data: PreprocessedData,
        defect_description: str = "",
        image_path: Optional[str] = None,
    ) -> SemanticLabel:
        """
        执行语义解析流程（主入口）

        流程：
          1. 如果有图片，调用 OCR 识别图片文字
          2. 从向量知识库检索相似案例
          3. 调用 LLM 进行语义解析
          4. 组装 SemanticLabel 对象

        参数:
            preprocessed_data: DataAgent 输出的预处理数据
            defect_description: 不良品文字描述（可选）
            image_path: 缺陷图片文件路径（可选）

        返回:
            SemanticLabel: 包含不良类型、大类、严重程度、描述、置信度的标签
        """
        logger.info(
            "SemanticAgent started | defect_id=%s", preprocessed_data.defect_id
        )

        try:
            # OCR 识别缺陷图片文字
            ocr_text: Optional[str] = None
            if image_path:
                ocr_result = await recognize_defect_image(image_path)
                ocr_text = ocr_result.get("ocr_text") or None

            # 检索相似历史案例
            knowledge_context = await self._retrieve_knowledge(defect_description)

            # 调用 LLM 进行语义解析
            llm_result = await self._analyze_with_llm(
                defect_description=defect_description,
                ocr_text=ocr_text,
                product_info=preprocessed_data.product_info,
                knowledge_context=knowledge_context,
            )

            # 组装标签对象
            label = SemanticLabel(
                defect_type=llm_result.get("defect_type", "未知"),
                defect_category=llm_result.get("defect_category", "外观缺陷"),
                severity=llm_result.get("severity", "medium"),
                description=llm_result.get("description", ""),
                ocr_text=ocr_text,
                confidence=llm_result.get("confidence", 0.0),
            )

            logger.info(
                "SemanticAgent completed | defect_id=%s | type=%s | category=%s | severity=%s | confidence=%.2f",
                preprocessed_data.defect_id,
                label.defect_type,
                label.defect_category,
                label.severity,
                label.confidence,
            )
            return label

        except Exception as exc:
            logger.error(
                "SemanticAgent failed | defect_id=%s | error=%s",
                preprocessed_data.defect_id,
                str(exc),
            )
            # LLM 调用失败时使用关键词匹配降级
            return self._fallback_label(defect_description)

    async def _retrieve_knowledge(self, query: str) -> str:
        """
        从向量知识库检索相似历史案例

        参数:
            query: 检索查询文本

        返回:
            str: 格式化的历史案例文本，无结果时返回空字符串
        """
        if not self._vector_db or not query:
            return ""
        try:
            results = await self._vector_db.similarity_search(query, top_k=3)
            if results:
                return "\n".join(
                    [f"- {r.get('content', '')}" for r in results]
                )
        except Exception as exc:
            logger.warning("Knowledge retrieval failed: %s", str(exc))
        return ""

    async def _analyze_with_llm(
        self,
        defect_description: str,
        ocr_text: Optional[str],
        product_info: Any,
        knowledge_context: str,
    ) -> Dict[str, Any]:
        """
        调用 LLM 进行语义解析

        组装包含不良描述、OCR文本、产品信息、知识库参考的完整 Prompt

        参数:
            defect_description: 不良品文字描述
            ocr_text: OCR 识别的图片文字
            product_info: 产品基础信息
            knowledge_context: 知识库检索结果

        返回:
            Dict[str, Any]: LLM 返回的 JSON 解析结果
        """
        user_content_parts: List[str] = []

        if defect_description:
            user_content_parts.append(f"不良品描述: {defect_description}")
        if ocr_text:
            user_content_parts.append(f"缺陷图片OCR文本: {ocr_text}")

        user_content_parts.append(
            f"产品信息: 型号={product_info.product_model}, 产线={product_info.line_id}"
        )
        if knowledge_context:
            user_content_parts.append(f"工艺知识库参考:\n{knowledge_context}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_content_parts)},
        ]

        return await llm_client.chat_json(
            messages=messages, agent_name="semantic_agent"
        )

    def _fallback_label(self, description: str) -> SemanticLabel:
        """
        LLM 调用失败时的降级解析

        使用关键词匹配从 DEFECT_TAXONOMY 中查找匹配的不良类型。
        降级解析的置信度固定为 0.3，提示需要人工复核。

        参数:
            description: 不良品文字描述

        返回:
            SemanticLabel: 降级解析的标签（置信度 0.3）
        """
        matched_type = "未知"
        matched_category = "外观缺陷"

        # 遍历分类体系进行关键词匹配
        for category, types in DEFECT_TAXONOMY.items():
            for t in types:
                if t in description:
                    matched_type = t
                    matched_category = category
                    break
            if matched_type != "未知":
                break

        severity = SEVERITY_MAP.get(matched_type, "medium")

        return SemanticLabel(
            defect_type=matched_type,
            defect_category=matched_category,
            severity=severity,
            description=description or "无法解析不良描述",
            confidence=0.3,
        )
