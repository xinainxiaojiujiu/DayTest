"""
================================================================================
 汽车涂装不良品智能分析系统 - 解决方案生成 Agent
================================================================================
 功能说明：
   - 基于根因分析结果，自动生成针对性的纠正措施和预防方案
   - 从向量知识库检索历史成功案例中的解决方案
   - 调用 LLM 生成分级方案（临时纠正 + 长期预防）
   - 每条方案包含责任方、优先级和预估工作量

 方案生成原则：
   1. 针对性：每条方案必须直接对应一个根因
   2. 可操作性：方案必须具体到可执行步骤，包含参数调整的具体数值
   3. 分级处理：区分临时纠正措施（立即执行）和长期预防措施（制度级）
   4. 责任明确：每条方案需指定责任方
   5. 参考先例：优先引用历史成功案例中的方案

 优先级定义：
   high（高）: 对应高权重根因，需立即执行
   medium（中）: 对应中等权重根因，24小时内执行
   low（低）: 对应低权重根因，纳入周计划

 新手提示：
   - 方案总数不超过 6 条，按优先级排序
   - 每条方案都包含预估工作量，便于排期
   - 历史案例参考可大幅提升方案质量
================================================================================
"""

import logging
from typing import Any, Dict, List, Optional

from src.models.schemas import (
    PreprocessedData,
    RootCauseResult,
    SemanticLabel,
    SolutionItem,
    SolutionResult,
)
from src.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# ======================== LLM System Prompt ========================
SYSTEM_PROMPT = """你是一名汽车涂装工艺改善专家。根据根因分析结果，你需要生成针对性的纠正措施和预防方案。

生成方案时请遵循以下原则：
1. **针对性**: 每条方案必须直接对应一个根因，不可泛泛而谈
2. **可操作性**: 方案必须具体到可执行步骤，包含参数调整的具体数值
3. **分级处理**: 区分临时纠正措施(立即执行)和长期预防措施(制度级)
4. **责任明确**: 每条方案需指定责任方(操作员/工艺工程师/设备工程师/质量工程师)
5. **参考先例**: 优先引用历史成功案例中的方案

请严格按照以下JSON格式输出：
{
    "solutions": [
        {
            "action": "纠正措施描述(需具体、可操作)",
            "responsible": "责任方",
            "priority": "优先级(low/medium/high)",
            "estimated_effort": "预估工作量(如: 0.5人天)",
            "reference_case": "历史案例参考(如有)"
        }
    ]
}

要求：
- 针对每个根因至少生成1条纠正措施
- 高权重根因对应的方案优先级应为high
- 方案总数不超过6条
- 临时措施和长期措施需搭配"""


class SolutionAgent:
    """
    解决方案生成 Agent

    职责：
      1. 从向量知识库检索历史成功案例中的解决方案
      2. 调用 LLM 生成针对性的纠正措施和预防方案
      3. 按优先级排序，最多输出 6 条方案

    依赖注入：
      - vector_db: 向量数据库实例（可选，用于检索历史方案）

    使用示例：
        agent = SolutionAgent(vector_db)
        result = await agent.execute(preprocessed_data, semantic_label, root_cause_result)
    """

    def __init__(self, vector_db: Any = None) -> None:
        """
        初始化解决方案生成 Agent

        参数:
            vector_db: 向量数据库实例
        """
        self._vector_db = vector_db

    async def execute(
        self,
        preprocessed_data: PreprocessedData,
        semantic_label: SemanticLabel,
        root_cause_result: RootCauseResult,
    ) -> SolutionResult:
        """
        执行解决方案生成流程（主入口）

        流程：
          1. 检索历史成功案例中的解决方案
          2. 调用 LLM 生成方案
          3. 解析并排序方案列表

        参数:
            preprocessed_data: DataAgent 输出的预处理数据
            semantic_label: SemanticAgent 输出的语义标签
            root_cause_result: RootCauseAgent 输出的根因分析结果

        返回:
            SolutionResult: 包含最多 6 条按优先级排序的解决方案
        """
        logger.info(
            "SolutionAgent started | defect_id=%s | causes_count=%d",
            preprocessed_data.defect_id,
            len(root_cause_result.causes),
        )

        try:
            # 检索历史方案
            knowledge_context = await self._retrieve_solution_knowledge(
                semantic_label.defect_type, root_cause_result
            )

            # 调用 LLM 生成方案
            llm_result = await self._generate_with_llm(
                preprocessed_data=preprocessed_data,
                semantic_label=semantic_label,
                root_cause_result=root_cause_result,
                knowledge_context=knowledge_context,
            )

            # 解析方案列表
            solutions = self._parse_solutions(llm_result)

            result = SolutionResult(
                defect_id=preprocessed_data.defect_id,
                solutions=solutions,
            )

            logger.info(
                "SolutionAgent completed | defect_id=%s | solutions_count=%d",
                preprocessed_data.defect_id,
                len(result.solutions),
            )
            return result

        except Exception as exc:
            logger.error(
                "SolutionAgent failed | defect_id=%s | error=%s",
                preprocessed_data.defect_id,
                str(exc),
            )
            raise

    async def _retrieve_solution_knowledge(
        self,
        defect_type: str,
        root_cause_result: RootCauseResult,
    ) -> str:
        """
        从向量知识库检索历史成功案例中的解决方案

        使用不良类型 + 根因关键词组合作为检索查询

        参数:
            defect_type: 不良类型
            root_cause_result: 根因分析结果

        返回:
            str: 格式化的历史方案文本
        """
        if not self._vector_db:
            return ""
        try:
            cause_keywords = " ".join(
                [c.root_cause for c in root_cause_result.causes]
            )
            query = f"涂装{defect_type}解决方案 {cause_keywords}"
            results = await self._vector_db.similarity_search(query, top_k=5)
            if results:
                return "\n".join(
                    [f"- {r.get('content', '')}" for r in results]
                )
        except Exception as exc:
            logger.warning("Solution knowledge retrieval failed: %s", str(exc))
        return ""

    async def _generate_with_llm(
        self,
        preprocessed_data: PreprocessedData,
        semantic_label: SemanticLabel,
        root_cause_result: RootCauseResult,
        knowledge_context: str,
    ) -> Dict[str, Any]:
        """
        调用 LLM 生成解决方案

        组装包含不良现象、产品信息、根因分析结果、历史方案参考的完整 Prompt

        参数:
            preprocessed_data: 预处理数据
            semantic_label: 语义标签
            root_cause_result: 根因分析结果
            knowledge_context: 知识库检索结果

        返回:
            Dict[str, Any]: LLM 返回的 JSON 方案列表
        """
        # 格式化根因列表
        causes_text = "\n".join(
            [
                f"  {i + 1}. [{c.dimension}] {c.root_cause} (权重={c.weight:.2f})\n"
                f"     证据: {c.evidence}"
                + (f"\n     SOP参考: {c.sop_reference}" if c.sop_reference else "")
                for i, c in enumerate(root_cause_result.causes)
            ]
        )

        user_content = f"""请针对以下涂装不良品根因分析结果生成解决方案：

【不良现象】:
- 类型: {semantic_label.defect_type} | 严重程度: {semantic_label.severity}
- 描述: {semantic_label.description}

【产品信息】:
- 型号: {preprocessed_data.product_info.product_model}
- 产线: {preprocessed_data.product_info.line_id}

【根因分析结果】:
{causes_text}

【推理链路】: {root_cause_result.reasoning_chain}"""

        if knowledge_context:
            user_content += f"\n\n【历史解决方案参考】:\n{knowledge_context}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return await llm_client.chat_json(
            messages=messages, agent_name="solution_agent"
        )

    def _parse_solutions(self, llm_result: Dict[str, Any]) -> List[SolutionItem]:
        """
        解析 LLM 返回的方案列表

        将 LLM 返回的 JSON 中的 solutions 数组转换为 SolutionItem 对象列表，
        并按优先级排序（high > medium > low），最多返回 6 条。

        参数:
            llm_result: LLM 返回的 JSON 字典

        返回:
            List[SolutionItem]: 按优先级排序的方案列表（最多6条）
        """
        raw_solutions = llm_result.get("solutions", [])
        solutions: List[SolutionItem] = []

        for item in raw_solutions:
            try:
                solutions.append(
                    SolutionItem(
                        action=item.get("action", ""),
                        responsible=item.get("responsible", ""),
                        priority=item.get("priority", "medium"),
                        estimated_effort=item.get("estimated_effort", ""),
                        reference_case=item.get("reference_case"),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Failed to parse solution item: %s | error=%s", item, str(exc)
                )

        # 按优先级排序
        priority_order = {"high": 0, "medium": 1, "low": 2}
        solutions.sort(key=lambda s: priority_order.get(s.priority, 1))

        return solutions[:6]
