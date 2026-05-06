"""
================================================================================
 汽车涂装不良品智能分析系统 - 闭环验证与知识库迭代 Agent
================================================================================
 功能说明：
   - 评估已实施解决方案的有效性
   - 决定是否将本次分析经验沉淀到工艺知识库
   - 自动将高价值案例写入向量数据库供后续参考
   - 将验证结果持久化到 MySQL 数据库

 闭环验证维度：
   1. 方案执行度：解决方案是否已按计划执行？
   2. 效果验证：实施后同类不良是否减少？不良率是否下降？
   3. 副作用检查：方案是否引入新的质量问题？
   4. 知识沉淀价值：本次分析是否包含可复用的经验？

 知识库沉淀条件：
   - 验证状态为 "passed"
   - 有效性评分 >= 0.7
   - 满足以上条件时自动写入向量数据库

 新手提示：
   - 本 Agent 是分析流程的最后一步，形成完整的 PDCA 闭环
   - 知识库自动沉淀机制确保系统越用越智能
   - 有效性评分 0.7 是默认阈值，可在配置中调整
================================================================================
"""

import logging
from typing import Any, Dict, Optional

from src.models.schemas import (
    ClosedLoopResult,
    RootCauseResult,
    SemanticLabel,
    SolutionResult,
)
from src.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# ======================== LLM System Prompt ========================
SYSTEM_PROMPT = """你是一名汽车涂装质量闭环验证专家。你的任务是评估已实施的解决方案是否有效，并决定是否将本次分析经验沉淀到工艺知识库中。

评估维度：
1. **方案执行度**: 解决方案是否已按计划执行？
2. **效果验证**: 实施后同类不良是否减少？不良率是否下降？
3. **副作用检查**: 方案是否引入新的质量问题？
4. **知识沉淀价值**: 本次分析是否包含可复用的经验？

请严格按照以下JSON格式输出：
{
    "verification_status": "验证状态(passed/failed/pending)",
    "effectiveness_score": 0.85,
    "knowledge_updated": true,
    "feedback": "闭环反馈描述(包含验证结论和改进建议)"
}

要求：
- effectiveness_score 范围 0-1，0.7以上为有效
- knowledge_updated 为 true 时需说明沉淀内容
- 如果验证失败，feedback 中需包含改进方向"""


class ClosedLoopAgent:
    """
    闭环验证与知识库迭代 Agent

    职责：
      1. 调用 LLM 评估方案有效性
      2. 对高价值案例自动沉淀到向量知识库
      3. 将验证结果持久化到 MySQL

    依赖注入：
      - vector_db: 向量数据库实例（可选，用于知识库写入）
      - mysql_crud: MySQL 数据库操作实例（可选，用于结果持久化）

    使用示例：
        agent = ClosedLoopAgent(vector_db, mysql_crud)
        result = await agent.execute(defect_id, semantic_label, root_cause, solution)
    """

    def __init__(
        self,
        vector_db: Any = None,
        mysql_crud: Any = None,
    ) -> None:
        """
        初始化闭环验证 Agent

        参数:
            vector_db: 向量数据库实例，用于知识库写入
            mysql_crud: MySQL 数据库操作实例，用于结果持久化
        """
        self._vector_db = vector_db
        self._mysql = mysql_crud

    async def execute(
        self,
        defect_id: str,
        semantic_label: SemanticLabel,
        root_cause_result: RootCauseResult,
        solution_result: SolutionResult,
        verification_data: Optional[Dict[str, Any]] = None,
    ) -> ClosedLoopResult:
        """
        执行闭环验证流程（主入口）

        流程：
          1. 调用 LLM 评估方案有效性
          2. 如果有效（评分 >= 0.7），自动沉淀到知识库
          3. 持久化验证结果到 MySQL

        参数:
            defect_id: 不良品唯一标识
            semantic_label: SemanticAgent 输出的语义标签
            root_cause_result: RootCauseAgent 输出的根因分析结果
            solution_result: SolutionAgent 输出的解决方案
            verification_data: 外部验证数据（如实施后的不良率变化等，可选）

        返回:
            ClosedLoopResult: 包含验证状态、有效性评分、知识库更新标志、反馈信息
        """
        logger.info("ClosedLoopAgent started | defect_id=%s", defect_id)

        try:
            # 步骤1: 调用 LLM 评估
            llm_result = await self._verify_with_llm(
                defect_id=defect_id,
                semantic_label=semantic_label,
                root_cause_result=root_cause_result,
                solution_result=solution_result,
                verification_data=verification_data,
            )

            # 步骤2: 组装验证结果
            result = ClosedLoopResult(
                defect_id=defect_id,
                verification_status=llm_result.get("verification_status", "pending"),
                effectiveness_score=float(llm_result.get("effectiveness_score", 0.0)),
                knowledge_updated=bool(llm_result.get("knowledge_updated", False)),
                feedback=llm_result.get("feedback", ""),
            )

            # 步骤3: 高价值案例自动沉淀到知识库
            if result.knowledge_updated and result.effectiveness_score >= 0.7:
                await self._update_knowledge_base(
                    defect_id=defect_id,
                    semantic_label=semantic_label,
                    root_cause_result=root_cause_result,
                    solution_result=solution_result,
                    result=result,
                )

            # 步骤4: 持久化验证结果
            await self._persist_closed_loop_result(result)

            logger.info(
                "ClosedLoopAgent completed | defect_id=%s | status=%s | score=%.2f | knowledge_updated=%s",
                defect_id,
                result.verification_status,
                result.effectiveness_score,
                result.knowledge_updated,
            )
            return result

        except Exception as exc:
            logger.error(
                "ClosedLoopAgent failed | defect_id=%s | error=%s",
                defect_id,
                str(exc),
            )
            raise

    async def _verify_with_llm(
        self,
        defect_id: str,
        semantic_label: SemanticLabel,
        root_cause_result: RootCauseResult,
        solution_result: SolutionResult,
        verification_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        调用 LLM 进行闭环验证评估

        组装包含不良现象、根因分析、解决方案、验证数据的完整 Prompt

        参数:
            defect_id: 不良品ID
            semantic_label: 语义标签
            root_cause_result: 根因分析结果
            solution_result: 解决方案
            verification_data: 外部验证数据

        返回:
            Dict[str, Any]: LLM 返回的 JSON 验证结果
        """
        # 格式化根因列表
        causes_text = "\n".join(
            [
                f"  - [{c.dimension}] {c.root_cause} (权重={c.weight:.2f})"
                for c in root_cause_result.causes
            ]
        )

        # 格式化方案列表
        solutions_text = "\n".join(
            [
                f"  - [{s.priority}] {s.action} (责任方: {s.responsible})"
                for s in solution_result.solutions
            ]
        )

        # 格式化验证数据
        verification_text = ""
        if verification_data:
            verification_text = f"\n\n【验证数据】:\n"
            for key, value in verification_data.items():
                verification_text += f"  - {key}: {value}\n"

        user_content = f"""请对以下涂装不良品分析结果进行闭环验证：

【不良现象】: {semantic_label.defect_type} ({semantic_label.severity})
【描述】: {semantic_label.description}

【根因分析】:
{causes_text}

【解决方案】:
{solutions_text}
{verification_text}"""

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return await llm_client.chat_json(
            messages=messages, agent_name="closed_loop_agent"
        )

    async def _update_knowledge_base(
        self,
        defect_id: str,
        semantic_label: SemanticLabel,
        root_cause_result: RootCauseResult,
        solution_result: SolutionResult,
        result: ClosedLoopResult,
    ) -> None:
        """
        将高价值案例沉淀到向量知识库

        沉淀内容包含：
          - 不良类型和严重程度
          - 根因分析摘要
          - 解决方案摘要
          - 有效性评分和反馈

        参数:
            defect_id: 不良品ID
            semantic_label: 语义标签
            root_cause_result: 根因分析结果
            solution_result: 解决方案
            result: 闭环验证结果
        """
        if not self._vector_db:
            logger.warning("Vector DB not configured, skipping knowledge update")
            return

        try:
            # 组装知识文档
            causes_summary = "; ".join(
                [f"[{c.dimension}]{c.root_cause}" for c in root_cause_result.causes]
            )
            solutions_summary = "; ".join(
                [s.action for s in solution_result.solutions]
            )

            document = (
                f"不良类型: {semantic_label.defect_type} | "
                f"严重程度: {semantic_label.severity} | "
                f"根因: {causes_summary} | "
                f"解决方案: {solutions_summary} | "
                f"有效性评分: {result.effectiveness_score:.2f} | "
                f"反馈: {result.feedback}"
            )

            # 元数据用于后续检索过滤
            metadata = {
                "defect_id": defect_id,
                "defect_type": semantic_label.defect_type,
                "severity": semantic_label.severity,
                "effectiveness_score": result.effectiveness_score,
            }

            await self._vector_db.add_document(
                doc_id=f"case_{defect_id}",
                content=document,
                metadata=metadata,
            )

            logger.info(
                "Knowledge base updated | defect_id=%s | type=%s",
                defect_id,
                semantic_label.defect_type,
            )

        except Exception as exc:
            logger.error(
                "Failed to update knowledge base | defect_id=%s | error=%s",
                defect_id,
                str(exc),
            )

    async def _persist_closed_loop_result(self, result: ClosedLoopResult) -> None:
        """
        将闭环验证结果持久化到 MySQL

        参数:
            result: 闭环验证结果对象
        """
        if not self._mysql:
            return
        try:
            await self._mysql.save_closed_loop_result(result)
        except Exception as exc:
            logger.error(
                "Failed to persist closed loop result | defect_id=%s | error=%s",
                result.defect_id,
                str(exc),
            )
