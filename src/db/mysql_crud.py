"""
================================================================================
 汽车涂装不良品智能分析系统 - MySQL 数据库操作模块
================================================================================
 功能说明：
   - 管理 MySQL 异步连接池的创建与复用
   - 提供事务安全的数据库会话上下文管理器
   - 自动创建系统所需的 6 张核心数据表
   - 封装不良品分析全流程数据的增删改查操作
   - 支持 Token 消耗统计与历史记录查询

 数据表说明：
   preprocessed_data    - 预处理后的原始数据（MES+机器人+传感器）
   semantic_labels      - 不良现象语义解析标签
   root_cause_results   - 多维度根因推理结果
   solution_results     - 纠正措施与预防方案
   closed_loop_results  - 闭环验证与有效性评估
   llm_token_logs       - 大模型调用 Token 消耗日志

 新手提示：
   - 表结构会在首次启动时自动创建，无需手动建表
   - 所有写操作使用 ON DUPLICATE KEY UPDATE 实现幂等写入
   - MySQL 未连接时系统仍可使用模拟数据运行
================================================================================
"""

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from sqlalchemy.pool import NullPool

from src.config import settings
from src.models.schemas import (
    ClosedLoopResult,
    LLMTokenLog,
    PreprocessedData,
)

logger = logging.getLogger(__name__)

_engine = None


def get_engine():
    """
    获取 MySQL 异步引擎实例（懒加载单例模式）

    首次调用时根据配置创建连接池，后续调用直接返回已有实例。
    连接池参数：
      - pool_size: 常驻连接数（默认10）
      - max_overflow: 峰值额外连接数（默认20）
      - pool_pre_ping: 每次使用前检测连接有效性

    返回:
        sqlalchemy AsyncEngine 实例
    """
    global _engine
    if _engine is None:
        connection_url = (
            f"mysql+aiomysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}"
            f"@{settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}"
        )
        _engine = create_async_engine(
            connection_url,
            pool_size=settings.MYSQL_POOL_SIZE,
            max_overflow=settings.MYSQL_MAX_OVERFLOW,
            pool_pre_ping=True,
            echo=settings.DEBUG,
        )
    return _engine


class MySQLCRUD:
    """
    MySQL 数据库增删改查操作封装

    职责：
      1. 管理数据库连接生命周期
      2. 提供事务安全的会话管理
      3. 封装 6 张核心表的 CRUD 操作
      4. 支持历史查询和统计分析

    使用示例：
        mysql = MySQLCRUD()
        await mysql.init_tables()
        await mysql.save_preprocessed_data(data)
    """

    def __init__(self) -> None:
        """初始化 MySQL CRUD 实例，获取数据库引擎"""
        self._engine = get_engine()

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncConnection]:
        """
        获取事务安全的数据库会话上下文管理器

        使用 async with 语法自动管理事务的提交和回滚：
          async with mysql.session() as conn:
              await conn.execute(...)

        退出上下文时自动提交事务，异常时自动回滚。
        """
        async with self._engine.connect() as conn:
            async with conn.begin():
                yield conn

    async def init_tables(self) -> None:
        create_tables_sql = """
        CREATE TABLE IF NOT EXISTS preprocessed_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            defect_id VARCHAR(64) UNIQUE NOT NULL,
            product_model VARCHAR(128),
            line_id VARCHAR(64),
            timestamp DATETIME,
            work_order VARCHAR(64),
            shift VARCHAR(32),
            operator_id VARCHAR(64),
            robot_params_json TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_defect_id (defect_id),
            INDEX idx_line_id (line_id),
            INDEX idx_timestamp (timestamp)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        CREATE TABLE IF NOT EXISTS semantic_labels (
            id INT AUTO_INCREMENT PRIMARY KEY,
            defect_id VARCHAR(64) UNIQUE NOT NULL,
            defect_type VARCHAR(64),
            defect_category VARCHAR(64),
            severity VARCHAR(16),
            description TEXT,
            ocr_text TEXT,
            confidence FLOAT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_defect_id (defect_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        CREATE TABLE IF NOT EXISTS root_cause_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            defect_id VARCHAR(64) UNIQUE NOT NULL,
            causes_json TEXT,
            reasoning_chain TEXT,
            analyzed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_defect_id (defect_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        CREATE TABLE IF NOT EXISTS solution_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            defect_id VARCHAR(64) UNIQUE NOT NULL,
            solutions_json TEXT,
            generated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_defect_id (defect_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        CREATE TABLE IF NOT EXISTS closed_loop_results (
            id INT AUTO_INCREMENT PRIMARY KEY,
            defect_id VARCHAR(64) UNIQUE NOT NULL,
            verification_status VARCHAR(32),
            effectiveness_score FLOAT,
            knowledge_updated TINYINT(1),
            feedback TEXT,
            verified_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_defect_id (defect_id),
            INDEX idx_verification_status (verification_status)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

        CREATE TABLE IF NOT EXISTS llm_token_logs (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            request_id VARCHAR(64) UNIQUE NOT NULL,
            agent_name VARCHAR(64),
            model VARCHAR(64),
            prompt_tokens INT,
            completion_tokens INT,
            total_tokens INT,
            latency_ms FLOAT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_agent_name (agent_name),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        try:
            async with self._engine.begin() as conn:
                for statement in create_tables_sql.strip().split(";"):
                    stmt = statement.strip()
                    if stmt:
                        await conn.execute(text(stmt))
            logger.info("MySQL tables initialized")
        except Exception as exc:
            logger.error("Failed to initialize MySQL tables: %s", str(exc))
            raise

    async def save_preprocessed_data(self, data: PreprocessedData) -> None:
        """
        保存预处理后的原始数据

        将 MES 产品信息、机器人工艺参数等写入 preprocessed_data 表。
        使用 ON DUPLICATE KEY UPDATE 确保同一 defect_id 重复写入时更新而非报错。

        参数:
            data: PreprocessedData 对象，包含产品信息和工艺参数
        """
        try:
            async with self.session() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO preprocessed_data
                        (defect_id, product_model, line_id, timestamp, work_order, shift, operator_id, robot_params_json)
                        VALUES (:defect_id, :product_model, :line_id, :timestamp, :work_order, :shift, :operator_id, :robot_params_json)
                        ON DUPLICATE KEY UPDATE
                            product_model = VALUES(product_model),
                            line_id = VALUES(line_id),
                            timestamp = VALUES(timestamp)
                    """),
                    {
                        "defect_id": data.defect_id,
                        "product_model": data.product_info.product_model,
                        "line_id": data.product_info.line_id,
                        "timestamp": data.product_info.timestamp,
                        "work_order": data.product_info.work_order,
                        "shift": data.product_info.shift,
                        "operator_id": data.product_info.operator_id,
                        "robot_params_json": json.dumps(data.robot_params.raw_params, default=str, ensure_ascii=False),
                    },
                )
            logger.info("Preprocessed data saved | defect_id=%s", data.defect_id)
        except Exception as exc:
            logger.error("Failed to save preprocessed data | defect_id=%s | error=%s", data.defect_id, str(exc))
            raise

    async def save_semantic_label(self, defect_id: str, label: Any) -> None:
        """
        保存不良现象语义解析标签

        将 SemanticAgent 解析出的不良类型、严重程度、置信度等信息写入 semantic_labels 表。

        参数:
            defect_id: 不良品唯一标识
            label: SemanticLabel 对象，包含不良类型和严重程度
        """
        try:
            async with self.session() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO semantic_labels
                        (defect_id, defect_type, defect_category, severity, description, ocr_text, confidence)
                        VALUES (:defect_id, :defect_type, :defect_category, :severity, :description, :ocr_text, :confidence)
                        ON DUPLICATE KEY UPDATE
                            defect_type = VALUES(defect_type),
                            severity = VALUES(severity)
                    """),
                    {
                        "defect_id": defect_id,
                        "defect_type": label.defect_type,
                        "defect_category": label.defect_category,
                        "severity": label.severity,
                        "description": label.description,
                        "ocr_text": label.ocr_text or "",
                        "confidence": label.confidence,
                    },
                )
        except Exception as exc:
            logger.error("Failed to save semantic label | defect_id=%s | error=%s", defect_id, str(exc))
            raise

    async def save_root_cause_result(self, result: Any) -> None:
        """
        保存多维度根因推理结果

        将 RootCauseAgent 推理出的根因列表（Top3）和推理链路写入 root_cause_results 表。
        根因列表以 JSON 格式存储，包含维度、权重、证据等字段。

        参数:
            result: RootCauseResult 对象，包含根因列表和推理链路
        """
        try:
            async with self.session() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO root_cause_results (defect_id, causes_json, reasoning_chain)
                        VALUES (:defect_id, :causes_json, :reasoning_chain)
                        ON DUPLICATE KEY UPDATE causes_json = VALUES(causes_json), reasoning_chain = VALUES(reasoning_chain)
                    """),
                    {
                        "defect_id": result.defect_id,
                        "causes_json": json.dumps([c.model_dump() for c in result.causes], ensure_ascii=False),
                        "reasoning_chain": result.reasoning_chain,
                    },
                )
        except Exception as exc:
            logger.error("Failed to save root cause result | defect_id=%s | error=%s", result.defect_id, str(exc))
            raise

    async def save_solution_result(self, result: Any) -> None:
        """
        保存解决方案生成结果

        将 SolutionAgent 生成的纠正措施和预防方案写入 solution_results 表。
        方案以 JSON 数组格式存储，包含责任方、优先级、预估工作量。

        参数:
            result: SolutionResult 对象，包含方案列表
        """
        try:
            async with self.session() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO solution_results (defect_id, solutions_json)
                        VALUES (:defect_id, :solutions_json)
                        ON DUPLICATE KEY UPDATE solutions_json = VALUES(solutions_json)
                    """),
                    {
                        "defect_id": result.defect_id,
                        "solutions_json": json.dumps([s.model_dump() for s in result.solutions], ensure_ascii=False),
                    },
                )
        except Exception as exc:
            logger.error("Failed to save solution result | defect_id=%s | error=%s", result.defect_id, str(exc))
            raise

    async def save_closed_loop_result(self, result: ClosedLoopResult) -> None:
        """
        保存闭环验证结果

        将 ClosedLoopAgent 的验证状态、有效性评分、知识库更新标记写入 closed_loop_results 表。

        参数:
            result: ClosedLoopResult 对象，包含验证状态和有效性评分
        """
        try:
            async with self.session() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO closed_loop_results
                        (defect_id, verification_status, effectiveness_score, knowledge_updated, feedback)
                        VALUES (:defect_id, :verification_status, :effectiveness_score, :knowledge_updated, :feedback)
                        ON DUPLICATE KEY UPDATE
                            verification_status = VALUES(verification_status),
                            effectiveness_score = VALUES(effectiveness_score)
                    """),
                    {
                        "defect_id": result.defect_id,
                        "verification_status": result.verification_status,
                        "effectiveness_score": result.effectiveness_score,
                        "knowledge_updated": int(result.knowledge_updated),
                        "feedback": result.feedback,
                    },
                )
        except Exception as exc:
            logger.error("Failed to save closed loop result | defect_id=%s | error=%s", result.defect_id, str(exc))
            raise

    async def save_llm_token_log(self, token_log: LLMTokenLog) -> None:
        """
        保存大模型调用 Token 消耗日志

        记录每次 LLM 调用的 Token 使用量和响应延迟，用于成本监控和性能优化。
        该方法通常由 LLMClient 在每次调用完成后自动触发。

        参数:
            token_log: LLMTokenLog 对象，包含 Token 消耗和延迟信息
        """
        try:
            async with self.session() as conn:
                await conn.execute(
                    text("""
                        INSERT INTO llm_token_logs
                        (request_id, agent_name, model, prompt_tokens, completion_tokens, total_tokens, latency_ms)
                        VALUES (:request_id, :agent_name, :model, :prompt_tokens, :completion_tokens, :total_tokens, :latency_ms)
                    """),
                    {
                        "request_id": token_log.request_id,
                        "agent_name": token_log.agent_name,
                        "model": token_log.model,
                        "prompt_tokens": token_log.prompt_tokens,
                        "completion_tokens": token_log.completion_tokens,
                        "total_tokens": token_log.total_tokens,
                        "latency_ms": token_log.latency_ms,
                    },
                )
        except Exception as exc:
            logger.warning("Failed to save LLM token log | request_id=%s | error=%s", token_log.request_id, str(exc))

    async def get_defect_history(
        self,
        defect_type: Optional[str] = None,
        line_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        查询历史不良品记录

        支持按不良类型和产线编号组合筛选，结果按时间倒序排列。
        通过 LEFT JOIN 关联预处理数据、语义标签和根因结果三张表。

        参数:
            defect_type: 不良类型筛选（可选，如"缩孔"）
            line_id: 产线编号筛选（可选，如"LINE-01"）
            limit: 返回记录数上限（默认50）

        返回:
            List[Dict]: 包含 defect_id, product_model, line_id, timestamp,
                       defect_type, severity, reasoning_chain 的历史记录列表
        """
        try:
            conditions = []
            params: Dict[str, Any] = {"limit": limit}
            if defect_type:
                conditions.append("sl.defect_type = :defect_type")
                params["defect_type"] = defect_type
            if line_id:
                conditions.append("pd.line_id = :line_id")
                params["line_id"] = line_id

            where_clause = " AND ".join(conditions) if conditions else "1=1"

            sql = f"""
                SELECT pd.defect_id, pd.product_model, pd.line_id, pd.timestamp,
                       sl.defect_type, sl.severity, rcr.reasoning_chain
                FROM preprocessed_data pd
                LEFT JOIN semantic_labels sl ON pd.defect_id = sl.defect_id
                LEFT JOIN root_cause_results rcr ON pd.defect_id = rcr.defect_id
                WHERE {where_clause}
                ORDER BY pd.timestamp DESC
                LIMIT :limit
            """

            async with self.session() as conn:
                result = await conn.execute(text(sql), params)
                rows = result.fetchall()
                return [dict(row._mapping) for row in rows]

        except Exception as exc:
            logger.error("Failed to fetch defect history: %s", str(exc))
            return []

    async def get_token_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        查询 Token 消耗统计汇总

        按 Agent 名称分组统计指定时间范围内的 Token 消耗情况，
        包括请求次数、Prompt/Completion Token 总量和平均响应延迟。

        参数:
            start_date: 统计开始日期（可选）
            end_date: 统计结束日期（可选）

        返回:
            Dict: 包含 agents 列表，每项含 agent_name, request_count,
                  total_prompt_tokens, total_completion_tokens, total_tokens, avg_latency_ms
        """
        try:
            date_filter = ""
            params: Dict[str, Any] = {}
            if start_date:
                date_filter += " AND created_at >= :start_date"
                params["start_date"] = start_date
            if end_date:
                date_filter += " AND created_at <= :end_date"
                params["end_date"] = end_date

            sql = f"""
                SELECT
                    agent_name,
                    COUNT(*) as request_count,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    AVG(latency_ms) as avg_latency_ms
                FROM llm_token_logs
                WHERE 1=1 {date_filter}
                GROUP BY agent_name
            """

            async with self.session() as conn:
                result = await conn.execute(text(sql), params)
                rows = result.fetchall()
                return {"agents": [dict(row._mapping) for row in rows]}

        except Exception as exc:
            logger.error("Failed to fetch token usage summary: %s", str(exc))
            return {"agents": []}
