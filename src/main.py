"""
================================================================================
 汽车涂装不良品智能分析系统 - FastAPI 主入口
================================================================================
 功能说明：
   - 提供 RESTful API 接口，支持不良品全链路分析
   - 管理 5 个 Agent 的生命周期和依赖注入
   - 管理数据库连接池的初始化和优雅关闭
   - 提供 CORS 跨域支持，允许前端页面调用

 API 接口列表：
   GET  /health                  - 健康检查
   POST /analyze                 - 全链路分析（5个Agent串联执行）
   POST /analyze/root-cause      - 仅执行根因分析
   POST /analyze/semantic        - 仅执行语义解析
   POST /solutions/generate      - 仅生成解决方案
   POST /closed-loop/verify      - 闭环验证
   GET  /history/defects         - 查询历史不良品记录
   GET  /statistics/token-usage  - Token消耗统计
   GET  /knowledge/count         - 知识库文档数量

 新手提示：
   - 启动后访问 http://localhost:8000/docs 查看交互式API文档
   - 所有接口都可在 Swagger UI 中直接测试
   - 首次启动会自动初始化数据库表结构
================================================================================
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.agents import (
    ClosedLoopAgent,
    DataAgent,
    RootCauseAgent,
    SemanticAgent,
    SolutionAgent,
)
from src.connectors.mes_connector import MESConnector
from src.connectors.robot_connector import RobotConnector
from src.connectors.sensor_connector import SensorConnector
from src.db.influx_crud import InfluxCRUD
from src.db.mysql_crud import MySQLCRUD
from src.db.vector_db import VectorDB
from src.models.schemas import (
    ClosedLoopResult,
    DefectIDRequest,
    RootCauseResult,
    SemanticLabel,
    SolutionResult,
)
from src.utils.llm_client import llm_client

# ======================== 日志配置 ========================
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ======================== 全局服务实例（在 lifespan 中初始化） ========================
_mysql: Optional[MySQLCRUD] = None
"""MySQL 数据库操作实例"""

_influx: Optional[InfluxCRUD] = None
"""InfluxDB 时序数据库操作实例"""

_vector_db: Optional[VectorDB] = None
"""ChromaDB 向量数据库操作实例"""

_mes: Optional[MESConnector] = None
"""MES 系统连接器实例"""

_robot: Optional[RobotConnector] = None
"""安川机器人连接器实例"""

_sensor: Optional[SensorConnector] = None
"""环境传感器连接器实例"""


def _get_dependencies() -> Dict[str, Any]:
    """
    获取 Agent 所需的依赖注入字典

    将全局服务实例按 Agent 构造函数期望的参数名进行映射。
    各 Agent 的 __init__ 方法通过 **kwargs 接收这些依赖。

    返回:
        Dict[str, Any]: 包含 mysql, influx, vector_db, mes, robot, sensor 的字典
    """
    return {
        "mysql": _mysql,
        "influx": _influx,
        "vector_db": _vector_db,
        "mes": _mes,
        "robot": _robot,
        "sensor": _sensor,
    }


def _build_error_hint(error_msg: str) -> str:
    """
    根据错误信息生成新手友好的提示文本

    分析错误消息中的关键词，返回对应的排查建议。
    帮助新手用户快速定位和解决问题。

    参数:
        error_msg: 原始错误消息

    返回:
        str: 带提示的完整错误信息
    """
    msg_lower = error_msg.lower()
    if "api_key" in msg_lower or "unauthorized" in msg_lower or "auth" in msg_lower:
        return f"{error_msg} [提示] 请检查 .env 文件中的 API Key 是否正确配置"
    elif "connection" in msg_lower or "timeout" in msg_lower or "refused" in msg_lower:
        return f"{error_msg} [提示] 请检查网络连接或目标服务是否可用"
    elif "mysql" in msg_lower:
        return f"{error_msg} [提示] MySQL 未连接，系统将使用模拟数据运行，不影响分析功能"
    elif "chroma" in msg_lower or "vector" in msg_lower:
        return f"{error_msg} [提示] 向量数据库未连接，相似案例检索将返回空结果"
    return error_msg


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理器

    启动时（yield 之前）：
      1. 初始化所有数据库连接和外部系统连接器
      2. 设置 LLM Token 日志回调
      3. 自动创建数据库表结构

    关闭时（yield 之后）：
      1. 关闭所有外部连接
      2. 释放数据库连接池资源
    """
    global _mysql, _influx, _vector_db, _mes, _robot, _sensor

    logger.info("正在初始化系统组件...")

    # 初始化数据库和连接器
    _mysql = MySQLCRUD()
    _influx = InfluxCRUD()
    _vector_db = VectorDB()
    _mes = MESConnector()
    _robot = RobotConnector()
    _sensor = SensorConnector()

    # 设置 LLM Token 日志回调，将 Token 消耗记录写入 MySQL
    llm_client.set_token_log_callback(_mysql.save_llm_token_log)

    # 尝试初始化数据库表（失败不阻塞启动）
    try:
        await _mysql.init_tables()
        logger.info("数据库表初始化完成")
    except Exception as exc:
        logger.warning("数据库表初始化跳过（可能MySQL未启动）: %s", str(exc))

    logger.info("%s 启动完成 | debug=%s | llm_platform=%s",
                settings.APP_NAME, settings.DEBUG, settings.LLM_PLATFORM)
    yield

    # 优雅关闭：释放所有外部连接
    logger.info("正在关闭系统组件...")
    await _mes.close()
    await _robot.close()
    await _sensor.close()
    await _influx.close()
    logger.info("%s 已关闭", settings.APP_NAME)


# ======================== FastAPI 应用实例 ========================
app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="汽车涂装不良品智能分析与根因定位多 Agent 系统 - 通过5个协同Agent实现从数据采集到根因定位的全链路自动化分析",
    lifespan=lifespan,
)

# ======================== CORS 跨域配置 ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================== API 路由定义 ========================

@app.get("/")
async def root():
    """
    系统首页 - 提供新手引导信息

    返回系统简介、可用接口列表和快速上手指南。
    新手用户可通过此页面了解系统功能和使用方式。
    """
    return {
        "app": settings.APP_NAME,
        "version": "1.0.0",
        "description": "汽车涂装不良品智能分析与根因定位多Agent系统",
        "llm_platform": settings.LLM_PLATFORM,
        "新手快速上手": {
            "步骤1": "访问 /docs 查看交互式 API 文档（Swagger UI）",
            "步骤2": "使用 POST /analyze 接口提交不良品ID进行全链路分析",
            "步骤3": "查看返回的根因分析结果和解决方案",
            "步骤4": "使用 POST /closed-loop/verify 进行闭环验证",
        },
        "核心接口": {
            "GET  /health": "健康检查 - 确认服务是否正常运行",
            "POST /analyze": "全链路分析 - 5个Agent串联执行完整分析流程",
            "POST /analyze/root-cause": "根因分析 - 仅定位问题根因",
            "POST /analyze/semantic": "语义解析 - 标准化不良品描述",
            "POST /solutions/generate": "方案生成 - 自动生成纠正措施",
            "POST /closed-loop/verify": "闭环验证 - 评估方案有效性并沉淀知识",
            "GET  /history/defects": "历史查询 - 查看历史不良品记录",
            "GET  /statistics/token-usage": "Token统计 - 监控LLM调用成本",
            "GET  /knowledge/count": "知识库 - 查看已沉淀案例数量",
        },
        "配置提示": {
            "LLM平台": f"当前使用: {settings.LLM_PLATFORM}，可在 .env 中修改 LLM_PLATFORM 切换平台",
            "API Key": "请在 .env 文件中配置 DASHSCOPE_API_KEY 或 DIFY_API_KEY",
            "数据库": "MySQL/InfluxDB/ChromaDB 未连接时系统自动使用模拟数据运行",
        },
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    健康检查接口

    用于监控系统是否正常运行，可被负载均衡器或监控系统定期调用。

    返回:
        - status: "healthy" 表示服务正常
        - app: 应用名称
        - timestamp: 当前时间戳
    """
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/analyze", response_model=Dict[str, Any])
async def analyze_defect(request: DefectIDRequest, background_tasks: BackgroundTasks):
    """
    全链路不良品分析接口（核心接口）

    依次执行 5 个 Agent 完成完整的分析流程：
      1. DataAgent - 数据采集与预处理
      2. SemanticAgent - 不良现象语义解析
      3. RootCauseAgent - 多维度根因推理
      4. SolutionAgent - 解决方案生成
      5. ClosedLoopAgent - 闭环验证与知识库迭代

    每个阶段的中间结果都会持久化到 MySQL 数据库。

    请求示例:
        POST /analyze
        {"defect_id": "DEF-2024-001"}

    返回:
        包含语义标签、根因列表、推理链路、解决方案、闭环验证结果的完整JSON
    """
    defect_id = request.defect_id
    logger.info("收到全链路分析请求 | defect_id=%s", defect_id)

    try:
        # 创建各 Agent 实例并注入依赖
        data_agent = DataAgent(**_get_dependencies())
        semantic_agent = SemanticAgent(_vector_db)
        root_cause_agent = RootCauseAgent(_vector_db)
        solution_agent = SolutionAgent(_vector_db)
        closed_loop_agent = ClosedLoopAgent(_vector_db, _mysql)

        # 阶段1: 数据采集
        preprocessed = await data_agent.execute(defect_id)
        await _mysql.save_preprocessed_data(preprocessed)

        # 阶段2: 语义解析
        semantic_label = await semantic_agent.execute(preprocessed)
        await _mysql.save_semantic_label(defect_id, semantic_label)

        # 阶段3: 根因推理
        root_cause_result = await root_cause_agent.execute(preprocessed, semantic_label)
        await _mysql.save_root_cause_result(root_cause_result)

        # 阶段4: 解决方案生成
        solution_result = await solution_agent.execute(preprocessed, semantic_label, root_cause_result)
        await _mysql.save_solution_result(solution_result)

        # 阶段5: 闭环验证
        closed_loop_result = await closed_loop_agent.execute(
            defect_id, semantic_label, root_cause_result, solution_result
        )

        logger.info("全链路分析完成 | defect_id=%s", defect_id)

        return {
            "defect_id": defect_id,
            "semantic_label": semantic_label.model_dump(),
            "root_causes": [c.model_dump() for c in root_cause_result.causes],
            "reasoning_chain": root_cause_result.reasoning_chain,
            "solutions": [s.model_dump() for s in solution_result.solutions],
            "closed_loop": closed_loop_result.model_dump(),
            "analyzed_at": datetime.now().isoformat(),
        }

    except Exception as exc:
        logger.error("全链路分析失败 | defect_id=%s | error=%s", defect_id, str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"分析失败: {_build_error_hint(str(exc))}"
        ) from exc


@app.post("/analyze/root-cause", response_model=RootCauseResult)
async def analyze_root_cause(request: DefectIDRequest):
    """
    根因分析接口

    仅执行数据采集、语义解析和根因推理三个阶段，
    适用于只需要定位问题根因而无需生成解决方案的场景。

    请求示例:
        POST /analyze/root-cause
        {"defect_id": "DEF-2024-001"}
    """
    defect_id = request.defect_id
    logger.info("收到根因分析请求 | defect_id=%s", defect_id)

    try:
        data_agent = DataAgent(**_get_dependencies())
        semantic_agent = SemanticAgent(_vector_db)
        root_cause_agent = RootCauseAgent(_vector_db)

        preprocessed = await data_agent.execute(defect_id)
        semantic_label = await semantic_agent.execute(preprocessed)
        root_cause_result = await root_cause_agent.execute(preprocessed, semantic_label)

        return root_cause_result

    except Exception as exc:
        logger.error("根因分析失败 | defect_id=%s | error=%s", defect_id, str(exc))
        raise HTTPException(status_code=500, detail=f"根因分析失败: {_build_error_hint(str(exc))}") from exc


@app.post("/analyze/semantic", response_model=SemanticLabel)
async def analyze_semantic(
    request: DefectIDRequest,
    defect_description: str = "",
    image_path: Optional[str] = None,
):
    """
    语义解析接口

    对不良品描述文本和缺陷图片进行语义解析，
    输出标准化的不良类型标签和严重程度评估。

    参数:
        defect_description: 不良品文字描述（可选）
        image_path: 缺陷图片文件路径（可选，支持OCR识别）

    请求示例:
        POST /analyze/semantic?defect_description=表面出现橘皮状纹理
        {"defect_id": "DEF-2024-001"}
    """
    defect_id = request.defect_id
    try:
        data_agent = DataAgent(**_get_dependencies())
        semantic_agent = SemanticAgent(_vector_db)

        preprocessed = await data_agent.execute(defect_id)
        semantic_label = await semantic_agent.execute(preprocessed, defect_description, image_path)

        return semantic_label

    except Exception as exc:
        logger.error("语义解析失败 | defect_id=%s | error=%s", defect_id, str(exc))
        raise HTTPException(status_code=500, detail=f"语义解析失败: {_build_error_hint(str(exc))}") from exc


@app.post("/solutions/generate", response_model=SolutionResult)
async def generate_solutions(request: DefectIDRequest):
    """
    解决方案生成接口

    基于根因分析结果，自动生成针对性的纠正措施和预防方案。
    方案包含责任方、优先级和预估工作量。

    请求示例:
        POST /solutions/generate
        {"defect_id": "DEF-2024-001"}
    """
    defect_id = request.defect_id
    logger.info("收到方案生成请求 | defect_id=%s", defect_id)

    try:
        data_agent = DataAgent(**_get_dependencies())
        semantic_agent = SemanticAgent(_vector_db)
        root_cause_agent = RootCauseAgent(_vector_db)
        solution_agent = SolutionAgent(_vector_db)

        preprocessed = await data_agent.execute(defect_id)
        semantic_label = await semantic_agent.execute(preprocessed)
        root_cause_result = await root_cause_agent.execute(preprocessed, semantic_label)
        solution_result = await solution_agent.execute(preprocessed, semantic_label, root_cause_result)

        return solution_result

    except Exception as exc:
        logger.error("方案生成失败 | defect_id=%s | error=%s", defect_id, str(exc))
        raise HTTPException(status_code=500, detail=f"方案生成失败: {_build_error_hint(str(exc))}") from exc


@app.post("/closed-loop/verify", response_model=ClosedLoopResult)
async def verify_closed_loop(
    defect_id: str,
    verification_data: Optional[Dict[str, Any]] = None,
):
    """
    闭环验证接口

    评估已实施解决方案的有效性，决定是否将分析经验沉淀到知识库。
    有效性评分 >= 0.7 的案例会自动写入向量数据库供后续参考。

    参数:
        defect_id: 不良品ID
        verification_data: 验证数据（如实施后的不良率变化等）

    请求示例:
        POST /closed-loop/verify?defect_id=DEF-2024-001
    """
    logger.info("收到闭环验证请求 | defect_id=%s", defect_id)

    try:
        closed_loop_agent = ClosedLoopAgent(_vector_db, _mysql)

        result = await closed_loop_agent.execute(
            defect_id=defect_id,
            semantic_label=SemanticLabel(
                defect_type="未知", defect_category="未知", severity="medium"
            ),
            root_cause_result=RootCauseResult(
                defect_id=defect_id, causes=[], reasoning_chain=""
            ),
            solution_result=SolutionResult(defect_id=defect_id, solutions=[]),
            verification_data=verification_data,
        )

        return result

    except Exception as exc:
        logger.error("闭环验证失败 | defect_id=%s | error=%s", defect_id, str(exc))
        raise HTTPException(status_code=500, detail=f"闭环验证失败: {_build_error_hint(str(exc))}") from exc


@app.get("/history/defects")
async def get_defect_history(
    defect_type: Optional[str] = None,
    line_id: Optional[str] = None,
    limit: int = 50,
):
    """
    历史不良品查询接口

    支持按不良类型和产线编号筛选历史记录。

    参数:
        defect_type: 不良类型筛选（可选，如"缩孔"）
        line_id: 产线编号筛选（可选，如"LINE-01"）
        limit: 返回记录数上限（默认50）

    请求示例:
        GET /history/defects?defect_type=缩孔&limit=20
    """
    try:
        history = await _mysql.get_defect_history(
            defect_type=defect_type,
            line_id=line_id,
            limit=limit,
        )
        return {"history": history, "count": len(history)}

    except Exception as exc:
        logger.error("历史查询失败: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"历史查询失败: {_build_error_hint(str(exc))}") from exc


@app.get("/statistics/token-usage")
async def get_token_usage(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    """
    Token 消耗统计接口

    查询指定时间范围内各 Agent 的 Token 消耗情况，
    用于成本监控和优化。

    参数:
        start_date: 开始日期（ISO格式，如"2024-01-01"）
        end_date: 结束日期（ISO格式，如"2024-01-31"）

    请求示例:
        GET /statistics/token-usage?start_date=2024-01-01&end_date=2024-01-31
    """
    try:
        start = datetime.fromisoformat(start_date) if start_date else None
        end = datetime.fromisoformat(end_date) if end_date else None

        summary = await _mysql.get_token_usage_summary(start_date=start, end_date=end)
        return summary

    except Exception as exc:
        logger.error("Token统计失败: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"Token统计失败: {_build_error_hint(str(exc))}") from exc


@app.get("/knowledge/count")
async def get_knowledge_count():
    """
    知识库文档数量查询接口

    返回向量数据库中已沉淀的工艺知识案例数量。

    请求示例:
        GET /knowledge/count
    """
    try:
        count = await _vector_db.count_documents()
        return {"document_count": count}
    except Exception as exc:
        logger.error("知识库查询失败: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"知识库查询失败: {_build_error_hint(str(exc))}") from exc


# ======================== 直接运行入口 ========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
