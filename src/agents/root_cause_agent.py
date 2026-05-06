"""
================================================================================
 汽车涂装不良品智能分析系统 - 多维度根因推理 Agent
================================================================================
 功能说明：
   - 基于"人-机-料-法-环-测"六维度框架进行系统根因推理
   - 自动检测工艺参数和环境参数的 SOP 违规项
   - 从向量知识库检索相似历史案例辅助推理
   - 调用 LLM 进行深度根因分析，输出 Top 3 根因

 SOP 标准范围（工艺参数）：
   喷涂电压: 50-80 kV
   雾化压力: 0.15-0.35 MPa
   枪距: 200-300 mm
   流量: 200-500 cc/min
   机器人速度: 600-1200 mm/s
   重叠率: 40-60%
   成型气压: 0.10-0.25 MPa
   触发延迟: 20-100 ms

 SOP 标准范围（环境参数）：
   温度: 20-28°C
   湿度: 40-65%
   压差: 5-20 Pa

 六维度分析框架：
   人(人员): 操作员是否按SOP操作？培训是否到位？
   机(设备): 设备参数是否在SOP范围内？是否需要维护？
   料(物料): 涂料批次是否合格？稀释比例是否正确？
   法(工艺): 工艺参数设置是否合理？喷涂路径是否优化？
   环(环境): 温湿度是否在控制范围？洁净度是否达标？
   测(检测): 检测方法是否正确？设备是否校准？

 新手提示：
   - 本 Agent 是系统的核心推理引擎
   - SOP 违规检测是自动的，无需手动配置
   - 根因按权重从高到低排序，权重最高的最可能是真正根因
================================================================================
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.models.schemas import (
    PreprocessedData,
    RootCauseItem,
    RootCauseResult,
    SemanticLabel,
)
from src.utils.llm_client import llm_client

logger = logging.getLogger(__name__)

# ======================== 工艺参数 SOP 标准范围 ========================
SOP_RANGES: Dict[str, Dict[str, Any]] = {
    "voltage": {"min": 50.0, "max": 80.0, "unit": "kV", "name": "喷涂电压"},
    "atomization_pressure": {"min": 0.15, "max": 0.35, "unit": "MPa", "name": "雾化压力"},
    "gun_distance": {"min": 200.0, "max": 300.0, "unit": "mm", "name": "枪距"},
    "flow_rate": {"min": 200.0, "max": 500.0, "unit": "cc/min", "name": "流量"},
    "robot_speed": {"min": 600.0, "max": 1200.0, "unit": "mm/s", "name": "机器人速度"},
    "overlap_rate": {"min": 40.0, "max": 60.0, "unit": "%", "name": "重叠率"},
    "shaping_air": {"min": 0.10, "max": 0.25, "unit": "MPa", "name": "成型气压"},
    "trigger_delay": {"min": 20.0, "max": 100.0, "unit": "ms", "name": "触发延迟"},
}
"""工艺参数 SOP 标准范围：min=下限, max=上限, unit=单位, name=中文名称"""

# ======================== 环境参数 SOP 标准范围 ========================
ENV_SOP_RANGES: Dict[str, Dict[str, Any]] = {
    "temperature": {"min": 20.0, "max": 28.0, "unit": "°C", "name": "温度"},
    "humidity": {"min": 40.0, "max": 65.0, "unit": "%", "name": "湿度"},
    "pressure_diff": {"min": 5.0, "max": 20.0, "unit": "Pa", "name": "压差"},
}
"""环境参数 SOP 标准范围"""

# ======================== LLM System Prompt ========================
SYSTEM_PROMPT = """你是一名拥有20年经验的资深汽车涂装工艺专家。你的任务是对涂装不良品进行多维度根因推理分析。

你必须严格按照"人-机-料-法-环-测"六维度分析框架进行系统推理：

1. **人(人员)**: 操作员是否按SOP操作？是否存在误操作？培训是否到位？
2. **机(设备)**: 喷涂设备参数是否在SOP范围内？设备是否需要维护？机器人运动是否异常？
3. **料(物料)**: 涂料批次是否合格？稀释比例是否正确？涂料是否过期？
4. **法(工艺)**: 工艺参数设置是否合理？喷涂路径是否优化？节拍是否匹配？
5. **环(环境)**: 温湿度是否在控制范围？洁净度是否达标？压差是否正常？
6. **测(检测)**: 检测方法是否正确？检测设备是否校准？判定标准是否一致？

关键规则：
- 必须逐一检查每个工艺参数是否在SOP标准范围内
- 对超出SOP范围的参数，必须明确指出偏差值和可能影响
- 对环境参数（温湿度、压差）同样需检查是否越限
- 权重分配需基于实际数据证据，而非主观臆断
- 所有根因必须有对应的数据证据支撑

请严格按照以下JSON格式输出：
{
    "causes": [
        {
            "root_cause": "根因描述(需具体、可操作)",
            "dimension": "维度(人/机/料/法/环/测)",
            "weight": 0.35,
            "evidence": "数据证据(引用具体参数值和SOP范围)",
            "sop_reference": "SOP参考条款(如有)"
        }
    ],
    "reasoning_chain": "推理链路摘要(描述从数据到结论的推理过程)"
}

要求：
- causes 数组按 weight 从高到低排序
- 最多输出 Top 3 根因
- weight 之和不超过 1.0
- 每个根因的 evidence 必须引用具体数据"""


class RootCauseAgent:
    """
    多维度根因推理 Agent

    职责：
      1. 自动检测工艺参数和环境参数的 SOP 违规项
      2. 从向量知识库检索相似历史案例
      3. 调用 LLM 进行六维度根因推理
      4. 结合 SOP 违规检测结果调整根因权重

    依赖注入：
      - vector_db: 向量数据库实例（可选，用于检索历史案例）

    使用示例：
        agent = RootCauseAgent(vector_db)
        result = await agent.execute(preprocessed_data, semantic_label)
    """

    def __init__(self, vector_db: Any = None) -> None:
        """
        初始化根因推理 Agent

        参数:
            vector_db: 向量数据库实例
        """
        self._vector_db = vector_db

    async def execute(
        self,
        preprocessed_data: PreprocessedData,
        semantic_label: SemanticLabel,
    ) -> RootCauseResult:
        """
        执行根因推理流程（主入口）

        流程：
          1. 检测 SOP 违规项（工艺参数 + 环境参数）
          2. 检索相似历史案例
          3. 调用 LLM 进行六维度推理
          4. 解析 LLM 返回的根因列表
          5. 结合 SOP 违规检测结果调整权重
          6. 取 Top 3 根因输出

        参数:
            preprocessed_data: DataAgent 输出的预处理数据
            semantic_label: SemanticAgent 输出的语义标签

        返回:
            RootCauseResult: 包含 Top 3 根因和推理链路的分析结果
        """
        logger.info(
            "RootCauseAgent started | defect_id=%s | defect_type=%s",
            preprocessed_data.defect_id,
            semantic_label.defect_type,
        )

        try:
            # 步骤1: 检测 SOP 违规项
            sop_violations = self._check_sop_violations(preprocessed_data)

            # 步骤2: 检索相似案例
            knowledge_context = await self._retrieve_similar_cases(
                semantic_label.defect_type
            )

            # 步骤3: 调用 LLM 推理
            llm_result = await self._reason_with_llm(
                preprocessed_data=preprocessed_data,
                semantic_label=semantic_label,
                sop_violations=sop_violations,
                knowledge_context=knowledge_context,
            )

            # 步骤4: 解析根因列表
            causes = self._parse_causes(llm_result)

            # 步骤5: 结合 SOP 违规调整权重
            if sop_violations and causes:
                causes = self._adjust_weights_with_sop(causes, sop_violations)

            # 步骤6: 组装结果
            result = RootCauseResult(
                defect_id=preprocessed_data.defect_id,
                causes=causes[:3],
                reasoning_chain=llm_result.get("reasoning_chain", ""),
            )

            logger.info(
                "RootCauseAgent completed | defect_id=%s | top_cause=%s | top_weight=%.2f",
                preprocessed_data.defect_id,
                result.causes[0].root_cause if result.causes else "N/A",
                result.causes[0].weight if result.causes else 0.0,
            )
            return result

        except Exception as exc:
            logger.error(
                "RootCauseAgent failed | defect_id=%s | error=%s",
                preprocessed_data.defect_id,
                str(exc),
            )
            raise

    def _check_sop_violations(
        self, data: PreprocessedData
    ) -> List[Dict[str, Any]]:
        """
        检测 SOP 违规项

        检查内容：
          1. 8 项关键工艺参数是否在 SOP 范围内
          2. 3 项环境参数（温度/湿度/压差）均值是否在 SOP 范围内

        参数:
            data: 预处理后的完整数据

        返回:
            List[Dict]: SOP 违规项列表，每项包含参数名、实际值、SOP范围、偏差方向、维度
        """
        violations: List[Dict[str, Any]] = []

        # 检查工艺参数
        params = data.robot_params
        for param_key, sop in SOP_RANGES.items():
            value = getattr(params, param_key, None)
            if value is None:
                continue
            if value < sop["min"] or value > sop["max"]:
                violations.append(
                    {
                        "param": sop["name"],
                        "key": param_key,
                        "value": value,
                        "sop_min": sop["min"],
                        "sop_max": sop["max"],
                        "unit": sop["unit"],
                        "deviation": "偏低" if value < sop["min"] else "偏高",
                        "dimension": "机",
                    }
                )

        # 检查环境参数均值
        sensor = data.sensor_data
        if sensor.temperature:
            avg_temp = sum(sensor.temperature) / len(sensor.temperature)
            self._check_env_violation(
                violations, "温度", avg_temp, ENV_SOP_RANGES["temperature"], "环"
            )
        if sensor.humidity:
            avg_humidity = sum(sensor.humidity) / len(sensor.humidity)
            self._check_env_violation(
                violations, "湿度", avg_humidity, ENV_SOP_RANGES["humidity"], "环"
            )
        if sensor.pressure_diff:
            avg_pressure = sum(sensor.pressure_diff) / len(sensor.pressure_diff)
            self._check_env_violation(
                violations, "压差", avg_pressure, ENV_SOP_RANGES["pressure_diff"], "环"
            )

        if violations:
            logger.info(
                "SOP violations detected | count=%d | params=%s",
                len(violations),
                [v["param"] for v in violations],
            )
        return violations

    def _check_env_violation(
        self,
        violations: List[Dict[str, Any]],
        name: str,
        value: float,
        sop: Dict[str, Any],
        dimension: str,
    ) -> None:
        """
        检查单个环境参数是否违规

        参数:
            violations: 违规列表（原地修改）
            name: 参数中文名称
            value: 参数实际值
            sop: SOP 标准范围字典
            dimension: 所属维度（环）
        """
        if value < sop["min"] or value > sop["max"]:
            violations.append(
                {
                    "param": name,
                    "key": name,
                    "value": round(value, 2),
                    "sop_min": sop["min"],
                    "sop_max": sop["max"],
                    "unit": sop["unit"],
                    "deviation": "偏低" if value < sop["min"] else "偏高",
                    "dimension": dimension,
                }
            )

    async def _retrieve_similar_cases(self, defect_type: str) -> str:
        """
        从向量知识库检索相似历史根因分析案例

        参数:
            defect_type: 不良类型（如"缩孔"）

        返回:
            str: 格式化的历史案例文本
        """
        if not self._vector_db:
            return ""
        try:
            results = await self._vector_db.similarity_search(
                f"涂装{defect_type}根因分析", top_k=5
            )
            if results:
                return "\n".join(
                    [f"- {r.get('content', '')}" for r in results]
                )
        except Exception as exc:
            logger.warning("Knowledge retrieval failed: %s", str(exc))
        return ""

    async def _reason_with_llm(
        self,
        preprocessed_data: PreprocessedData,
        semantic_label: SemanticLabel,
        sop_violations: List[Dict[str, Any]],
        knowledge_context: str,
    ) -> Dict[str, Any]:
        """
        调用 LLM 进行六维度根因推理

        组装包含不良现象、产品信息、工艺参数（含SOP范围）、环境数据、
        SOP违规检测结果、历史案例参考的完整 Prompt

        参数:
            preprocessed_data: 预处理数据
            semantic_label: 语义标签
            sop_violations: SOP 违规检测结果
            knowledge_context: 知识库检索结果

        返回:
            Dict[str, Any]: LLM 返回的 JSON 根因分析结果
        """
        params = preprocessed_data.robot_params
        sensor = preprocessed_data.sensor_data

        # 格式化 SOP 违规信息
        violation_text = ""
        if sop_violations:
            violation_lines = []
            for v in sop_violations:
                violation_lines.append(
                    f"  - {v['param']}: 实际值={v['value']}{v['unit']}, "
                    f"SOP范围=[{v['sop_min']}, {v['sop_max']}]{v['unit']}, "
                    f"偏差={v['deviation']}, 维度={v['dimension']}"
                )
            violation_text = f"\n\n【SOP违规项(已检测)】:\n" + "\n".join(violation_lines)

        # 计算环境参数均值
        env_summary = ""
        if sensor.temperature:
            avg_t = sum(sensor.temperature) / len(sensor.temperature)
            avg_h = sum(sensor.humidity) / len(sensor.humidity) if sensor.humidity else 0
            avg_p = sum(sensor.pressure_diff) / len(sensor.pressure_diff) if sensor.pressure_diff else 0
            env_summary = (
                f"温度均值={avg_t:.1f}°C, 湿度均值={avg_h:.1f}%, 压差均值={avg_p:.1f}Pa"
            )

        # 组装 User Prompt
        user_content = f"""请对以下涂装不良品进行根因分析：

【不良现象】:
- 类型: {semantic_label.defect_type}
- 大类: {semantic_label.defect_category}
- 严重程度: {semantic_label.severity}
- 描述: {semantic_label.description}

【产品信息】:
- 型号: {preprocessed_data.product_info.product_model}
- 产线: {preprocessed_data.product_info.line_id}
- 时间: {preprocessed_data.product_info.timestamp}

【关键工艺参数】:
- 喷涂电压: {params.voltage} kV (SOP: {SOP_RANGES['voltage']['min']}-{SOP_RANGES['voltage']['max']} kV)
- 雾化压力: {params.atomization_pressure} MPa (SOP: {SOP_RANGES['atomization_pressure']['min']}-{SOP_RANGES['atomization_pressure']['max']} MPa)
- 枪距: {params.gun_distance} mm (SOP: {SOP_RANGES['gun_distance']['min']}-{SOP_RANGES['gun_distance']['max']} mm)
- 流量: {params.flow_rate} cc/min (SOP: {SOP_RANGES['flow_rate']['min']}-{SOP_RANGES['flow_rate']['max']} cc/min)
- 机器人速度: {params.robot_speed} mm/s (SOP: {SOP_RANGES['robot_speed']['min']}-{SOP_RANGES['robot_speed']['max']} mm/s)
- 重叠率: {params.overlap_rate} % (SOP: {SOP_RANGES['overlap_rate']['min']}-{SOP_RANGES['overlap_rate']['max']} %)
- 成型气压: {params.shaping_air} MPa (SOP: {SOP_RANGES['shaping_air']['min']}-{SOP_RANGES['shaping_air']['max']} MPa)
- 触发延迟: {params.trigger_delay} ms (SOP: {SOP_RANGES['trigger_delay']['min']}-{SOP_RANGES['trigger_delay']['max']} ms)

【环境数据】: {env_summary}
{violation_text}"""

        if knowledge_context:
            user_content += f"\n\n【历史相似案例参考】:\n{knowledge_context}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return await llm_client.chat_json(
            messages=messages, agent_name="root_cause_agent"
        )

    def _parse_causes(self, llm_result: Dict[str, Any]) -> List[RootCauseItem]:
        """
        解析 LLM 返回的根因列表

        将 LLM 返回的 JSON 中的 causes 数组转换为 RootCauseItem 对象列表，
        并按权重降序排序。

        参数:
            llm_result: LLM 返回的 JSON 字典

        返回:
            List[RootCauseItem]: 按权重降序排列的根因列表
        """
        raw_causes = llm_result.get("causes", [])
        causes: List[RootCauseItem] = []

        for item in raw_causes:
            try:
                causes.append(
                    RootCauseItem(
                        root_cause=item.get("root_cause", "未知根因"),
                        dimension=item.get("dimension", "机"),
                        weight=float(item.get("weight", 0.0)),
                        evidence=item.get("evidence", ""),
                        sop_reference=item.get("sop_reference"),
                    )
                )
            except Exception as exc:
                logger.warning("Failed to parse cause item: %s | error=%s", item, str(exc))

        causes.sort(key=lambda c: c.weight, reverse=True)
        return causes

    def _adjust_weights_with_sop(
        self,
        causes: List[RootCauseItem],
        violations: List[Dict[str, Any]],
    ) -> List[RootCauseItem]:
        """
        根据 SOP 违规检测结果调整根因权重

        策略：
          - 如果 LLM 推理的根因维度与 SOP 违规维度一致，提升该根因权重 20%
          - 确保所有权重之和不超过 1.0

        参数:
            causes: LLM 推理的根因列表
            violations: SOP 违规检测结果

        返回:
            List[RootCauseItem]: 权重调整后的根因列表
        """
        violation_dimensions = {v["dimension"] for v in violations}

        for cause in causes:
            if cause.dimension in violation_dimensions:
                cause.weight = min(cause.weight * 1.2, 0.9)
                logger.info(
                    "Weight boosted for SOP-matched cause | dimension=%s | new_weight=%.2f",
                    cause.dimension,
                    cause.weight,
                )

        # 归一化权重
        total_weight = sum(c.weight for c in causes)
        if total_weight > 1.0:
            for cause in causes:
                cause.weight = round(cause.weight / total_weight, 2)

        causes.sort(key=lambda c: c.weight, reverse=True)
        return causes
