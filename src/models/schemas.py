from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DefectIDRequest(BaseModel):
    defect_id: str = Field(..., description="不良品唯一标识")


class MESProductInfo(BaseModel):
    product_model: str = Field(..., description="产品型号")
    line_id: str = Field(..., description="产线编号")
    timestamp: datetime = Field(..., description="生产时间戳")
    work_order: str = Field(default="", description="工单号")
    shift: str = Field(default="", description="班次")
    operator_id: str = Field(default="", description="操作员ID")


class RobotProcessParams(BaseModel):
    voltage: Optional[float] = Field(default=None, description="喷涂电压(kV)")
    atomization_pressure: Optional[float] = Field(default=None, description="雾化压力(MPa)")
    gun_distance: Optional[float] = Field(default=None, description="枪距(mm)")
    flow_rate: Optional[float] = Field(default=None, description="流量(cc/min)")
    robot_speed: Optional[float] = Field(default=None, description="机器人速度(mm/s)")
    overlap_rate: Optional[float] = Field(default=None, description="重叠率(%)")
    shaping_air: Optional[float] = Field(default=None, description="成型气压(MPa)")
    trigger_delay: Optional[float] = Field(default=None, description="触发延迟(ms)")
    raw_params: Dict[str, Any] = Field(default_factory=dict, description="原始200+项参数")


class SensorTimeSeries(BaseModel):
    temperature: List[float] = Field(default_factory=list, description="温度序列(°C)")
    humidity: List[float] = Field(default_factory=list, description="湿度序列(%)")
    pressure_diff: List[float] = Field(default_factory=list, description="压差序列(Pa)")
    timestamps: List[datetime] = Field(default_factory=list, description="时间戳序列")


class PreprocessedData(BaseModel):
    defect_id: str
    product_info: MESProductInfo
    robot_params: RobotProcessParams
    sensor_data: SensorTimeSeries
    cleaned_at: datetime = Field(default_factory=datetime.now)


class SemanticLabel(BaseModel):
    defect_type: str = Field(..., description="不良类型(如:缩孔/橘皮/流挂/色差/颗粒)")
    defect_category: str = Field(..., description="不良大类(外观/色差/附着力)")
    severity: str = Field(default="medium", description="严重程度(low/medium/high/critical)")
    description: str = Field(default="", description="自然语言描述")
    ocr_text: Optional[str] = Field(default=None, description="OCR识别的缺陷图片文字")
    confidence: float = Field(default=0.0, description="解析置信度(0-1)")


class RootCauseItem(BaseModel):
    root_cause: str = Field(..., description="根因描述")
    dimension: str = Field(..., description="维度(人/机/料/法/环/测)")
    weight: float = Field(..., ge=0, le=1, description="影响权重(0-1)")
    evidence: str = Field(..., description="数据证据")
    sop_reference: Optional[str] = Field(default=None, description="SOP参考条款")


class RootCauseResult(BaseModel):
    defect_id: str
    causes: List[RootCauseItem] = Field(..., description="按权重排序的根因列表(Top3)")
    reasoning_chain: str = Field(default="", description="推理链路摘要")
    analyzed_at: datetime = Field(default_factory=datetime.now)


class SolutionItem(BaseModel):
    action: str = Field(..., description="纠正措施")
    responsible: str = Field(default="", description="责任方")
    priority: str = Field(default="medium", description="优先级(low/medium/high)")
    estimated_effort: str = Field(default="", description="预估工作量")
    reference_case: Optional[str] = Field(default=None, description="历史案例参考")


class SolutionResult(BaseModel):
    defect_id: str
    solutions: List[SolutionItem]
    generated_at: datetime = Field(default_factory=datetime.now)


class ClosedLoopResult(BaseModel):
    defect_id: str
    verification_status: str = Field(..., description="验证状态(passed/failed/pending)")
    effectiveness_score: float = Field(default=0.0, description="有效性评分(0-1)")
    knowledge_updated: bool = Field(default=False, description="是否更新知识库")
    feedback: str = Field(default="", description="闭环反馈描述")
    verified_at: datetime = Field(default_factory=datetime.now)


class LLMTokenLog(BaseModel):
    request_id: str = Field(..., description="请求唯一ID")
    agent_name: str = Field(..., description="调用Agent名称")
    model: str = Field(..., description="模型名称")
    prompt_tokens: int = Field(..., description="Prompt消耗Token数")
    completion_tokens: int = Field(..., description="Completion消耗Token数")
    total_tokens: int = Field(..., description="总消耗Token数")
    latency_ms: float = Field(..., description="响应延迟(ms)")
    created_at: datetime = Field(default_factory=datetime.now)
