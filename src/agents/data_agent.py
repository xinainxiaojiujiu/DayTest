"""
================================================================================
 汽车涂装不良品智能分析系统 - 数据采集与预处理 Agent
================================================================================
 功能说明：
   - 从 MES 系统获取产品基础信息（型号、产线、工单等）
   - 从安川机器人获取喷涂工艺参数（电压、压力、枪距等200+项）
   - 从环境传感器获取时序数据（温度、湿度、压差）
   - 对采集到的数据进行清洗：去除异常值、NaN、无穷值
   - 将清洗后的数据持久化到 MySQL 和 InfluxDB

 数据清洗规则：
   - 机器人参数：过滤 NaN/Inf 值，过滤绝对值超过 1e6 的极端值
   - 传感器数据：过滤 NaN/Inf 值，温度范围 [-40, 80]°C，湿度范围 [0, 100]%，压差绝对值 < 1000Pa

 容错设计：
   - 当外部系统（MES/机器人/传感器）未连接时，自动使用模拟数据
   - 确保系统在离线环境下也能正常运行和测试

 新手提示：
   - 本 Agent 是分析流程的第一步，为后续所有 Agent 提供数据基础
   - 无需连接真实设备即可运行，系统会自动生成模拟数据
================================================================================
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.config import settings
from src.models.schemas import (
    MESProductInfo,
    PreprocessedData,
    RobotProcessParams,
    SensorTimeSeries,
)

logger = logging.getLogger(__name__)


class DataAgent:
    """
    数据采集与预处理 Agent

    职责：
      1. 并行从 MES、机器人、传感器三个数据源采集数据
      2. 对原始数据进行清洗和标准化
      3. 将预处理结果持久化到数据库

    依赖注入：
      - mes_connector: MES 系统连接器（可选，未配置时使用模拟数据）
      - robot_connector: 安川机器人连接器（可选，未配置时使用模拟数据）
      - sensor_connector: 环境传感器连接器（可选，未配置时使用模拟数据）
      - mysql_crud: MySQL 数据库操作实例（可选）
      - influx_crud: InfluxDB 时序数据库操作实例（可选）

    使用示例：
        agent = DataAgent(mes=mes_conn, robot=robot_conn, sensor=sensor_conn)
        preprocessed_data = await agent.execute("DEF-2024-001")
    """

    def __init__(
        self,
        mes_connector: Any = None,
        robot_connector: Any = None,
        sensor_connector: Any = None,
        mysql_crud: Any = None,
        influx_crud: Any = None,
    ) -> None:
        """
        初始化数据采集 Agent

        参数:
            mes_connector: MES 系统连接器实例
            robot_connector: 安川机器人连接器实例
            sensor_connector: 环境传感器连接器实例
            mysql_crud: MySQL 数据库 CRUD 操作实例
            influx_crud: InfluxDB 时序数据库 CRUD 操作实例
        """
        self._mes = mes_connector
        self._robot = robot_connector
        self._sensor = sensor_connector
        self._mysql = mysql_crud
        self._influx = influx_crud

    async def execute(self, defect_id: str) -> PreprocessedData:
        """
        执行数据采集与预处理流程（主入口）

        流程：
          1. 并行采集三个数据源的数据（MES + 机器人 + 传感器）
          2. 清洗机器人参数（去除异常值）
          3. 清洗传感器时序数据（去除异常值）
          4. 组装 PreprocessedData 对象
          5. 持久化到 MySQL 和 InfluxDB

        参数:
            defect_id: 不良品唯一标识，如 "DEF-2024-001"

        返回:
            PreprocessedData: 包含产品信息、工艺参数、环境数据的完整预处理结果

        异常:
            数据采集失败时向上抛出异常
        """
        logger.info("DataAgent started | defect_id=%s", defect_id)

        try:
            # 并行采集三个数据源，提高效率
            product_info, robot_params, sensor_data = await asyncio.gather(
                self._fetch_product_info(defect_id),
                self._fetch_robot_params(defect_id),
                self._fetch_sensor_data(defect_id),
            )

            # 数据清洗
            robot_params = self._clean_robot_params(robot_params)
            sensor_data = self._clean_sensor_data(sensor_data)

            # 组装预处理结果
            preprocessed = PreprocessedData(
                defect_id=defect_id,
                product_info=product_info,
                robot_params=robot_params,
                sensor_data=sensor_data,
            )

            # 持久化到数据库
            await self._persist_data(preprocessed)

            logger.info(
                "DataAgent completed | defect_id=%s | params_count=%d | sensor_points=%d",
                defect_id,
                len(robot_params.raw_params),
                len(sensor_data.timestamps),
            )
            return preprocessed

        except Exception as exc:
            logger.error("DataAgent failed | defect_id=%s | error=%s", defect_id, str(exc))
            raise

    async def _fetch_product_info(self, defect_id: str) -> MESProductInfo:
        """
        从 MES 系统获取产品基础信息

        容错：如果 MES 连接器未配置或连接失败，返回模拟数据

        参数:
            defect_id: 不良品唯一标识

        返回:
            MESProductInfo: 包含产品型号、产线编号、工单号、班次、操作员等信息
        """
        if self._mes is None:
            logger.warning("MES connector not configured, returning mock data")
            return MESProductInfo(
                product_model="MODEL-A",
                line_id="LINE-01",
                timestamp=datetime.now(),
                work_order=f"WO-{defect_id}",
                shift="A",
                operator_id="OP-001",
            )
        try:
            data: Dict[str, Any] = await self._mes.get_product_info(defect_id)
            return MESProductInfo(**data)
        except Exception as exc:
            logger.error("Failed to fetch MES data | defect_id=%s | error=%s", defect_id, str(exc))
            raise

    async def _fetch_robot_params(self, defect_id: str) -> RobotProcessParams:
        """
        从安川机器人获取喷涂工艺参数

        获取 8 项关键参数 + 200+ 项原始参数：
          - 喷涂电压 (kV)
          - 雾化压力 (MPa)
          - 枪距 (mm)
          - 流量 (cc/min)
          - 机器人速度 (mm/s)
          - 重叠率 (%)
          - 成型气压 (MPa)
          - 触发延迟 (ms)

        容错：如果机器人连接器未配置，返回模拟数据

        参数:
            defect_id: 不良品唯一标识

        返回:
            RobotProcessParams: 包含关键工艺参数和原始参数字典
        """
        if self._robot is None:
            logger.warning("Robot connector not configured, returning mock data")
            return RobotProcessParams(
                voltage=60.0,
                atomization_pressure=0.25,
                gun_distance=250.0,
                flow_rate=350.0,
                robot_speed=800.0,
                overlap_rate=50.0,
                shaping_air=0.15,
                trigger_delay=50.0,
                raw_params={"voltage": 60.0, "atomization_pressure": 0.25},
            )
        try:
            data: Dict[str, Any] = await self._robot.get_process_params(defect_id)
            return RobotProcessParams(
                voltage=data.get("voltage"),
                atomization_pressure=data.get("atomization_pressure"),
                gun_distance=data.get("gun_distance"),
                flow_rate=data.get("flow_rate"),
                robot_speed=data.get("robot_speed"),
                overlap_rate=data.get("overlap_rate"),
                shaping_air=data.get("shaping_air"),
                trigger_delay=data.get("trigger_delay"),
                raw_params=data,
            )
        except Exception as exc:
            logger.error("Failed to fetch robot data | defect_id=%s | error=%s", defect_id, str(exc))
            raise

    async def _fetch_sensor_data(self, defect_id: str) -> SensorTimeSeries:
        """
        从环境传感器获取时序数据

        采集不良发生前 N 秒（由 SENSOR_TIME_WINDOW_SECONDS 配置）的环境数据：
          - 温度序列 (°C)
          - 湿度序列 (%)
          - 压差序列 (Pa)

        容错：如果传感器连接器未配置，返回带正态噪声的模拟数据

        参数:
            defect_id: 不良品唯一标识

        返回:
            SensorTimeSeries: 包含温度、湿度、压差的时间序列数据
        """
        if self._sensor is None:
            logger.warning("Sensor connector not configured, returning mock data")
            now = datetime.now()
            timestamps = [now - timedelta(seconds=i) for i in range(60, 0, -1)]
            return SensorTimeSeries(
                temperature=[25.0 + np.random.normal(0, 0.5) for _ in range(60)],
                humidity=[55.0 + np.random.normal(0, 1.0) for _ in range(60)],
                pressure_diff=[10.0 + np.random.normal(0, 0.3) for _ in range(60)],
                timestamps=timestamps,
            )
        try:
            data: Dict[str, Any] = await self._sensor.get_time_series(
                defect_id=defect_id,
                window_seconds=settings.SENSOR_TIME_WINDOW_SECONDS,
            )
            return SensorTimeSeries(**data)
        except Exception as exc:
            logger.error("Failed to fetch sensor data | defect_id=%s | error=%s", defect_id, str(exc))
            raise

    def _clean_robot_params(self, params: RobotProcessParams) -> RobotProcessParams:
        """
        清洗机器人工艺参数

        清洗规则：
          1. 移除 None 值
          2. 移除 NaN 和 Inf 值
          3. 移除绝对值超过 1e6 的极端异常值

        参数:
            params: 原始机器人工艺参数

        返回:
            RobotProcessParams: 清洗后的工艺参数
        """
        cleaned_raw: Dict[str, Any] = {}
        for key, value in params.raw_params.items():
            if value is None:
                continue
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    logger.warning("Removed outlier param | key=%s | value=%s", key, value)
                    continue
                if abs(value) > 1e6:
                    logger.warning("Removed extreme param | key=%s | value=%s", key, value)
                    continue
            cleaned_raw[key] = value

        params.raw_params = cleaned_raw
        return params

    def _clean_sensor_data(self, data: SensorTimeSeries) -> SensorTimeSeries:
        """
        清洗传感器时序数据

        清洗规则：
          1. 对齐各序列长度（取最短长度）
          2. 过滤 NaN 和 Inf 值
          3. 过滤物理范围异常值：
             - 温度: [-40°C, 80°C]
             - 湿度: [0%, 100%]
             - 压差: [-1000Pa, 1000Pa]

        参数:
            data: 原始传感器时序数据

        返回:
            SensorTimeSeries: 清洗后的时序数据
        """
        if not data.timestamps:
            return data

        # 对齐各序列长度
        min_len = min(len(data.timestamps), len(data.temperature), len(data.humidity), len(data.pressure_diff))

        valid_indices: List[int] = []
        for i in range(min_len):
            t, h, p = data.temperature[i], data.humidity[i], data.pressure_diff[i]
            if any(np.isnan(v) or np.isinf(v) for v in [t, h, p]):
                continue
            if t < -40 or t > 80:
                continue
            if h < 0 or h > 100:
                continue
            if abs(p) > 1000:
                continue
            valid_indices.append(i)

        if not valid_indices:
            logger.warning("All sensor data points filtered out as outliers")
            return data

        data.temperature = [data.temperature[i] for i in valid_indices]
        data.humidity = [data.humidity[i] for i in valid_indices]
        data.pressure_diff = [data.pressure_diff[i] for i in valid_indices]
        data.timestamps = [data.timestamps[i] for i in valid_indices]

        return data

    async def _persist_data(self, data: PreprocessedData) -> None:
        """
        将预处理数据持久化到数据库

        写入目标：
          - MySQL: 产品信息、工艺参数（JSON格式）
          - InfluxDB: 传感器时序数据（温度、湿度、压差）

        参数:
            data: 预处理完成的数据对象
        """
        try:
            if self._mysql:
                await self._mysql.save_preprocessed_data(data)
            if self._influx:
                await self._influx.write_sensor_time_series(
                    defect_id=data.defect_id,
                    timestamps=data.sensor_data.timestamps,
                    temperature=data.sensor_data.temperature,
                    humidity=data.sensor_data.humidity,
                    pressure_diff=data.sensor_data.pressure_diff,
                )
        except Exception as exc:
            logger.error("Failed to persist data | defect_id=%s | error=%s", data.defect_id, str(exc))
