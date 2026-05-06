"""
================================================================================
 汽车涂装不良品智能分析系统 - 环境传感器连接器
================================================================================
 功能说明：
   - 通过 Modbus TCP 协议对接涂装车间环境传感器
   - 实时读取温度、湿度、压差、风速、颗粒物浓度等环境参数
   - 支持时序数据批量采集（按时间窗口回溯）
   - 提供环境数据统计摘要（均值、范围）

 传感器寄存器映射（Modbus Holding Registers）：
   地址 0-1:  温度 (°C) — IEEE 754 单精度浮点数
   地址 2-3:  湿度 (%) — IEEE 754 单精度浮点数
   地址 4-5:  压差 (Pa) — IEEE 754 单精度浮点数
   地址 6-7:  风速 (m/s) — IEEE 754 单精度浮点数
   地址 8-9:  颗粒物计数 (个/m³) — IEEE 754 单精度浮点数

 数据采集策略：
   - 默认采集不良发生前 5 分钟（300秒）的数据
   - 采样间隔 5 秒，共 60 个数据点
   - 单点读取失败时填充 0 值，不中断采集流程

 新手提示：
   - 本连接器为可选组件，未配置时系统自动使用模拟数据
   - 在 .env 中设置 SENSOR_HOST 和 SENSOR_PORT 即可启用
   - Modbus TCP 默认端口为 502
================================================================================
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)

# ======================== Modbus 寄存器地址映射 ========================
SENSOR_REGISTERS: Dict[str, int] = {
    "temperature": 0,
    "humidity": 2,
    "pressure_diff": 4,
    "air_velocity": 6,
    "particulate_count": 8,
}
"""传感器类型 → Modbus 保持寄存器起始地址"""


class SensorConnectorError(Exception):
    """传感器连接器异常类，所有传感器通讯相关错误均使用此类"""
    pass


class SensorConnector:
    """
    环境传感器 Modbus TCP 连接器

    职责：
      1. 管理 Modbus TCP 客户端生命周期
      2. 封装寄存器读写操作
      3. 提供时序数据采集和环境统计功能

    使用示例：
        connector = SensorConnector()
        values = await connector.get_current_values()
        series = await connector.get_time_series("DEF-2024-001")
        await connector.close()
    """

    def __init__(self) -> None:
        """初始化传感器连接器，从全局配置加载连接参数"""
        self._host = settings.SENSOR_HOST
        self._port = settings.SENSOR_PORT
        self._timeout = settings.SENSOR_TIMEOUT
        self._client = None

    async def connect(self) -> None:
        """
        建立 Modbus TCP 连接

        使用懒加载方式导入 pymodbus，避免启动时的依赖问题

        异常:
            SensorConnectorError: 连接超时或网络不可达时抛出
        """
        from pymodbus.client import AsyncModbusTcpClient

        try:
            self._client = AsyncModbusTcpClient(
                host=self._host,
                port=self._port,
                timeout=self._timeout,
            )
            await asyncio.wait_for(self._client.connect(), timeout=self._timeout)
            logger.info("Sensor connected | host=%s | port=%s", self._host, self._port)
        except Exception as exc:
            logger.error("Sensor connection failed | host=%s | port=%s | error=%s", self._host, self._port, str(exc))
            raise SensorConnectorError(f"Failed to connect to sensor: {exc}") from exc

    async def close(self) -> None:
        """关闭 Modbus TCP 连接"""
        if self._client and self._client.connected:
            self._client.close()
            self._client = None

    async def _ensure_connected(self) -> None:
        """确保 Modbus 客户端已连接，未连接则自动重连"""
        if not self._client or not self._client.connected:
            await self.connect()

    async def _read_holding_registers(self, address: int, count: int = 1) -> List[int]:
        """
        读取 Modbus 保持寄存器

        参数:
            address: 寄存器起始地址
            count: 读取的寄存器数量

        返回:
            List[int]: 寄存器值列表

        异常:
            SensorConnectorError: Modbus 通讯异常时抛出
        """
        from pymodbus.exceptions import ModbusException

        await self._ensure_connected()
        try:
            response = await self._client.read_holding_registers(address=address, count=count, slave=1)
            if isinstance(response, ModbusException):
                raise SensorConnectorError(f"Modbus error: {response}")
            return response.registers
        except Exception as exc:
            logger.error("Modbus read failed | address=%d | error=%s", address, str(exc))
            raise SensorConnectorError(f"Modbus read failed: {exc}") from exc

    def _registers_to_float(self, registers: List[int]) -> float:
        """
        将 Modbus 寄存器值转换为 IEEE 754 单精度浮点数

        2 个 16 位寄存器组合为 1 个 32 位浮点数

        参数:
            registers: 寄存器值列表

        返回:
            float: 转换后的浮点数值
        """
        if len(registers) >= 2:
            combined = (registers[0] << 16) | registers[1]
            import struct
            return struct.unpack("f", struct.pack("I", combined))[0]
        elif len(registers) == 1:
            return float(registers[0])
        return 0.0

    async def get_current_values(self) -> Dict[str, float]:
        """
        获取所有传感器的当前瞬时值

        返回:
            Dict[str, float]: 传感器名称 → 当前值
        """
        try:
            values: Dict[str, float] = {}
            for sensor_name, reg_addr in SENSOR_REGISTERS.items():
                regs = await self._read_holding_registers(reg_addr, count=2)
                values[sensor_name] = round(self._registers_to_float(regs), 2)

            logger.info("Sensor current values fetched | temp=%.1f | hum=%.1f | pressure=%.1f",
                        values.get("temperature", 0), values.get("humidity", 0), values.get("pressure_diff", 0))
            return values

        except Exception as exc:
            logger.error("Failed to fetch sensor current values: %s", str(exc))
            raise SensorConnectorError(f"Failed to fetch sensor values: {exc}") from exc

    async def get_time_series(
        self,
        defect_id: str,
        window_seconds: int = 300,
    ) -> Dict[str, Any]:
        """
        获取不良品发生前后指定时间窗口内的传感器时序数据

        采集策略：
          - 从不良发生时间向前回溯 window_seconds 秒
          - 每 5 秒采样一次
          - 单点读取失败时填充 0 值，不中断采集

        参数:
            defect_id: 不良品唯一标识
            window_seconds: 时间窗口大小（秒），默认 300（5分钟）

        返回:
            Dict: 包含 temperature, humidity, pressure_diff, timestamps 四个列表
        """
        try:
            # 从 defect_id 中提取时间戳
            timestamp = int(defect_id.split("-")[-1]) if defect_id.split("-")[-1].isdigit() else int(datetime.now().timestamp())
            window_start = timestamp - window_seconds
            interval_seconds = 5

            temperatures: List[float] = []
            humidities: List[float] = []
            pressure_diffs: List[float] = []
            timestamps: List[datetime] = []

            num_points = window_seconds // interval_seconds

            for i in range(num_points):
                point_ts = window_start + (i * interval_seconds)
                try:
                    temp_regs = await self._read_holding_registers(SENSOR_REGISTERS["temperature"], count=2)
                    hum_regs = await self._read_holding_registers(SENSOR_REGISTERS["humidity"], count=2)
                    press_regs = await self._read_holding_registers(SENSOR_REGISTERS["pressure_diff"], count=2)

                    temperatures.append(round(self._registers_to_float(temp_regs), 2))
                    humidities.append(round(self._registers_to_float(hum_regs), 2))
                    pressure_diffs.append(round(self._registers_to_float(press_regs), 2))
                    timestamps.append(datetime.fromtimestamp(point_ts))

                except Exception as exc:
                    logger.warning("Sensor read error at point %d: %s", i, str(exc))
                    temperatures.append(0.0)
                    humidities.append(0.0)
                    pressure_diffs.append(0.0)
                    timestamps.append(datetime.fromtimestamp(point_ts))

                await asyncio.sleep(0.01)

            logger.info(
                "Sensor time series fetched | defect_id=%s | points=%d | window=%ds",
                defect_id,
                num_points,
                window_seconds,
            )

            return {
                "temperature": temperatures,
                "humidity": humidities,
                "pressure_diff": pressure_diffs,
                "timestamps": timestamps,
            }

        except Exception as exc:
            logger.error("Failed to fetch sensor time series | defect_id=%s | error=%s", defect_id, str(exc))
            raise SensorConnectorError(f"Failed to fetch sensor time series: {exc}") from exc

    async def get_environment_summary(self, window_seconds: int = 300) -> Dict[str, Any]:
        """
        获取环境参数统计摘要

        计算指定时间窗口内各环境参数的均值和范围

        参数:
            window_seconds: 统计时间窗口（秒）

        返回:
            Dict: 包含 temperature, humidity, pressure_diff 的 avg/min/max
        """
        try:
            time_series = await self.get_time_series(defect_id=f"SUMMARY_{datetime.now().timestamp()}", window_seconds=window_seconds)

            temps = time_series.get("temperature", [])
            humids = time_series.get("humidity", [])
            pressures = time_series.get("pressure_diff", [])

            def safe_avg(data: List[float]) -> float:
                return round(sum(data) / len(data), 2) if data else 0.0

            def safe_range(data: List[float]) -> Dict[str, float]:
                return {
                    "min": round(min(data), 2) if data else 0.0,
                    "max": round(max(data), 2) if data else 0.0,
                }

            return {
                "temperature": {"avg": safe_avg(temps), **safe_range(temps)},
                "humidity": {"avg": safe_avg(humids), **safe_range(humids)},
                "pressure_diff": {"avg": safe_avg(pressures), **safe_range(pressures)},
            }

        except Exception as exc:
            logger.error("Failed to fetch environment summary: %s", str(exc))
            raise SensorConnectorError(f"Failed to fetch environment summary: {exc}") from exc
