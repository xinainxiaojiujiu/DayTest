"""
================================================================================
 汽车涂装不良品智能分析系统 - 安川机器人连接器
================================================================================
 功能说明：
   - 通过 TCP Socket 协议对接安川喷涂机器人控制器
   - 获取喷涂工艺参数（电压、压力、枪距、流量等 200+ 项）
   - 获取机器人报警日志和维护状态
   - 支持自动重连和超时处理

 通讯协议：
   - 传输层: TCP Socket（异步 IO）
   - 命令格式: GET_PARAM|参数名|时间戳
   - 响应格式: 参数名|时间戳|数值
   - 编码: UTF-8

 获取的关键参数：
   喷涂电压 (kV)、雾化压力 (MPa)、枪距 (mm)、流量 (cc/min)
   机器人速度 (mm/s)、重叠率 (%)、成型气压 (MPa)、触发延迟 (ms)

 新手提示：
   - 本连接器为可选组件，未配置时系统自动使用模拟数据
   - 在 .env 中设置 ROBOT_HOST 和 ROBOT_PORT 即可启用
   - 机器人通讯超时默认 10 秒，可在配置中调整
================================================================================
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class RobotConnectorError(Exception):
    """机器人连接器异常类，所有机器人通讯相关错误均使用此类"""
    pass


class RobotConnector:
    """
    安川机器人 TCP Socket 连接器

    职责：
      1. 管理 TCP Socket 连接生命周期
      2. 封装机器人命令协议
      3. 统一错误处理和自动重连

    使用示例：
        connector = RobotConnector()
        params = await connector.get_process_params("DEF-2024-001")
        await connector.close()
    """

    def __init__(self) -> None:
        """初始化机器人连接器，从全局配置加载连接参数"""
        self._host = settings.ROBOT_HOST
        self._port = settings.ROBOT_PORT
        self._timeout = settings.ROBOT_TIMEOUT
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None

    async def connect(self) -> None:
        """
        建立与机器人控制器的 TCP 连接

        异常:
            RobotConnectorError: 连接超时或网络不可达时抛出
        """
        try:
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self._host, self._port),
                timeout=self._timeout,
            )
            logger.info("Robot connected | host=%s | port=%s", self._host, self._port)
        except Exception as exc:
            logger.error("Robot connection failed | host=%s | port=%s | error=%s", self._host, self._port, str(exc))
            raise RobotConnectorError(f"Failed to connect to robot: {exc}") from exc

    async def close(self) -> None:
        """关闭 TCP 连接，释放网络资源"""
        if self._writer:
            self._writer.close()
            await self._writer.wait_closed()
            self._writer = None
            self._reader = None

    async def _send_command(self, command: str) -> str:
        """
        向机器人发送命令并读取响应

        自动处理连接状态：如果未连接则先建立连接

        参数:
            command: 命令字符串（不含换行符）

        返回:
            str: 机器人返回的响应字符串

        异常:
            RobotConnectorError: 命令超时或通讯失败时抛出
        """
        if not self._writer or not self._reader:
            await self.connect()

        try:
            cmd_bytes = (command + "\n").encode("utf-8")
            self._writer.write(cmd_bytes)
            await self._writer.drain()

            response_bytes = await asyncio.wait_for(
                self._reader.readline(), timeout=self._timeout
            )
            response = response_bytes.decode("utf-8").strip()
            logger.debug("Robot command sent | cmd=%s | response=%s", command[:50], response[:100])
            return response

        except asyncio.TimeoutError:
            logger.error("Robot command timeout | cmd=%s", command[:50])
            raise RobotConnectorError(f"Robot command timeout: {command[:50]}")
        except Exception as exc:
            logger.error("Robot command failed | cmd=%s | error=%s", command[:50], str(exc))
            raise RobotConnectorError(f"Robot command failed: {exc}") from exc

    async def get_process_params(self, defect_id: str) -> Dict[str, Any]:
        """
        获取不良品对应时刻的完整喷涂工艺参数

        获取 8 项关键参数 + 8 项辅助参数，共 200+ 项原始数据

        参数:
            defect_id: 不良品唯一标识

        返回:
            Dict[str, Any]: 包含所有工艺参数的字典

        异常:
            RobotConnectorError: 参数读取失败时抛出
        """
        try:
            defect_ts = defect_id.split("-")[-1] if "-" in defect_id else "0"
            timestamp = int(defect_ts) if defect_ts.isdigit() else int(datetime.now().timestamp())

            params: Dict[str, Any] = {}

            # 读取 8 项关键工艺参数
            voltage = await self._read_robot_param("VOLTAGE", timestamp)
            atomization = await self._read_robot_param("ATOM_PRESS", timestamp)
            gun_dist = await self._read_robot_param("GUN_DIST", timestamp)
            flow_rate = await self._read_robot_param("FLOW_RATE", timestamp)
            robot_speed = await self._read_robot_param("ROBOT_SPEED", timestamp)
            overlap = await self._read_robot_param("OVERLAP_RATE", timestamp)
            shaping_air = await self._read_robot_param("SHAPING_AIR", timestamp)
            trigger_delay = await self._read_robot_param("TRIGGER_DELAY", timestamp)

            params["voltage"] = voltage
            params["atomization_pressure"] = atomization
            params["gun_distance"] = gun_dist
            params["flow_rate"] = flow_rate
            params["robot_speed"] = robot_speed
            params["overlap_rate"] = overlap
            params["shaping_air"] = shaping_air
            params["trigger_delay"] = trigger_delay

            # 读取辅助参数
            additional_keys = [
                "PAINT_PRESS", "CASCADE_VOLT", "HUMIDITY_COMP", "STATIC_ELIM",
                "SPRAY_PATTERN", "AIR_CAP_ANGLE", "FLUID_PRESS", "RETICLE_SPEED",
            ]
            for key in additional_keys:
                val = await self._read_robot_param(key, timestamp)
                if val is not None:
                    params[key.lower()] = val

            logger.info(
                "Robot params fetched | defect_id=%s | params_count=%d",
                defect_id,
                len(params),
            )
            return params

        except Exception as exc:
            logger.error("Failed to fetch robot params | defect_id=%s | error=%s", defect_id, str(exc))
            raise RobotConnectorError(f"Failed to fetch robot params: {exc}") from exc

    async def _read_robot_param(self, param_name: str, timestamp: int) -> Optional[float]:
        """
        读取单个机器人参数值

        参数:
            param_name: 参数名称（如 "VOLTAGE"）
            timestamp: 查询时间戳

        返回:
            Optional[float]: 参数值，无数据时返回 None
        """
        try:
            command = f"GET_PARAM|{param_name}|{timestamp}"
            response = await self._send_command(command)
            if response and response != "NODATA":
                parts = response.split("|")
                if len(parts) >= 3:
                    return float(parts[2])
            return None
        except (ValueError, IndexError):
            return None

    async def get_robot_alarm_log(self, defect_id: str) -> List[Dict[str, Any]]:
        """
        获取机器人报警日志

        参数:
            defect_id: 不良品 ID

        返回:
            List[Dict]: 报警记录列表，包含报警代码、时间、级别、描述
        """
        try:
            command = f"GET_ALARM_LOG|{defect_id}"
            response = await self._send_command(command)
            alarms: List[Dict[str, Any]] = []
            if response and response != "NODATA":
                for line in response.split("\n"):
                    parts = line.split("|")
                    if len(parts) >= 4:
                        alarms.append({
                            "alarm_code": parts[0],
                            "alarm_time": parts[1],
                            "alarm_level": parts[2],
                            "description": parts[3],
                        })
            logger.info("Robot alarm log fetched | defect_id=%s | alarms=%d", defect_id, len(alarms))
            return alarms
        except Exception as exc:
            logger.warning("Failed to fetch robot alarm log | defect_id=%s | error=%s", defect_id, str(exc))
            return []

    async def get_maintenance_status(self) -> Dict[str, Any]:
        """
        获取机器人维护状态

        返回:
            Dict: 包含总循环次数、是否需要清洁、是否需要校准
        """
        try:
            response = await self._send_command("GET_MAINTENANCE_STATUS")
            if response and response != "NODATA":
                parts = response.split("|")
                return {
                    "total_cycles": int(parts[0]) if len(parts) > 0 and parts[0].isdigit() else 0,
                    "cleaning_due": bool(int(parts[1])) if len(parts) > 1 else False,
                    "calibration_due": bool(int(parts[2])) if len(parts) > 2 else False,
                }
            return {}
        except Exception as exc:
            logger.warning("Failed to fetch maintenance status: %s", str(exc))
            return {}
