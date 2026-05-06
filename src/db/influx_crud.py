"""
================================================================================
 汽车涂装不良品智能分析系统 - InfluxDB 时序数据库操作模块
================================================================================
 功能说明：
   - 管理 InfluxDB 异步客户端的创建与复用
   - 批量写入环境传感器时序数据（温度/湿度/压差）
   - 支持按时间范围查询历史传感器数据
   - 提供聚合统计查询（均值等）

 数据说明：
   measurement: coating_env_sensor
   tags:        defect_id（不良品标识）
   fields:      temperature（温度°C）, humidity（湿度%）, pressure_diff（压差Pa）

 新手提示：
   - InfluxDB 用于存储高频时序数据，与 MySQL 的关系型数据互补
   - 默认采集不良发生前 5 分钟的数据，每 5 秒一个采样点
   - InfluxDB 未连接时系统使用模拟数据运行
================================================================================
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.config import settings

logger = logging.getLogger(__name__)


class InfluxCRUDError(Exception):
    """InfluxDB 操作异常类"""
    pass


_client = None


def get_client():
    """
    获取 InfluxDB 异步客户端实例（懒加载单例模式）

    首次调用时根据配置创建客户端，后续调用直接返回已有实例。
    使用 InfluxDBClientAsync 支持异步写入和查询。

    返回:
        InfluxDBClientAsync 实例
    """
    global _client
    if _client is None:
        from influxdb_client.client.influxdb_client_async import InfluxDBClientAsync
        
        _client = InfluxDBClientAsync(
            url=settings.INFLUX_URL,
            token=settings.INFLUX_TOKEN,
            org=settings.INFLUX_ORG,
        )
    return _client


class InfluxCRUD:
    """
    InfluxDB 时序数据库操作封装

    职责：
      1. 管理 InfluxDB 客户端生命周期
      2. 批量写入传感器时序数据
      3. 按时间范围和不良品ID查询历史数据
      4. 提供聚合统计功能

    使用示例：
        influx = InfluxCRUD()
        await influx.write_sensor_time_series(defect_id, timestamps, temps, humids, pressures)
        data = await influx.query_sensor_data(defect_id)
        await influx.close()
    """

    def __init__(self) -> None:
        """初始化 InfluxDB CRUD 实例，加载 bucket 和 org 配置"""
        self._bucket = settings.INFLUX_BUCKET
        self._org = settings.INFLUX_ORG

    def _get_client(self):
        """获取 InfluxDB 客户端实例"""
        return get_client()

    async def write_sensor_time_series(
        self,
        defect_id: str,
        timestamps: List[datetime],
        temperature: List[float],
        humidity: List[float],
        pressure_diff: List[float],
    ) -> None:
        """
        批量写入环境传感器时序数据

        将温度、湿度、压差三个指标的时间序列数据批量写入 InfluxDB。
        每个时间点生成一个 Point 对象，包含 defect_id 标签和三个 field 值。

        参数:
            defect_id: 不良品唯一标识
            timestamps: 时间戳列表
            temperature: 温度值列表（°C）
            humidity: 湿度值列表（%）
            pressure_diff: 压差值列表（Pa）

        异常:
            InfluxCRUDError: 写入失败时抛出
        """
        from influxdb_client import Point
        
        try:
            client = self._get_client()
            write_api = client.write_api()
            points = []

            for i in range(len(timestamps)):
                point = (
                    Point("coating_env_sensor")
                    .tag("defect_id", defect_id)
                    .field("temperature", temperature[i])
                    .field("humidity", humidity[i])
                    .field("pressure_diff", pressure_diff[i])
                    .time(timestamps[i])
                )
                points.append(point)

            await write_api.write(bucket=self._bucket, org=self._org, record=points)
            logger.info(
                "InfluxDB time series written | defect_id=%s | points=%d",
                defect_id,
                len(points),
            )

        except Exception as exc:
            logger.error(
                "Failed to write sensor time series | defect_id=%s | error=%s",
                defect_id,
                str(exc),
            )
            raise InfluxCRUDError(f"Failed to write time series: {exc}") from exc

    async def query_sensor_data(
        self,
        defect_id: str,
        start: Optional[datetime] = None,
        stop: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        查询指定不良品的传感器时序数据

        使用 Flux 查询语言从 InfluxDB 中检索指定时间范围内的传感器数据。
        默认查询最近 1 小时的数据。

        参数:
            defect_id: 不良品唯一标识
            start: 查询起始时间（可选，默认1小时前）
            stop: 查询结束时间（可选，默认当前时间）

        返回:
            Dict: 包含 timestamps, temperature, humidity, pressure_diff 四个列表
        """
        try:
            client = self._get_client()
            query_api = client.query_api()

            start_str = start.isoformat() if start else "-1h"
            stop_str = stop.isoformat() if stop else "now()"

            query = f'''
                from(bucket: "{self._bucket}")
                |> range(start: {start_str}, stop: {stop_str})
                |> filter(fn: (r) => r["defect_id"] == "{defect_id}")
                |> filter(fn: (r) => r["_measurement"] == "coating_env_sensor")
                |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
                |> sort(columns: ["_time"])
            '''

            result = await query_api.query(query, org=self._org)

            if not result or len(result) == 0:
                return {"timestamps": [], "temperature": [], "humidity": [], "pressure_diff": []}

            table = result[0]
            timestamps: List[datetime] = []
            temperatures: List[float] = []
            humidities: List[float] = []
            pressure_diffs: List[float] = []

            for record in table.records:
                timestamps.append(record.get_time())
                temperatures.append(record.values.get("temperature", 0.0))
                humidities.append(record.values.get("humidity", 0.0))
                pressure_diffs.append(record.values.get("pressure_diff", 0.0))

            return {
                "timestamps": timestamps,
                "temperature": temperatures,
                "humidity": humidities,
                "pressure_diff": pressure_diffs,
            }

        except Exception as exc:
            logger.error(
                "Failed to query sensor data | defect_id=%s | error=%s",
                defect_id,
                str(exc),
            )
            raise InfluxCRUDError(f"Failed to query sensor data: {exc}") from exc

    async def query_aggregated_stats(
        self,
        defect_id: str,
    ) -> Dict[str, Any]:
        """
        查询传感器数据的聚合统计信息

        计算指定不良品在过去 24 小时内各传感器指标的均值。
        用于快速了解不良发生时的环境概况。

        参数:
            defect_id: 不良品唯一标识

        返回:
            Dict: 包含 avg_temperature, avg_humidity, avg_pressure_diff 等聚合值
        """
        try:
            client = self._get_client()
            query_api = client.query_api()

            query = f'''
                from(bucket: "{self._bucket}")
                |> range(start: -24h)
                |> filter(fn: (r) => r["defect_id"] == "{defect_id}")
                |> filter(fn: (r) => r["_measurement"] == "coating_env_sensor")
                |> mean()
            '''

            result = await query_api.query(query, org=self._org)

            stats: Dict[str, Any] = {}
            if result and len(result) > 0:
                for record in result[0].records:
                    field = record.get_field()
                    stats[f"avg_{field}"] = record.get_value()

            return stats

        except Exception as exc:
            logger.error(
                "Failed to query aggregated stats | defect_id=%s | error=%s",
                defect_id,
                str(exc),
            )
            return {}

    async def close(self) -> None:
        """
        关闭 InfluxDB 客户端连接

        释放网络资源和连接池，应在应用关闭时调用。
        """
        global _client
        if _client:
            await _client.close()
            _client = None
