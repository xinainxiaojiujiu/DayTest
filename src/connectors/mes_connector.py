"""
================================================================================
 汽车涂装不良品智能分析系统 - MES 系统连接器
================================================================================
 功能说明：
   - 通过 HTTP API 对接产线 MES（制造执行系统）
   - 获取产品基础信息（型号、产线、工单、班次、操作员）
   - 获取工单详情和产线状态
   - 将分析结果回传至 MES 系统

 接口列表：
   GET  /defects/{defect_id}/product-info     - 获取产品信息
   GET  /work-orders/{work_order_id}          - 获取工单详情
   GET  /production-lines/{line_id}/status    - 获取产线状态
   POST /defects/reporting                    - 上报分析结果

 认证方式：
   - 通过 X-API-Key 请求头进行 API Key 认证

 新手提示：
   - 本连接器为可选组件，未配置时系统自动使用模拟数据
   - 在 .env 中设置 MES_API_BASE_URL 和 MES_API_KEY 即可启用
================================================================================
"""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import httpx

from src.config import settings

logger = logging.getLogger(__name__)


class MESConnectorError(Exception):
    """MES 连接器异常类，所有 MES 相关错误均使用此类"""
    pass


class MESConnector:
    """
    MES 系统 HTTP API 连接器

    职责：
      1. 管理 HTTP 客户端生命周期
      2. 封装 MES 系统 API 调用
      3. 统一错误处理和日志记录

    使用示例：
        connector = MESConnector()
        info = await connector.get_product_info("DEF-2024-001")
        await connector.close()
    """

    def __init__(self) -> None:
        """初始化 MES 连接器，从全局配置加载连接参数"""
        self._base_url = settings.MES_API_BASE_URL
        self._api_key = settings.MES_API_KEY
        self._timeout = settings.MES_TIMEOUT
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """
        获取或创建 HTTP 客户端实例（懒加载 + 自动重连）

        返回:
            httpx.AsyncClient: 已配置的异步 HTTP 客户端
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"X-API-Key": self._api_key, "Content-Type": "application/json"},
            )
        return self._client

    async def close(self) -> None:
        """关闭 HTTP 客户端，释放连接资源"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def get_product_info(self, defect_id: str) -> Dict[str, Any]:
        """
        根据不良品 ID 获取产品基础信息

        参数:
            defect_id: 不良品唯一标识

        返回:
            Dict[str, Any]: 包含 product_model, line_id, timestamp 等字段

        异常:
            MESConnectorError: 404 未找到或网络错误时抛出
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/defects/{defect_id}/product-info")
            if response.status_code == 404:
                logger.warning("MES product info not found | defect_id=%s", defect_id)
                raise MESConnectorError(f"Product info not found for defect_id={defect_id}")
            response.raise_for_status()
            data = response.json()
            logger.info("MES product info fetched | defect_id=%s | model=%s", defect_id, data.get("product_model"))
            return data
        except httpx.HTTPStatusError as exc:
            logger.error("MES HTTP error | defect_id=%s | status=%s | response=%s", defect_id, exc.response.status_code, exc.response.text)
            raise MESConnectorError(f"MES HTTP error: {exc.response.status_code}") from exc
        except Exception as exc:
            logger.error("MES request failed | defect_id=%s | error=%s", defect_id, str(exc))
            raise MESConnectorError(f"MES request failed: {exc}") from exc

    async def get_work_order(self, work_order_id: str) -> Dict[str, Any]:
        """
        获取工单详情

        参数:
            work_order_id: 工单号

        返回:
            Dict[str, Any]: 工单详细信息
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/work-orders/{work_order_id}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error("MES work order fetch failed | work_order=%s | error=%s", work_order_id, str(exc))
            raise MESConnectorError(f"Failed to fetch work order: {exc}") from exc

    async def get_production_line_status(self, line_id: str) -> Dict[str, Any]:
        """
        获取产线实时状态

        参数:
            line_id: 产线编号

        返回:
            Dict[str, Any]: 产线状态信息（运行/停机/维护等）
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/production-lines/{line_id}/status")
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            logger.error("MES line status fetch failed | line=%s | error=%s", line_id, str(exc))
            raise MESConnectorError(f"Failed to fetch line status: {exc}") from exc

    async def report_defect_analysis(
        self,
        defect_id: str,
        analysis_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        将分析结果回传至 MES 系统

        参数:
            defect_id: 不良品 ID
            analysis_result: 完整的分析结果字典

        返回:
            Dict[str, Any]: MES 系统的响应
        """
        try:
            client = await self._get_client()
            payload = {
                "defect_id": defect_id,
                "analysis_result": analysis_result,
                "reported_at": datetime.now().isoformat(),
            }
            response = await client.post("/defects/reporting", json=payload)
            response.raise_for_status()
            logger.info("Defect analysis reported to MES | defect_id=%s", defect_id)
            return response.json()
        except Exception as exc:
            logger.error("MES report failed | defect_id=%s | error=%s", defect_id, str(exc))
            raise MESConnectorError(f"Failed to report to MES: {exc}") from exc
