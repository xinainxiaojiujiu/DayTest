"""
================================================================================
 汽车涂装不良品智能分析系统 - LLM 客户端
================================================================================
 功能说明：
   - 封装大模型调用接口，支持阿里云百炼（DashScope）和 Dify 双平台
   - 提供 chat（文本对话）和 chat_json（JSON结构化输出）两种调用方式
   - 内置指数退避重试机制，自动处理网络超时和API限流
   - 自动记录每次调用的 Token 消耗量和响应延迟
   - 支持自定义温度参数和最大输出 Token 数

 平台切换方式：
   在 .env 文件中设置 LLM_PLATFORM=dify 即可切换到 Dify 平台
   默认使用阿里云百炼（LLM_PLATFORM=dashscope）

 Dify 平台 API 规范：
   - 请求地址: {DIFY_API_URL}/chat-messages
   - 认证方式: Authorization: Bearer {api_key}
   - 请求格式: 遵循 Dify Chat API 标准格式
   - 支持流式(streaming)和阻塞(blocking)两种响应模式

 新手提示：
   - 只需在 .env 中配置 API Key 即可使用，无需修改代码
   - 如果调用失败，系统会自动重试最多3次
   - Token 消耗日志会自动记录到 MySQL 数据库（如已配置）
================================================================================
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.config import settings
from src.models.schemas import LLMTokenLog

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """LLM 客户端异常基类，所有LLM调用相关错误均使用此类"""
    pass


class LLMClient:
    """
    大模型调用客户端

    支持两种平台：
    1. 阿里云百炼（DashScope）- 通过 dashscope SDK 调用通义千问系列模型
    2. Dify 平台 - 通过 HTTP API 调用公司级部署的 Dify 应用

    使用方式：
        client = LLMClient()
        result = await client.chat(messages=[...], agent_name="my_agent")
        json_result = await client.chat_json(messages=[...], agent_name="my_agent")
    """

    def __init__(self) -> None:
        """初始化 LLM 客户端，从全局配置加载参数"""
        # 平台选择
        self._platform: str = settings.LLM_PLATFORM

        # 阿里云百炼配置
        self._dashscope_api_key: str = settings.DASHSCOPE_API_KEY
        self._dashscope_model: str = settings.DASHSCOPE_MODEL

        # Dify 平台配置
        self._dify_api_url: str = settings.DIFY_API_URL
        self._dify_api_key: str = settings.DIFY_API_KEY
        self._dify_response_mode: str = settings.DIFY_RESPONSE_MODE

        # 通用配置
        self._max_tokens: int = settings.LLM_MAX_TOKENS
        self._temperature: float = settings.LLM_TEMPERATURE
        self._timeout: float = settings.LLM_TIMEOUT

        # Token 日志回调函数（由 main.py 在启动时注入）
        self._token_log_callback: Optional[Any] = None

    def set_token_log_callback(self, callback: Any) -> None:
        """
        设置 Token 消耗日志回调函数

        参数:
            callback: 异步回调函数，接收 LLMTokenLog 对象作为参数
                     通常为 MySQLCRUD.save_llm_token_log 方法
        """
        self._token_log_callback = callback

    @retry(
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
        wait=wait_exponential(
            multiplier=1, min=settings.LLM_RETRY_DELAY, max=30
        ),
        retry=retry_if_exception_type((LLMClientError, TimeoutError)),
        reraise=True,
    )
    async def chat(
        self,
        messages: List[Dict[str, str]],
        agent_name: str = "unknown",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, str]] = None,
        files: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        调用大模型进行文本对话

        参数:
            messages: 对话消息列表，格式为 [{"role": "system", "content": "..."}, ...]
            agent_name: 调用方 Agent 名称，用于日志追踪和 Token 统计
            temperature: 温度参数（0-1），None 则使用默认值
            max_tokens: 最大输出 Token 数，None 则使用默认值
            response_format: 响应格式约束，如 {"type": "json_object"}（仅 DashScope 支持）
            files: 文件附件列表（仅 Dify 平台支持），格式为:
                   [{"type": "image", "transfer_method": "remote_url", "url": "https://..."}]

        返回:
            str: 大模型返回的文本内容

        异常:
            LLMClientError: 调用失败时抛出，会自动重试
        """
        if self._platform == "dify":
            return await self._chat_dify(
                messages=messages,
                agent_name=agent_name,
                temperature=temperature,
                max_tokens=max_tokens,
                files=files,
            )
        else:
            return await self._chat_dashscope(
                messages=messages,
                agent_name=agent_name,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

    async def _chat_dashscope(
        self,
        messages: List[Dict[str, str]],
        agent_name: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        response_format: Optional[Dict[str, str]],
    ) -> str:
        """
        通过阿里云百炼 DashScope SDK 调用大模型

        使用懒加载方式导入 dashscope，避免启动时的 asyncio 初始化问题
        """
        import dashscope
        dashscope.api_key = self._dashscope_api_key

        start_ms = time.time() * 1000
        request_id = str(uuid.uuid4())

        try:
            # 调用 DashScope Generation API
            response = dashscope.Generation.call(
                model=self._dashscope_model,
                messages=messages,
                result_format="message",
                temperature=temperature or self._temperature,
                max_tokens=max_tokens or self._max_tokens,
                timeout=self._timeout,
                **({"response_format": response_format} if response_format else {}),
            )

            latency_ms = time.time() * 1000 - start_ms

            # 检查响应状态
            if response.status_code != 200:
                logger.error(
                    "LLM call failed | request_id=%s | status=%s | code=%s | msg=%s",
                    request_id,
                    response.status_code,
                    response.code,
                    response.message,
                )
                raise LLMClientError(
                    f"LLM API error: code={response.code}, msg={response.message}"
                )

            # 提取 Token 使用量
            usage = response.usage
            prompt_tokens = usage.get("input_tokens", 0) if usage else 0
            completion_tokens = usage.get("output_tokens", 0) if usage else 0
            total_tokens = prompt_tokens + completion_tokens

            # 记录 Token 日志
            token_log = LLMTokenLog(
                request_id=request_id,
                agent_name=agent_name,
                model=self._dashscope_model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
            )
            await self._log_token_usage(token_log)

            # 提取响应内容
            content = response.output.choices[0].message.content
            logger.info(
                "LLM call success | request_id=%s | agent=%s | tokens=%d/%d | latency=%.0fms",
                request_id,
                agent_name,
                prompt_tokens,
                completion_tokens,
                latency_ms,
            )
            return content

        except LLMClientError:
            raise
        except Exception as exc:
            latency_ms = time.time() * 1000 - start_ms
            logger.error(
                "LLM call exception | request_id=%s | agent=%s | latency=%.0fms | error=%s",
                request_id,
                agent_name,
                latency_ms,
                str(exc),
            )
            raise LLMClientError(f"LLM call failed: {exc}") from exc

    async def _chat_dify(
        self,
        messages: List[Dict[str, str]],
        agent_name: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        files: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        通过 Dify 平台 HTTP API 调用大模型

        Dify API 规范（严格遵循公司级部署平台标准）：
        - 请求地址: {DIFY_API_URL}/chat-messages
        - 认证方式: Authorization: Bearer {api_key}
        - 请求体格式遵循 Dify Chat API 标准

        参数映射说明：
        - messages 中的 system 消息会被提取为 Dify 的 inputs 上下文
        - messages 中的最后一条 user 消息作为 Dify 的 query
        - 历史对话消息通过 conversation_id 机制传递
        - files 参数支持传入缺陷图片等附件进行多模态分析

        业务查询模板示例：
            用户输入: "不良品 DEF-2024-001，表面出现缩孔，请分析根因"
            Dify query: "不良品 DEF-2024-001，表面出现缩孔，请分析根因"
            （query 内容直接来自业务系统的实际不良品描述，而非固定模板）
        """
        import httpx

        start_ms = time.time() * 1000
        request_id = str(uuid.uuid4())

        # 提取 system prompt 和 user query
        system_content = ""
        user_query = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_query = msg["content"]

        # 如果没有明确的 user 消息，使用最后一条消息
        if not user_query and messages:
            user_query = messages[-1].get("content", "")

        # 构建 Dify API 请求体（严格遵循公司级部署平台规范）
        dify_payload: Dict[str, Any] = {
            "inputs": {
                "system_prompt": system_content,
                "temperature": temperature or self._temperature,
                "max_tokens": max_tokens or self._max_tokens,
            },
            "query": user_query,
            "response_mode": self._dify_response_mode,
            "conversation_id": "",
            "user": f"agent_{agent_name}",
        }

        # 如果提供了文件附件（如缺陷图片），添加到请求体
        if files:
            dify_payload["files"] = files

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._dify_api_url}/chat-messages",
                    headers={
                        "Authorization": f"Bearer {self._dify_api_key}",
                        "Content-Type": "application/json",
                    },
                    json=dify_payload,
                )

                latency_ms = time.time() * 1000 - start_ms

                if response.status_code != 200:
                    logger.error(
                        "Dify API call failed | request_id=%s | status=%d | body=%s",
                        request_id,
                        response.status_code,
                        response.text[:500],
                    )
                    raise LLMClientError(
                        f"Dify API error: status={response.status_code}, body={response.text[:200]}"
                    )

                result = response.json()

                # 提取 Dify 响应中的文本内容
                answer = result.get("answer", "")
                if not answer and "data" in result:
                    answer = result.get("data", {}).get("outputs", {}).get("text", "")

                # Dify 的 Token 使用量（如果API返回）
                metadata = result.get("metadata", {})
                usage_info = metadata.get("usage", {})
                prompt_tokens = usage_info.get("prompt_tokens", 0)
                completion_tokens = usage_info.get("completion_tokens", 0)
                total_tokens = usage_info.get("total_tokens", prompt_tokens + completion_tokens)

                # 记录 Token 日志
                token_log = LLMTokenLog(
                    request_id=request_id,
                    agent_name=agent_name,
                    model=f"dify-{self._dify_api_url}",
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=round(latency_ms, 2),
                )
                await self._log_token_usage(token_log)

                logger.info(
                    "Dify call success | request_id=%s | agent=%s | tokens=%d/%d | latency=%.0fms",
                    request_id,
                    agent_name,
                    prompt_tokens,
                    completion_tokens,
                    latency_ms,
                )

                if not answer:
                    raise LLMClientError("Dify returned empty answer")

                return answer

        except LLMClientError:
            raise
        except Exception as exc:
            latency_ms = time.time() * 1000 - start_ms
            logger.error(
                "Dify call exception | request_id=%s | agent=%s | latency=%.0fms | error=%s",
                request_id,
                agent_name,
                latency_ms,
                str(exc),
            )
            raise LLMClientError(f"Dify call failed: {exc}") from exc

    async def chat_json(
        self,
        messages: List[Dict[str, str]],
        agent_name: str = "unknown",
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        调用大模型并返回 JSON 结构化数据

        内部先调用 chat 方法获取文本响应，然后解析为 JSON。
        如果直接解析失败，会尝试从响应文本中提取 JSON 片段。

        参数:
            messages: 对话消息列表
            agent_name: 调用方 Agent 名称
            temperature: 温度参数

        返回:
            Dict[str, Any]: 解析后的 JSON 字典

        异常:
            LLMClientError: JSON 解析失败时抛出
        """
        # DashScope 平台使用原生 JSON 模式
        if self._platform == "dashscope":
            raw = await self.chat(
                messages=messages,
                agent_name=agent_name,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        else:
            # Dify 平台通过 system prompt 要求 JSON 输出
            raw = await self.chat(
                messages=messages,
                agent_name=agent_name,
                temperature=temperature,
            )

        # 尝试解析 JSON
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # 如果直接解析失败，尝试从文本中提取 JSON 片段
            json_start = raw.find("{")
            json_end = raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                try:
                    return json.loads(raw[json_start:json_end])
                except json.JSONDecodeError:
                    pass
            raise LLMClientError(f"Failed to parse LLM JSON response: {raw[:200]}")

    async def _log_token_usage(self, token_log: LLMTokenLog) -> None:
        """
        记录 Token 消耗日志

        优先使用回调函数（写入 MySQL），
        如果回调未设置则输出到日志文件
        """
        try:
            if self._token_log_callback:
                await self._token_log_callback(token_log)
            else:
                logger.debug(
                    "Token usage | agent=%s | prompt=%d | completion=%d | total=%d | latency=%.0fms",
                    token_log.agent_name,
                    token_log.prompt_tokens,
                    token_log.completion_tokens,
                    token_log.total_tokens,
                    token_log.latency_ms,
                )
        except Exception as exc:
            logger.warning("Failed to log token usage: %s", exc)


# ======================== 全局 LLM 客户端实例 ========================
llm_client = LLMClient()
"""
全局 LLM 客户端单例

在程序任意位置通过以下方式使用：
    from src.utils.llm_client import llm_client
    result = await llm_client.chat(messages=[...], agent_name="my_agent")

平台切换：
    在 .env 中设置 LLM_PLATFORM=dify 即可切换到 Dify 平台
"""
