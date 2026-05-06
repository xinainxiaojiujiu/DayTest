"""
================================================================================
 汽车涂装不良品智能分析系统 - 配置中心
================================================================================
 功能说明：
   - 加载 .env 环境变量文件中的配置参数
   - 提供类型安全的配置访问接口
   - 支持 MySQL、InfluxDB、ChromaDB、Redis 等数据库连接配置
   - 支持阿里云百炼（DashScope）和 Dify 双平台 LLM 调用配置
   - 支持 MES、安川机器人、环境传感器等外部系统对接配置

 新手提示：
   - 所有配置项都有默认值，首次使用无需修改
   - 只需在 .env 文件中设置 DASHSCOPE_API_KEY 或 DIFY_API_KEY 即可启动
   - 数据库等外部服务未连接时，系统会自动使用模拟数据运行
================================================================================
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Literal


class Settings(BaseSettings):
    """应用全局配置类，自动从 .env 文件和环境变量加载配置"""

    # ======================== 应用基础配置 ========================
    APP_NAME: str = "AutomotiveCoatingDefectAnalysis"
    """应用名称，用于日志标识和API文档标题"""

    DEBUG: bool = False
    """调试模式开关，开启后输出详细日志，关闭后仅输出INFO级别"""

    # ======================== LLM 平台选择 ========================
    LLM_PLATFORM: Literal["dashscope", "dify"] = "dashscope"
    """LLM平台选择：dashscope=阿里云百炼, dify=Dify公司级部署平台"""

    # ======================== MySQL 数据库配置 ========================
    MYSQL_HOST: str = Field(default="127.0.0.1", alias="MYSQL_HOST")
    """MySQL 服务器地址，默认本地"""

    MYSQL_PORT: int = Field(default=3306, alias="MYSQL_PORT")
    """MySQL 端口号，默认 3306"""

    MYSQL_USER: str = Field(default="root", alias="MYSQL_USER")
    """MySQL 用户名"""

    MYSQL_PASSWORD: str = Field(default="changeme", alias="MYSQL_PASSWORD")
    """MySQL 密码"""

    MYSQL_DATABASE: str = Field(default="coating_defect", alias="MYSQL_DATABASE")
    """MySQL 数据库名"""

    MYSQL_POOL_SIZE: int = 10
    """MySQL 连接池大小"""

    MYSQL_MAX_OVERFLOW: int = 20
    """MySQL 连接池最大溢出连接数"""

    # ======================== InfluxDB 时序数据库配置 ========================
    INFLUX_URL: str = Field(default="http://127.0.0.1:8086", alias="INFLUX_URL")
    """InfluxDB 服务地址"""

    INFLUX_TOKEN: str = Field(default="changeme", alias="INFLUX_TOKEN")
    """InfluxDB 认证 Token"""

    INFLUX_ORG: str = Field(default="coating", alias="INFLUX_ORG")
    """InfluxDB 组织名称"""

    INFLUX_BUCKET: str = Field(default="sensor_data", alias="INFLUX_BUCKET")
    """InfluxDB 存储桶名称"""

    # ======================== ChromaDB 向量数据库配置 ========================
    CHROMA_HOST: str = Field(default="127.0.0.1", alias="CHROMA_HOST")
    """ChromaDB 服务器地址"""

    CHROMA_PORT: int = Field(default=8000, alias="CHROMA_PORT")
    """ChromaDB 端口号"""

    CHROMA_COLLECTION: str = Field(default="process_knowledge", alias="CHROMA_COLLECTION")
    """ChromaDB 集合名称，用于存储工艺知识库"""

    # ======================== Redis 缓存配置 ========================
    REDIS_HOST: str = Field(default="127.0.0.1", alias="REDIS_HOST")
    """Redis 服务器地址"""

    REDIS_PORT: int = Field(default=6379, alias="REDIS_PORT")
    """Redis 端口号"""

    REDIS_PASSWORD: Optional[str] = Field(default=None, alias="REDIS_PASSWORD")
    """Redis 密码（可选）"""

    REDIS_DB: int = 0
    """Redis 数据库编号"""

    CELERY_BROKER_URL: str = Field(
        default="redis://127.0.0.1:6379/0", alias="CELERY_BROKER_URL"
    )
    """Celery 消息代理地址"""

    CELERY_RESULT_BACKEND: str = Field(
        default="redis://127.0.0.1:6379/1", alias="CELERY_RESULT_BACKEND"
    )
    """Celery 结果后端地址"""

    # ======================== 阿里云百炼（DashScope）配置 ========================
    DASHSCOPE_API_KEY: str = Field(default="changeme", alias="DASHSCOPE_API_KEY")
    """阿里云百炼 API Key，从阿里云控制台获取"""

    DASHSCOPE_MODEL: str = Field(default="qwen-max", alias="DASHSCOPE_MODEL")
    """阿里云百炼模型名称，可选: qwen-max, qwen-plus, qwen-turbo"""

    # ======================== Dify 平台配置 ========================
    DIFY_API_URL: str = Field(
        default="https://aicenter.dongfeng-nissan.com.cn/v1",
        alias="DIFY_API_URL"
    )
    """Dify 平台 API 服务器地址"""

    DIFY_API_KEY: str = Field(default="changeme", alias="DIFY_API_KEY")
    """Dify 平台 API Key，从 Dify 控制台获取"""

    DIFY_RESPONSE_MODE: Literal["streaming", "blocking"] = "streaming"
    """Dify 响应模式：streaming=流式返回, blocking=阻塞返回"""

    # ======================== LLM 通用配置 ========================
    LLM_MAX_RETRIES: int = 3
    """LLM 调用最大重试次数"""

    LLM_RETRY_DELAY: float = 2.0
    """LLM 调用重试初始延迟（秒）"""

    LLM_TIMEOUT: float = 60.0
    """LLM 调用超时时间（秒）"""

    LLM_MAX_TOKENS: int = 4096
    """LLM 单次调用最大输出 Token 数"""

    LLM_TEMPERATURE: float = 0.1
    """LLM 温度参数（0-1），越低越确定性，越高越随机"""

    # ======================== MES 系统对接配置 ========================
    MES_API_BASE_URL: str = Field(
        default="http://mes.internal/api/v1", alias="MES_API_BASE_URL"
    )
    """MES 系统 API 基础地址"""

    MES_API_KEY: str = Field(default="changeme", alias="MES_API_KEY")
    """MES 系统 API Key"""

    MES_TIMEOUT: float = 30.0
    """MES 接口超时时间（秒）"""

    # ======================== 安川机器人对接配置 ========================
    ROBOT_HOST: str = Field(default="192.168.1.100", alias="ROBOT_HOST")
    """安川机器人控制器 IP 地址"""

    ROBOT_PORT: int = Field(default=10000, alias="ROBOT_PORT")
    """安川机器人通讯端口"""

    ROBOT_TIMEOUT: float = 10.0
    """机器人通讯超时时间（秒）"""

    # ======================== 环境传感器配置（Modbus TCP） ========================
    SENSOR_HOST: str = Field(default="192.168.1.200", alias="SENSOR_HOST")
    """环境传感器 Modbus TCP 网关 IP 地址"""

    SENSOR_PORT: int = Field(default=502, alias="SENSOR_PORT")
    """Modbus TCP 端口号，默认 502"""

    SENSOR_TIMEOUT: float = 5.0
    """传感器通讯超时时间（秒）"""

    SENSOR_TIME_WINDOW_SECONDS: int = 300
    """传感器数据采集时间窗口（秒），默认取不良发生前5分钟"""

    # ======================== OCR 配置 ========================
    OCR_USE_GPU: bool = False
    """OCR 是否使用 GPU 加速，默认使用 CPU"""

    OCR_LANG: str = "ch"
    """OCR 识别语言，ch=中文"""

    class Config:
        """Pydantic Settings 配置"""
        env_file = ".env"
        """环境变量文件路径"""
        env_file_encoding = "utf-8"
        """环境变量文件编码"""
        case_sensitive = True
        """环境变量名大小写敏感"""


# ======================== 全局配置实例 ========================
settings = Settings()
"""全局配置单例，在程序任意位置通过 `from src.config import settings` 引用"""
