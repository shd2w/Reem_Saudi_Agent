from functools import lru_cache
from pydantic import Field, SecretStr, AnyHttpUrl
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

# Load environment variables from .env file (only once at import time)
load_dotenv()


class Settings(BaseSettings):
    # Application
    app_name: str = Field(default="Agent Orchestrator")
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_json: bool = Field(default=False)

    # External services
    agent_api_base_url: AnyHttpUrl | str = Field(default="https://sys.wajan-clinic.com/api", env="AGENT_API_URL")
    patient_db_api_url: AnyHttpUrl | str = Field(default="https://sys.wajan-clinic.com/api", env="PATIENT_DB_API_URL")
    wasender_base_url: AnyHttpUrl | str = Field(default="https://wasenderapi.com", env="WASENDER_API_URL")
    agent_login_url: AnyHttpUrl | str | None = Field(default="https://sys.wajan-clinic.com/api/auth/login", env="AGENT_LOGIN_URL")
    wasender_login_url: AnyHttpUrl | str | None = None

    # Redis / Celery
    redis_url: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Security
    api_key: str | None = Field(default=None, env="API_KEY")
    agent_api_user: str | None = Field(default="agent@gmail.com", env="AGENT_API_USER")
    agent_api_password: SecretStr | None = Field(default=SecretStr("123456789"), env="AGENT_API_PASSWORD")
    agent_api_token: str | None = Field(default=None, env="AGENT_API_TOKEN")
    wasender_api_key: SecretStr | None = Field(default=None, env="WASENDER_API_KEY")
    elevenlabs_api_key: SecretStr | None = Field(default=None, env="ELEVENLABS_API_KEY")
    
    # OpenAI
    openai_api_key: SecretStr | None = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    openai_timeout_seconds: int = Field(default=15, env="OPENAI_TIMEOUT_SECONDS")
    openai_max_retries: int = Field(default=1, env="OPENAI_MAX_RETRIES")
    
    # Adaptive Confidence Thresholds (Issue #10)
    enable_adaptive_confidence: bool = Field(default=True, env="ENABLE_ADAPTIVE_CONFIDENCE")
    confidence_min_samples: int = Field(default=10, env="CONFIDENCE_MIN_SAMPLES")
    confidence_adjustment_rate: float = Field(default=0.02, env="CONFIDENCE_ADJUSTMENT_RATE")
    # LangGraph Feature Flags (for gradual rollout)
    use_langgraph: bool = Field(default=False, env="USE_LANGGRAPH")
    langgraph_rollout_percentage: int = Field(default=0, env="LANGGRAPH_ROLLOUT_PERCENTAGE")
    langgraph_sessions: list[str] = Field(default_factory=list, env="LANGGRAPH_SESSIONS")
    langgraph_use_checkpointing: bool = Field(default=True, env="LANGGRAPH_USE_CHECKPOINTING")  # Issue #5: Enable Redis persistence
    
    # Intelligent Agent (NEW: LLM with Function Calling - TRUE AI)
    use_intelligent_agent: bool = Field(default=True, env="USE_INTELLIGENT_AGENT")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }

    # Convenience uppercase aliases to match docs/examples
    @property
    def AGENT_API_URL(self) -> str:
        return str(self.agent_api_base_url)

    @property
    def PATIENT_DB_API_URL(self) -> str:
        return str(self.patient_db_api_url)

    @property
    def AGENT_API_USER(self) -> str | None:
        return self.agent_api_user

    @property
    def AGENT_API_PASSWORD(self) -> str | None:
        return None if self.agent_api_password is None else self.agent_api_password.get_secret_value()

    @property
    def WASENDER_API_KEY(self) -> str | None:
        return None if self.wasender_api_key is None else self.wasender_api_key.get_secret_value()

    @property
    def ELEVENLABS_API_KEY(self) -> str | None:
        return None if self.elevenlabs_api_key is None else self.elevenlabs_api_key.get_secret_value()

    @property
    def REDIS_URL(self) -> str:
        return self.redis_url

    @property
    def LOG_LEVEL(self) -> str:
        return self.log_level

    @property
    def AWS_ACCESS_KEY_ID(self) -> str | None:
        return self.aws_access_key_id

    @property
    def AWS_SECRET_ACCESS_KEY(self) -> str | None:
        return self.aws_secret_access_key

    @property
    def AWS_S3_BUCKET(self) -> str | None:
        return self.aws_s3_bucket
    
    # LangGraph feature flags (uppercase aliases)
    @property
    def USE_LANGGRAPH(self) -> bool:
        return self.use_langgraph
    
    @property
    def LANGGRAPH_ROLLOUT_PERCENTAGE(self) -> int:
        return self.langgraph_rollout_percentage
    
    @property
    def LANGGRAPH_SESSIONS(self) -> list[str]:
        return self.langgraph_sessions
    
    @property
    def LANGGRAPH_USE_CHECKPOINTING(self) -> bool:
        return self.langgraph_use_checkpointing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if not settings.celery_broker_url:
        settings.celery_broker_url = settings.redis_url
    if not settings.celery_result_backend:
        settings.celery_result_backend = settings.redis_url
    return settings


# Expose a singleton-like instance for simple imports: from app.config import settings
settings = get_settings()

