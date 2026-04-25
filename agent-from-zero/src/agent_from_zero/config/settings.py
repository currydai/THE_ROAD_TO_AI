from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    model_provider: str = Field(default="openai", alias="MODEL_PROVIDER")
    use_fake_model: bool = Field(default=True, alias="USE_FAKE_MODEL")

    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="agent-from-zero", alias="LANGSMITH_PROJECT")

    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")
    report_dir: Path = Field(default=Path("data/processed/reports"), alias="REPORT_DIR")
    vectorstore_dir: Path = Field(default=Path("data/vectorstore/faiss"), alias="VECTORSTORE_DIR")
    allowed_file_root: Path = Field(default=Path("data"), alias="ALLOWED_FILE_ROOT")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    @property
    def has_openai_key(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
