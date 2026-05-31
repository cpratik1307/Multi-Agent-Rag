"""Application settings loaded from .env via Pydantic BaseSettings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""

    # LangSmith tracing
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "multi-agent-rag"

    # Models
    gpt4o_model: str = "gpt-4o"
    fallback_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"

    # Retrieval
    faiss_index_path: str = "faiss_index"
    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5

    # Validation
    grounding_threshold: float = 0.7
    max_retries: int = 3


settings = Settings()
