from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "dev"
    frontend_origin: str = "http://localhost:5173"

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_password"

    oneke_base_url: str = ""
    classification_mode: str = "llm"
    require_real_oneke: bool = True

    openai_base_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""

    # Ollama embedding configuration
    ollama_base_url: str = "http://host.docker.internal:11434/v1"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    require_ollama_embedding: bool = False  # 默认不强制，允许降级到 OpenAI/Local

    sqlite_path: str = "app.db"
    neo4j_disabled: bool = False


settings = Settings()
