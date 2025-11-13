"""
Centralized configuration management for AI Assistant.
Loads settings from environment variables with defaults.
"""

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # === Application Settings ===
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # === LLM Configuration (Claude Code) ===
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    claude_model: str = Field(
        default="claude-3-5-sonnet-20241022", alias="CLAUDE_MODEL"
    )
    claude_temperature: float = Field(default=0.0, alias="CLAUDE_TEMPERATURE")
    claude_max_tokens: int = Field(default=4096, alias="CLAUDE_MAX_TOKENS")
    claude_timeout: int = Field(default=60, alias="CLAUDE_TIMEOUT")
    claude_max_retries: int = Field(default=3, alias="CLAUDE_MAX_RETRIES")

    # === Database Configuration ===
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="testdb", alias="POSTGRES_DB")
    postgres_user: str = Field(default="testuser", alias="POSTGRES_USER")
    postgres_password: str = Field(default="testpass", alias="POSTGRES_PASSWORD")

    # === SQLite Configuration ===
    sqlite_db_path: str = Field(
        default="./data/conversations.db", alias="SQLITE_DB_PATH"
    )

    # === Vector Store Configuration ===
    chroma_persist_directory: str = Field(
        default="./data/chroma", alias="CHROMA_PERSIST_DIRECTORY"
    )
    chroma_collection_name: str = Field(default="documents", alias="CHROMA_COLLECTION_NAME")

    # === Embedding Configuration ===
    embedding_model_name: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2", alias="EMBEDDING_MODEL_NAME"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=100, alias="EMBEDDING_BATCH_SIZE")

    # === Budget Monitoring ===
    budget_limit: float = Field(default=100.00, alias="BUDGET_LIMIT")
    alert_email: str = Field(default="admin@company.com", alias="ALERT_EMAIL")
    alert_thresholds: str = Field(default="80.0,90.0,100.0", alias="ALERT_THRESHOLDS")

    # === Performance Settings ===
    max_concurrent_queries: int = Field(default=10, alias="MAX_CONCURRENT_QUERIES")
    rate_limit_per_user: int = Field(default=10, alias="RATE_LIMIT_PER_USER")

    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def alert_thresholds_list(self) -> List[float]:
        """Parse alert thresholds from comma-separated string."""
        return [float(x.strip()) for x in self.alert_thresholds.split(",")]

    class Config:
        env_file = ".env"
        case_sensitive = False


# Singleton instance
settings = Settings()
