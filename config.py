"""
Configuration management with strict validation and environment loading.
Implements type-safe settings using Pydantic Settings with comprehensive validation.
"""

import os
from typing import Optional, List
from pydantic import Field, field_validator, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration with strict validation.
    All settings are loaded from environment variables or .env file.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key for LLM access")
    openai_base_url: str = Field("https://api.openai.com/v1", description="OpenAI API base URL")
    openai_model: str = Field("gpt-4-turbo-preview", description="LLM model to use")
    openai_max_tokens: int = Field(4096, ge=100, le=128000, description="Maximum tokens for response")
    openai_temperature: float = Field(0.2, ge=0.0, le=2.0, description="Temperature for generation")
    
    # Analysis Configuration
    analysis_timeout: int = Field(120, ge=10, le=600, description="Analysis timeout in seconds")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    batch_size: int = Field(50, ge=1, le=500, description="Batch size for processing")
    confidence_threshold: float = Field(0.75, ge=0.0, le=1.0, description="Confidence threshold")
    
    # Data Sources
    default_user_agent: str = Field(
        "Mozilla/5.0 (compatible; NeuroResearchEngine/2.0)",
        description="Default User-Agent for HTTP requests"
    )
    request_timeout: int = Field(30, ge=5, le=120, description="HTTP request timeout")
    
    # Output Configuration
    output_format: str = Field("json", pattern="^(json|csv|html|markdown)$", description="Output format")
    enable_visualizations: bool = Field(True, description="Enable visualization generation")
    log_level: str = Field("INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$", description="Logging level")
    
    # Integration Hooks
    webhook_url: Optional[str] = Field(None, description="General webhook URL")
    slack_webhook: Optional[str] = Field(None, description="Slack webhook URL")
    discord_webhook: Optional[str] = Field(None, description="Discord webhook URL")
    
    @field_validator('openai_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "sk-your-api-key-here":
            raise ValueError("OPENAI_API_KEY must be set to a valid API key")
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
    
    @field_validator('openai_base_url')
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("OPENAI_BASE_URL must be a valid HTTP/HTTPS URL")
        return v.rstrip('/')
    
    @property
    def api_endpoint(self) -> str:
        """Construct full API endpoint for chat completions."""
        return f"{self.openai_base_url}/chat/completions"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return os.getenv("ENVIRONMENT", "development") == "production"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global settings
    settings = Settings()
    return settings
