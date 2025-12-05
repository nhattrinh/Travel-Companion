"""
Configuration management system using Pydantic Settings.
Supports environment-based configuration for different deployment environments.
"""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List
from enum import Enum
import os
from pathlib import Path


class Environment(str, Enum):
    """Supported deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Supported log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelSettings(BaseSettings):
    """Model-specific configuration settings"""
    
    # OCR Model Configuration
    ocr_model_path: str = Field(default="models/ocr", description="Path to OCR model files")
    ocr_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    ocr_max_image_size_mb: int = Field(default=10, ge=1, le=50)
    
    # Translation Model Configuration
    translation_model_path: str = Field(default="models/translation", description="Path to translation model files")
    translation_batch_size: int = Field(default=32, ge=1, le=128)
    translation_timeout_seconds: int = Field(default=30, ge=5, le=300)
    
    # Navigation Model Configuration
    navigation_model_path: str = Field(default="models/navigation", description="Path to navigation model files")
    
    class Config:
        env_prefix = "MODEL_"


class ConcurrencySettings(BaseSettings):
    """Concurrency and performance configuration"""
    
    max_concurrent_requests: int = Field(default=10, ge=1, le=100)
    queue_timeout_seconds: int = Field(default=30, ge=5, le=300)
    processing_timeout_seconds: int = Field(default=120, ge=30, le=600)
    memory_limit_mb: int = Field(default=1024, ge=256, le=8192)
    max_batch_size: int = Field(default=10, ge=1, le=50)
    
    class Config:
        env_prefix = "CONCURRENCY_"


class RedisSettings(BaseSettings):
    """Redis cache configuration"""
    
    host: str = Field(default="redis")  # Default to docker service name
    port: int = Field(default=6379, ge=1, le=65535)
    password: Optional[str] = Field(default=None)
    db: int = Field(default=0, ge=0, le=15)
    max_connections: int = Field(default=20, ge=1, le=100)
    socket_timeout: int = Field(default=5, ge=1, le=30)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)  # 1 hour default
    
    @property
    def url(self) -> str:
        """Generate Redis URL from configuration"""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"
    
    model_config = {"env_prefix": "REDIS_"}


class SecuritySettings(BaseSettings):
    """Security and authentication configuration"""
    
    api_key_header: str = Field(default="X-API-Key")
    api_keys: List[str] = Field(default_factory=list)
    rate_limit_requests_per_minute: int = Field(default=60, ge=1, le=1000)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = Field(default=True)
    cors_allow_methods: List[str] = Field(
        default_factory=lambda: ["GET", "POST"]
    )
    cors_allow_headers: List[str] = Field(default_factory=lambda: ["*"])
    
    @field_validator('api_keys', mode='before')
    @classmethod
    def parse_api_keys(cls, v):
        """Parse API keys from environment variable or list"""
        if isinstance(v, str):
            return [key.strip() for key in v.split(",") if key.strip()]
        return v or []
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from environment variable or list"""
        if isinstance(v, str):
            origin_list = v.split(",")
            return [origin.strip() for origin in origin_list if origin.strip()]
        return v or ["*"]
    
    model_config = {"env_prefix": "SECURITY_"}


class FoodImageSettings(BaseSettings):
    """Food image service configuration"""
    
    service_url: str = Field(default="https://api.foodimages.com")
    api_key: Optional[str] = Field(default=None)
    max_images_per_item: int = Field(default=3, ge=1, le=10)
    search_timeout_seconds: int = Field(default=10, ge=1, le=60)
    placeholder_image_url: str = Field(default="/static/images/food-placeholder.jpg")
    cache_enabled: bool = Field(default=True)
    
    model_config = {"env_prefix": "FOOD_IMAGE_"}


class UnsplashSettings(BaseSettings):
    """Unsplash API configuration"""
    
    access_key: Optional[str] = Field(
        default=None,
        description="Unsplash API access key (Client-ID)"
    )
    secret_key: Optional[str] = Field(
        default=None,
        description="Unsplash API secret key (for OAuth flows)"
    )
    api_url: str = Field(default="https://api.unsplash.com")
    max_images_per_query: int = Field(default=3, ge=1, le=30)
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    
    model_config = {
        "env_prefix": "UNSPLASH_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


class Settings(BaseSettings):
    """Main application settings"""
    
    # Application Configuration
    app_name: str = Field(default="Menu Translation Backend")
    app_version: str = Field(default="1.0.0")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # API Key Configuration
    require_api_key: bool = Field(default=False, description="Whether to require API key for authentication")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = Field(default=False)
    workers: int = Field(default=1, ge=1, le=16)
    
    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    log_file: Optional[str] = Field(default=None)
    log_rotation: str = Field(default="1 day")
    log_retention: str = Field(default="30 days")
    
    # Storage Configuration
    temp_storage_path: str = Field(default="temp")
    max_temp_file_age_hours: int = Field(default=24, ge=1, le=168)
    cleanup_interval_minutes: int = Field(default=60, ge=5, le=1440)
    
    # Nested Settings
    models: ModelSettings = Field(default_factory=ModelSettings)
    concurrency: ConcurrencySettings = Field(
        default_factory=ConcurrencySettings
    )
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    food_images: FoodImageSettings = Field(default_factory=FoodImageSettings)
    unsplash: UnsplashSettings = Field(default_factory=UnsplashSettings)
    
    @field_validator('environment', mode='before')
    @classmethod
    def validate_environment(cls, v):
        """Validate and normalize environment setting"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    def get_temp_storage_path(self) -> Path:
        """Get absolute path for temporary storage"""
        return Path(self.temp_storage_path).resolve()
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == Environment.PRODUCTION
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == Environment.DEVELOPMENT
    
    def get_cors_config(self) -> Dict[str, Any]:
        """Get CORS configuration for FastAPI"""
        return {
            "allow_origins": self.security.cors_origins,
            "allow_credentials": self.security.cors_allow_credentials,
            "allow_methods": self.security.cors_allow_methods,
            "allow_headers": self.security.cors_allow_headers,
        }
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",
    }


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment and files"""
    global settings
    settings = Settings()
    return settings