"""
Configuration loader utility for environment-specific settings.
"""

import os
from pathlib import Path
from typing import Optional
from .settings import Settings, Environment


class ConfigLoader:
    """Utility class for loading environment-specific configurations"""
    
    @staticmethod
    def load_environment_config(environment: Optional[str] = None) -> Settings:
        """
        Load configuration for the specified environment.
        
        Args:
            environment: Target environment (development, staging, production, testing)
                        If None, uses ENVIRONMENT env var or defaults to development
        
        Returns:
            Settings instance with environment-specific configuration
        """
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")
        
        # Normalize environment name
        env = Environment(environment.lower())
        
        # Set environment-specific .env file
        env_file = f".env.{env.value}"
        env_file_path = Path(env_file)
        
        if env_file_path.exists():
            # Temporarily set the env file for this environment
            original_env_file = os.environ.get("ENV_FILE")
            os.environ["ENV_FILE"] = env_file
            
            try:
                # Create settings with environment-specific file
                settings = Settings(_env_file=env_file)
                return settings
            finally:
                # Restore original env file setting
                if original_env_file is not None:
                    os.environ["ENV_FILE"] = original_env_file
                elif "ENV_FILE" in os.environ:
                    del os.environ["ENV_FILE"]
        else:
            # Fallback to default settings if env file doesn't exist
            print(f"Warning: Environment file {env_file} not found, using default settings")
            return Settings()
    
    @staticmethod
    def get_available_environments() -> list[str]:
        """Get list of available environment configurations"""
        env_files = []
        for env_file in Path(".").glob(".env.*"):
            if env_file.name.startswith(".env."):
                env_name = env_file.name.replace(".env.", "")
                env_files.append(env_name)
        return sorted(env_files)
    
    @staticmethod
    def validate_environment_config(environment: str) -> bool:
        """
        Validate that an environment configuration exists and is valid.
        
        Args:
            environment: Environment name to validate
            
        Returns:
            True if configuration is valid, False otherwise
        """
        try:
            env = Environment(environment.lower())
            env_file = f".env.{env.value}"
            
            if not Path(env_file).exists():
                return False
            
            # Try to load the configuration
            settings = ConfigLoader.load_environment_config(environment)
            
            # Basic validation - check if required settings are present
            required_settings = [
                settings.app_name,
                settings.environment,
                settings.host,
                settings.port,
            ]
            
            return all(setting is not None for setting in required_settings)
            
        except (ValueError, Exception):
            return False
    
    @staticmethod
    def create_sample_env_file(environment: str, output_path: Optional[str] = None) -> str:
        """
        Create a sample .env file for the specified environment.
        
        Args:
            environment: Target environment
            output_path: Optional custom output path
            
        Returns:
            Path to the created sample file
        """
        env = Environment(environment.lower())
        
        if output_path is None:
            output_path = f".env.{env.value}.sample"
        
        # Load default settings to generate sample
        default_settings = Settings()
        
        sample_content = f"""# Sample configuration for {env.value} environment
# Copy this file to .env.{env.value} and modify as needed

# Application Configuration
APP_NAME={default_settings.app_name}
APP_VERSION={default_settings.app_version}
ENVIRONMENT={env.value}
DEBUG={'true' if env == Environment.DEVELOPMENT else 'false'}

# Server Configuration
HOST={default_settings.host}
PORT={default_settings.port}
RELOAD={'true' if env == Environment.DEVELOPMENT else 'false'}
WORKERS={1 if env == Environment.DEVELOPMENT else 4}

# Logging Configuration
LOG_LEVEL={default_settings.log_level.value}

# Model Configuration
MODEL_OCR_CONFIDENCE_THRESHOLD={default_settings.models.ocr_confidence_threshold}
MODEL_OCR_MAX_IMAGE_SIZE_MB={default_settings.models.ocr_max_image_size_mb}
MODEL_TRANSLATION_BATCH_SIZE={default_settings.models.translation_batch_size}
MODEL_TRANSLATION_TIMEOUT_SECONDS={default_settings.models.translation_timeout_seconds}

# Concurrency Configuration
CONCURRENCY_MAX_CONCURRENT_REQUESTS={default_settings.concurrency.max_concurrent_requests}
CONCURRENCY_QUEUE_TIMEOUT_SECONDS={default_settings.concurrency.queue_timeout_seconds}
CONCURRENCY_PROCESSING_TIMEOUT_SECONDS={default_settings.concurrency.processing_timeout_seconds}
CONCURRENCY_MEMORY_LIMIT_MB={default_settings.concurrency.memory_limit_mb}

# Redis Configuration
REDIS_HOST={default_settings.redis.host}
REDIS_PORT={default_settings.redis.port}
REDIS_DB={default_settings.redis.db}
REDIS_MAX_CONNECTIONS={default_settings.redis.max_connections}
REDIS_CACHE_TTL_SECONDS={default_settings.redis.cache_ttl_seconds}

# Security Configuration
SECURITY_API_KEYS=your-api-key-here
SECURITY_RATE_LIMIT_REQUESTS_PER_MINUTE={default_settings.security.rate_limit_requests_per_minute}
SECURITY_CORS_ORIGINS=*

# Food Image Service Configuration
FOOD_IMAGE_SERVICE_URL={default_settings.food_images.service_url}
FOOD_IMAGE_API_KEY=your-food-image-api-key
FOOD_IMAGE_MAX_IMAGES_PER_ITEM={default_settings.food_images.max_images_per_item}

# Storage Configuration
TEMP_STORAGE_PATH={default_settings.temp_storage_path}
MAX_TEMP_FILE_AGE_HOURS={default_settings.max_temp_file_age_hours}
"""
        
        with open(output_path, "w") as f:
            f.write(sample_content)
        
        return output_path


def load_config_for_environment(environment: Optional[str] = None) -> Settings:
    """Convenience function to load configuration for an environment"""
    return ConfigLoader.load_environment_config(environment)