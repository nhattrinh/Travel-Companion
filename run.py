#!/usr/bin/env python3
"""
Application startup script with environment configuration support.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the app directory to the Python path
sys.path.append(str(Path(__file__).parent / "app"))

from config.loader import ConfigLoader, load_config_for_environment


def main():
    """Main startup function with environment configuration"""
    parser = argparse.ArgumentParser(description="Menu Translation Backend Server")
    parser.add_argument(
        "--env", 
        choices=["development", "staging", "production", "testing"],
        default=None,
        help="Environment to run (default: from ENVIRONMENT env var or development)"
    )
    parser.add_argument(
        "--host",
        default=None,
        help="Host to bind to (overrides config)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (overrides config)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (overrides config)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (overrides config)"
    )
    parser.add_argument(
        "--list-envs",
        action="store_true",
        help="List available environment configurations"
    )
    parser.add_argument(
        "--validate-env",
        help="Validate a specific environment configuration"
    )
    parser.add_argument(
        "--create-sample",
        help="Create a sample .env file for the specified environment"
    )
    
    args = parser.parse_args()
    
    # Handle utility commands
    if args.list_envs:
        envs = ConfigLoader.get_available_environments()
        print("Available environment configurations:")
        for env in envs:
            print(f"  - {env}")
        return
    
    if args.validate_env:
        is_valid = ConfigLoader.validate_environment_config(args.validate_env)
        if is_valid:
            print(f"✓ Environment '{args.validate_env}' configuration is valid")
        else:
            print(f"✗ Environment '{args.validate_env}' configuration is invalid or missing")
            sys.exit(1)
        return
    
    if args.create_sample:
        try:
            sample_file = ConfigLoader.create_sample_env_file(args.create_sample)
            print(f"✓ Sample configuration created: {sample_file}")
        except Exception as e:
            print(f"✗ Failed to create sample configuration: {e}")
            sys.exit(1)
        return
    
    # Load configuration for the specified environment
    try:
        settings = load_config_for_environment(args.env)
        print(f"✓ Loaded configuration for environment: {settings.environment.value}")
    except Exception as e:
        print(f"✗ Failed to load configuration: {e}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.workers:
        settings.workers = args.workers
    if args.reload:
        settings.reload = True
    if args.debug:
        settings.debug = True
    
    # Validate configuration
    if not ConfigLoader.validate_environment_config(settings.environment.value):
        print(f"✗ Invalid configuration for environment: {settings.environment.value}")
        sys.exit(1)
    
    # Create temp storage directory if it doesn't exist
    temp_path = settings.get_temp_storage_path()
    temp_path.mkdir(parents=True, exist_ok=True)
    
    # Print startup information
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    print(f"   Environment: {settings.environment.value}")
    print(f"   Host: {settings.host}")
    print(f"   Port: {settings.port}")
    print(f"   Workers: {settings.workers}")
    print(f"   Debug: {settings.debug}")
    print(f"   Reload: {settings.reload}")
    print(f"   Log Level: {settings.log_level.value}")
    
    # Start the server
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        workers=settings.workers if not settings.reload else 1,
        log_level=settings.log_level.value.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()