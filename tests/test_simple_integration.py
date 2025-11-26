"""
Simple integration test for dependency injection without external dependencies.

Tests the basic integration of services through the dependency injection system.
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_dependency_injection():
    """Test basic dependency injection functionality."""
    logger.info("Testing basic dependency injection")
    
    try:
        # Import and test service container
        from app.core.dependencies import ServiceContainer
        
        # Create service container
        container = ServiceContainer()
        
        # Mock external dependencies to avoid import errors
        with patch('app.core.concurrency_manager.ConcurrencyManager'), \
             patch('app.core.lifecycle_manager.ModelLifecycleManager'), \
             patch('app.core.health_monitor.ModelHealthMonitor'), \
             patch('app.services.create_food_image_service') as mock_food_service:
            
            # Mock the food image service creation
            mock_food_service.return_value = MagicMock()
            
            # Initialize services
            await container.initialize_services()
            
            # Test that services are available
            assert container._initialized
            
            # Test model manager
            model_manager = container.get_model_manager()
            assert model_manager is not None
            
            # Test OCR service
            ocr_service = container.get_ocr_service()
            assert ocr_service is not None
            
            # Test translation service
            translation_service = container.get_translation_service()
            assert translation_service is not None
            
            # Test food image service
            food_image_service = container.get_food_image_service()
            assert food_image_service is not None
            
            # Test processing pipeline
            processing_pipeline = container.get_processing_pipeline()
            assert processing_pipeline is not None
            assert processing_pipeline.model_manager is model_manager
            assert processing_pipeline.ocr_service is ocr_service
            assert processing_pipeline.translation_service is translation_service
            assert processing_pipeline.food_image_service is food_image_service
            
            logger.info("✓ Basic dependency injection test passed")
            
            # Cleanup
            await container.cleanup_services()
            
    except Exception as e:
        logger.error(f"❌ Basic dependency injection test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_service_lifecycle():
    """Test service lifecycle management."""
    logger.info("Testing service lifecycle management")
    
    try:
        from app.core.dependencies import ServiceContainer
        
        container = ServiceContainer()
        
        # Mock external dependencies
        with patch('app.core.concurrency_manager.ConcurrencyManager'), \
             patch('app.core.lifecycle_manager.ModelLifecycleManager'), \
             patch('app.core.health_monitor.ModelHealthMonitor'), \
             patch('app.services.create_food_image_service') as mock_food_service:
            
            mock_food_service.return_value = MagicMock()
            
            # Test initialization
            assert not container._initialized
            await container.initialize_services()
            assert container._initialized
            
            # Test cleanup
            await container.cleanup_services()
            assert not container._initialized
            
            logger.info("✓ Service lifecycle test passed")
            
    except Exception as e:
        logger.error(f"❌ Service lifecycle test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def test_dependency_providers():
    """Test FastAPI dependency providers."""
    logger.info("Testing FastAPI dependency providers")
    
    try:
        from app.core.dependencies import (
            get_model_manager,
            get_ocr_service,
            get_translation_service,
            get_food_image_service,
            get_processing_pipeline
        )
        from fastapi import Request
        
        # Mock request with service container
        request = MagicMock(spec=Request)
        
        # Create and initialize service container
        from app.core.dependencies import ServiceContainer
        container = ServiceContainer()
        
        with patch('app.core.concurrency_manager.ConcurrencyManager'), \
             patch('app.core.lifecycle_manager.ModelLifecycleManager'), \
             patch('app.core.health_monitor.ModelHealthMonitor'), \
             patch('app.services.create_food_image_service') as mock_food_service:
            
            mock_food_service.return_value = MagicMock()
            
            await container.initialize_services()
            request.app.state.service_container = container
            
            # Test dependency providers
            model_manager = get_model_manager(container)
            assert model_manager is not None
            
            ocr_service = get_ocr_service(container)
            assert ocr_service is not None
            
            translation_service = get_translation_service(container)
            assert translation_service is not None
            
            food_image_service = get_food_image_service(container)
            assert food_image_service is not None
            
            processing_pipeline = get_processing_pipeline(container)
            assert processing_pipeline is not None
            
            logger.info("✓ Dependency providers test passed")
            
            # Cleanup
            await container.cleanup_services()
            
    except Exception as e:
        logger.error(f"❌ Dependency providers test failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def main():
    """Run all integration tests."""
    print("Running simple integration tests...")
    
    try:
        await test_basic_dependency_injection()
        await test_service_lifecycle()
        await test_dependency_providers()
        
        print("\n🎉 All simple integration tests passed!")
        print("✓ Service container initialization works")
        print("✓ Dependency injection chain works")
        print("✓ Service lifecycle management works")
        print("✓ FastAPI dependency providers work")
        
    except Exception as e:
        print(f"\n❌ Integration tests failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)