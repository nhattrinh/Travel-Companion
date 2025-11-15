"""
Integration test for model manager and health monitoring infrastructure.
Tests Requirements 1.2, 1.3, 1.4, 6.3, 6.4 implementation.
"""

import asyncio
import pytest
from datetime import datetime
from app.core import ModelManager, ModelHealthMonitor, ModelLifecycleManager, MockModelImplementation
from app.models.internal_models import ModelType, ModelStatus, ModelConfig


async def test_model_manager_basic_operations():
    """Test basic model manager operations."""
    print("Testing ModelManager basic operations...")
    
    # Create model manager
    manager = ModelManager()
    
    # Create mock model
    mock_model = MockModelImplementation(ModelType.OCR, {"config_path": "test"})
    
    # Test model registration
    config = ModelConfig(
        model_type=ModelType.OCR,
        config_path="test_config.json",
        max_failures=3,
        initialization_timeout_seconds=10
    )
    
    success = await manager.register_model(ModelType.OCR, mock_model, config)
    assert success, "Model registration should succeed"
    
    # Test model retrieval
    retrieved_model = await manager.get_model(ModelType.OCR)
    assert retrieved_model is not None, "Should retrieve registered model"
    assert retrieved_model.status == ModelStatus.READY, "Model should be ready"
    
    # Test model status
    status = await manager.get_model_status(ModelType.OCR)
    assert status == ModelStatus.READY, "Model status should be READY"
    
    # Test hot swap
    new_mock_model = MockModelImplementation(ModelType.OCR, {"config_path": "test2"})
    swap_success = await manager.hot_swap_model(ModelType.OCR, new_mock_model, config)
    assert swap_success, "Hot swap should succeed"
    
    # Verify new model is active
    swapped_model = await manager.get_model(ModelType.OCR)
    assert swapped_model is not None, "Should have a model after swap"
    assert swapped_model.status == ModelStatus.READY, "Swapped model should be ready"
    
    # Test cleanup
    await manager.cleanup_all_models()
    
    print("✓ ModelManager basic operations test passed")


async def test_health_monitoring():
    """Test health monitoring functionality."""
    print("Testing health monitoring...")
    
    # Create components
    manager = ModelManager()
    health_monitor = ModelHealthMonitor(check_interval_seconds=1, failure_threshold=2)
    
    # Register a mock model
    mock_model = MockModelImplementation(ModelType.TRANSLATION, {"config_path": "test"})
    config = ModelConfig(
        model_type=ModelType.TRANSLATION,
        config_path="test_config.json",
        health_check_interval_seconds=1
    )
    
    await manager.register_model(ModelType.TRANSLATION, mock_model, config)
    
    # Start health monitoring
    await health_monitor.start_monitoring(manager)
    
    # Wait for a few health checks
    await asyncio.sleep(2.5)
    
    # Check metrics
    metrics = health_monitor.get_model_metrics(ModelType.TRANSLATION)
    assert metrics is not None, "Should have health metrics"
    assert metrics.total_checks > 0, "Should have performed health checks"
    assert metrics.get_success_rate() > 0, "Should have successful health checks"
    
    # Get system health
    system_health = health_monitor.get_system_health()
    assert system_health.overall_status in ["healthy", "degraded", "unhealthy"], "Should have valid status"
    
    # Stop monitoring and cleanup
    await health_monitor.stop_monitoring()
    await manager.cleanup_all_models()
    
    print("✓ Health monitoring test passed")


async def test_lifecycle_management():
    """Test complete lifecycle management."""
    print("Testing lifecycle management...")
    
    # Create components
    manager = ModelManager()
    health_monitor = ModelHealthMonitor(check_interval_seconds=2)
    lifecycle_manager = ModelLifecycleManager(manager, health_monitor)
    
    # Define models configuration
    models_config = {
        ModelType.OCR: {
            "config_path": "ocr_config.json",
            "max_failures": 3,
            "initialization_timeout": 10,
            "health_check_interval": 2
        },
        ModelType.TRANSLATION: {
            "config_path": "translation_config.json", 
            "max_failures": 3,
            "initialization_timeout": 10,
            "health_check_interval": 2
        }
    }
    
    # Test system initialization
    success = await lifecycle_manager.initialize_system(models_config)
    assert success, "System initialization should succeed"
    
    # Verify models are registered and healthy
    ocr_model = await manager.get_model(ModelType.OCR)
    translation_model = await manager.get_model(ModelType.TRANSLATION)
    
    assert ocr_model is not None, "OCR model should be registered"
    assert translation_model is not None, "Translation model should be registered"
    assert ocr_model.status == ModelStatus.READY, "OCR model should be ready"
    assert translation_model.status == ModelStatus.READY, "Translation model should be ready"
    
    # Test model restart
    restart_success = await lifecycle_manager.restart_model(ModelType.OCR)
    assert restart_success, "Model restart should succeed"
    
    # Verify model is still working after restart
    restarted_model = await manager.get_model(ModelType.OCR)
    assert restarted_model is not None, "Model should be available after restart"
    assert restarted_model.status == ModelStatus.READY, "Model should be ready after restart"
    
    # Test system status
    status = lifecycle_manager.get_system_status()
    assert "uptime_seconds" in status, "Should include uptime"
    assert "model_manager_healthy" in status, "Should include manager health"
    
    # Test graceful shutdown
    await lifecycle_manager.graceful_shutdown(timeout_seconds=5)
    
    print("✓ Lifecycle management test passed")


async def test_error_handling():
    """Test error handling and failure scenarios."""
    print("Testing error handling...")
    
    manager = ModelManager()
    
    # Test handling of non-existent model
    non_existent_model = await manager.get_model(ModelType.NAVIGATION)
    assert non_existent_model is None, "Should return None for non-existent model"
    
    # Test model failure handling
    mock_model = MockModelImplementation(ModelType.NAVIGATION, {"config_path": "test"})
    
    # Simulate initialization failure by overriding initialize method
    async def failing_initialize():
        raise Exception("Simulated initialization failure")
    
    mock_model.initialize = failing_initialize
    
    config = ModelConfig(
        model_type=ModelType.NAVIGATION,
        config_path="test_config.json",
        max_failures=1
    )
    
    success = await manager.register_model(ModelType.NAVIGATION, mock_model, config)
    assert not success, "Registration should fail with failing model"
    
    # Verify model is marked as failed
    status = await manager.get_model_status(ModelType.NAVIGATION)
    assert status == ModelStatus.FAILED, "Model should be marked as failed"
    
    print("✓ Error handling test passed")


async def main():
    """Run all tests."""
    print("Starting model infrastructure tests...\n")
    
    try:
        await test_model_manager_basic_operations()
        await test_health_monitoring()
        await test_lifecycle_management()
        await test_error_handling()
        
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())