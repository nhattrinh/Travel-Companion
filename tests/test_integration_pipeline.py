"""
Integration test for the complete processing pipeline with dependency injection.

Tests the full integration of all services through the dependency injection system:
- Service container initialization
- Model manager integration
- Processing pipeline workflow
- Error handling and monitoring
- Concurrency management

Implements Requirements 6.1, 6.2, 6.3, 6.4 verification.
"""

import pytest
import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import create_app
from app.core.dependencies import service_container
from app.models.internal_models import ModelType, ConcurrencyConfig
from app.models.api_models import SupportedLanguage, MenuProcessingRequest


# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestProcessingPipelineIntegration:
    """Test complete processing pipeline integration."""
    
    @pytest.fixture
    async def app(self):
        """Create test FastAPI application with service container."""
        app = create_app()
        
        # Initialize service container for testing
        await service_container.initialize_services()
        app.state.service_container = service_container
        
        yield app
        
        # Cleanup
        await service_container.cleanup_services()
    
    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_service_container_initialization(self):
        """Test that service container initializes all services properly."""
        logger.info("Testing service container initialization")
        
        # Initialize service container
        await service_container.initialize_services()
        
        try:
            # Verify all services are available
            assert service_container._initialized
            
            # Test model manager
            model_manager = service_container.get_model_manager()
            assert model_manager is not None
            
            # Test OCR service
            ocr_service = service_container.get_ocr_service()
            assert ocr_service is not None
            assert ocr_service.model_manager is model_manager
            
            # Test translation service
            translation_service = service_container.get_translation_service()
            assert translation_service is not None
            assert translation_service.model_manager is model_manager
            
            # Test food image service
            food_image_service = service_container.get_food_image_service()
            assert food_image_service is not None
            
            # Test processing pipeline
            processing_pipeline = service_container.get_processing_pipeline()
            assert processing_pipeline is not None
            assert processing_pipeline.model_manager is model_manager
            assert processing_pipeline.ocr_service is ocr_service
            assert processing_pipeline.translation_service is translation_service
            assert processing_pipeline.food_image_service is food_image_service
            
            # Test concurrency manager
            concurrency_manager = service_container.get_concurrency_manager()
            assert concurrency_manager is not None
            
            # Test lifecycle manager
            lifecycle_manager = service_container.get_lifecycle_manager()
            assert lifecycle_manager is not None
            
            # Test health monitor
            health_monitor = service_container.get_health_monitor()
            assert health_monitor is not None
            
            logger.info("Service container initialization test passed")
            
        finally:
            await service_container.cleanup_services()
    
    @pytest.mark.asyncio
    async def test_dependency_injection_chain(self):
        """Test that dependency injection properly chains services."""
        logger.info("Testing dependency injection chain")
        
        await service_container.initialize_services()
        
        try:
            # Get processing pipeline through dependency chain
            processing_pipeline = service_container.get_processing_pipeline()
            
            # Verify all dependencies are properly injected
            assert processing_pipeline.model_manager is not None
            assert processing_pipeline.ocr_service is not None
            assert processing_pipeline.translation_service is not None
            assert processing_pipeline.food_image_service is not None
            
            # Verify services share the same model manager instance
            model_manager = service_container.get_model_manager()
            assert processing_pipeline.model_manager is model_manager
            assert processing_pipeline.ocr_service.model_manager is model_manager
            assert processing_pipeline.translation_service.model_manager is model_manager
            
            logger.info("Dependency injection chain test passed")
            
        finally:
            await service_container.cleanup_services()
    
    @pytest.mark.asyncio
    async def test_processing_pipeline_integration(self):
        """Test complete processing pipeline integration."""
        logger.info("Testing processing pipeline integration")
        
        await service_container.initialize_services()
        
        try:
            processing_pipeline = service_container.get_processing_pipeline()
            
            # Create test request
            request = MenuProcessingRequest(
                target_language=SupportedLanguage.SPANISH,
                source_language=SupportedLanguage.ENGLISH,
                include_images=True,
                max_images_per_item=3
            )
            
            # Mock image data
            test_image_data = b"fake_image_data"
            test_request_id = "test_request_123"
            
            # Mock the individual service methods to avoid actual model calls
            with patch.object(processing_pipeline.ocr_service, 'validate_image') as mock_validate, \
                 patch.object(processing_pipeline.ocr_service, 'preprocess_image') as mock_preprocess, \
                 patch.object(processing_pipeline.ocr_service, 'extract_text') as mock_extract, \
                 patch.object(processing_pipeline.ocr_service, 'filter_low_confidence_results') as mock_filter, \
                 patch.object(processing_pipeline.ocr_service, 'group_menu_items') as mock_group, \
                 patch.object(processing_pipeline.translation_service, 'detect_language') as mock_detect, \
                 patch.object(processing_pipeline.translation_service, 'batch_translate') as mock_translate, \
                 patch.object(processing_pipeline.food_image_service, 'get_most_relevant_images') as mock_images:
                
                # Setup mocks
                from app.models.internal_models import OCRResult, TranslationResult, FoodImage
                
                mock_validate.return_value = "validated_image"
                mock_preprocess.return_value = "preprocessed_image"
                
                # Mock OCR results
                ocr_results = [
                    OCRResult(text="Pizza Margherita", confidence=0.9, bbox=(10, 10, 100, 30)),
                    OCRResult(text="Pasta Carbonara", confidence=0.8, bbox=(10, 40, 100, 60))
                ]
                mock_extract.return_value = ocr_results
                mock_filter.return_value = ocr_results
                mock_group.return_value = {"group1": ocr_results}
                
                # Mock translation
                mock_detect.return_value = SupportedLanguage.ENGLISH
                translation_results = [
                    TranslationResult(translated_text="Pizza Margarita", source_language="en", confidence=0.9),
                    TranslationResult(translated_text="Pasta Carbonara", source_language="en", confidence=0.8)
                ]
                mock_translate.return_value = translation_results
                
                # Mock food images
                food_images = [
                    FoodImage(url="http://example.com/pizza.jpg", description="Pizza", relevance_score=0.9),
                    FoodImage(url="http://example.com/pasta.jpg", description="Pasta", relevance_score=0.8)
                ]
                mock_images.return_value = food_images
                
                # Process through pipeline
                response = await processing_pipeline.process_menu_image(
                    image_data=test_image_data,
                    request=request,
                    request_id=test_request_id
                )
                
                # Verify response
                assert response is not None
                assert response.request_id == test_request_id
                assert response.success is True
                assert len(response.menu_items) == 2
                assert response.menu_items[0].original_text == "Pizza Margherita"
                assert response.menu_items[0].translated_text == "Pizza Margarita"
                assert response.menu_items[1].original_text == "Pasta Carbonara"
                assert response.menu_items[1].translated_text == "Pasta Carbonara"
                
                # Verify all service methods were called
                mock_validate.assert_called_once()
                mock_preprocess.assert_called_once()
                mock_extract.assert_called_once()
                mock_filter.assert_called_once()
                mock_group.assert_called_once()
                mock_detect.assert_called_once()
                mock_translate.assert_called_once()
                mock_images.assert_called()
                
                logger.info("Processing pipeline integration test passed")
        
        finally:
            await service_container.cleanup_services()
    
    @pytest.mark.asyncio
    async def test_concurrency_manager_integration(self):
        """Test concurrency manager integration with processing pipeline."""
        logger.info("Testing concurrency manager integration")
        
        await service_container.initialize_services()
        
        try:
            concurrency_manager = service_container.get_concurrency_manager()
            processing_pipeline = service_container.get_processing_pipeline()
            
            # Test concurrent request processing
            async def mock_processing_func():
                await asyncio.sleep(0.1)  # Simulate processing time
                return {"result": "processed"}
            
            # Process multiple requests concurrently
            tasks = []
            for i in range(3):
                task = concurrency_manager.process_request(
                    request_id=f"test_request_{i}",
                    processing_func=mock_processing_func
                )
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks)
            
            # Verify all requests were processed
            assert len(results) == 3
            for result in results:
                assert result["result"] == "processed"
            
            logger.info("Concurrency manager integration test passed")
        
        finally:
            await service_container.cleanup_services()
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self):
        """Test error handling throughout the integrated system."""
        logger.info("Testing error handling integration")
        
        await service_container.initialize_services()
        
        try:
            processing_pipeline = service_container.get_processing_pipeline()
            
            # Create test request
            request = MenuProcessingRequest(
                target_language=SupportedLanguage.SPANISH,
                source_language=SupportedLanguage.ENGLISH,
                include_images=True,
                max_images_per_item=3
            )
            
            test_image_data = b"fake_image_data"
            test_request_id = "test_error_request"
            
            # Test OCR failure handling
            with patch.object(processing_pipeline.ocr_service, 'validate_image') as mock_validate:
                from app.services.exceptions import ImageValidationError
                mock_validate.side_effect = ImageValidationError("Invalid image format")
                
                try:
                    await processing_pipeline.process_menu_image(
                        image_data=test_image_data,
                        request=request,
                        request_id=test_request_id
                    )
                    assert False, "Expected ImageValidationError"
                except ImageValidationError:
                    pass  # Expected
            
            # Test graceful degradation with translation failure
            with patch.object(processing_pipeline.ocr_service, 'validate_image') as mock_validate, \
                 patch.object(processing_pipeline.ocr_service, 'preprocess_image') as mock_preprocess, \
                 patch.object(processing_pipeline.ocr_service, 'extract_text') as mock_extract, \
                 patch.object(processing_pipeline.ocr_service, 'filter_low_confidence_results') as mock_filter, \
                 patch.object(processing_pipeline.ocr_service, 'group_menu_items') as mock_group, \
                 patch.object(processing_pipeline.translation_service, 'detect_language') as mock_detect, \
                 patch.object(processing_pipeline.translation_service, 'batch_translate') as mock_translate:
                
                from app.models.internal_models import OCRResult, TranslationResult
                from app.services.exceptions import TranslationFailureError
                
                # Setup successful OCR
                mock_validate.return_value = "validated_image"
                mock_preprocess.return_value = "preprocessed_image"
                ocr_results = [OCRResult(text="Pizza", confidence=0.9, bbox=(10, 10, 100, 30))]
                mock_extract.return_value = ocr_results
                mock_filter.return_value = ocr_results
                mock_group.return_value = {"group1": ocr_results}
                mock_detect.return_value = SupportedLanguage.ENGLISH
                
                # Setup translation failure
                mock_translate.side_effect = TranslationFailureError("Translation service unavailable")
                
                # Process should succeed with graceful degradation
                response = await processing_pipeline.process_menu_image(
                    image_data=test_image_data,
                    request=request,
                    request_id=test_request_id
                )
                
                # Verify graceful degradation
                assert response is not None
                assert response.success is True  # Should still succeed
                assert len(response.menu_items) == 1
                assert response.menu_items[0].original_text == "Pizza"
                assert response.menu_items[0].translated_text == "Pizza"  # Original text returned
                assert response.menu_items[0].error_details == "Translation service unavailable"
            
            logger.info("Error handling integration test passed")
        
        finally:
            await service_container.cleanup_services()
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test health monitoring integration."""
        logger.info("Testing health monitoring integration")
        
        await service_container.initialize_services()
        
        try:
            health_monitor = service_container.get_health_monitor()
            lifecycle_manager = service_container.get_lifecycle_manager()
            model_manager = service_container.get_model_manager()
            
            # Test health monitoring
            system_health = health_monitor.get_system_health()
            assert system_health is not None
            
            # Test lifecycle manager status
            system_status = lifecycle_manager.get_system_status()
            assert system_status is not None
            
            # Test model manager health
            assert model_manager.is_healthy()
            
            logger.info("Health monitoring integration test passed")
        
        finally:
            await service_container.cleanup_services()


if __name__ == "__main__":
    # Run integration tests
    async def run_tests():
        test_instance = TestProcessingPipelineIntegration()
        
        print("Running integration tests...")
        
        try:
            await test_instance.test_service_container_initialization()
            print("✓ Service container initialization test passed")
            
            await test_instance.test_dependency_injection_chain()
            print("✓ Dependency injection chain test passed")
            
            await test_instance.test_processing_pipeline_integration()
            print("✓ Processing pipeline integration test passed")
            
            await test_instance.test_concurrency_manager_integration()
            print("✓ Concurrency manager integration test passed")
            
            await test_instance.test_error_handling_integration()
            print("✓ Error handling integration test passed")
            
            await test_instance.test_health_monitoring_integration()
            print("✓ Health monitoring integration test passed")
            
            print("\n🎉 All integration tests passed!")
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the tests
    asyncio.run(run_tests())