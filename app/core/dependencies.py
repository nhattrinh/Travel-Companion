"""
Dependency injection setup for FastAPI.
Provides dependency providers for core services with lifecycle management.
Implements Requirements 1.1, 1.2, 1.3 for service dependency injection.
"""

from fastapi import Depends, Request, HTTPException
from typing import Optional, Dict, Any
import logging
import asyncio
from functools import lru_cache

from app.core.models import ModelManager
from app.core.processing_pipeline import ProcessingPipeline
from app.core.concurrency_manager import ConcurrencyManager
from app.core.lifecycle_manager import ModelLifecycleManager
from app.core.health_monitor import ModelHealthMonitor
from app.config.settings import settings
from app.services import (
    OCRService,
    TranslationService,
    FoodImageService,
    create_food_image_service
)


logger = logging.getLogger(__name__)


class ServiceContainer:
    """
    Container for managing application services with lifecycle management.
    Implements Requirements 1.1, 1.2, 1.3 for centralized service management.
    """
    
    def __init__(self):
        self._model_manager: Optional[ModelManager] = None
        self._ocr_service: Optional[OCRService] = None
        self._translation_service: Optional[TranslationService] = None
        self._food_image_service: Optional[FoodImageService] = None
        self._processing_pipeline: Optional[ProcessingPipeline] = None
        self._concurrency_manager: Optional[ConcurrencyManager] = None
        self._lifecycle_manager: Optional[ModelLifecycleManager] = None
        self._health_monitor: Optional[ModelHealthMonitor] = None
        self._initialized = False
        self._initialization_lock = asyncio.Lock()
    
    async def initialize_services(self) -> None:
        """
        Initialize all services with proper dependency order.
        Implements Requirement 1.1 - service lifecycle management.
        """
        async with self._initialization_lock:
            if self._initialized:
                return
            
            logger.info("Initializing service container")
            
            try:
                # Initialize core services first
                self._model_manager = ModelManager()
                self._health_monitor = ModelHealthMonitor()
                self._lifecycle_manager = ModelLifecycleManager(
                    self._model_manager, 
                    self._health_monitor
                )
                
                # Initialize concurrency manager
                from app.models.internal_models import ConcurrencyConfig
                concurrency_config = ConcurrencyConfig(
                    max_concurrent_requests=settings.concurrency.max_concurrent_requests,
                    queue_timeout_seconds=30,
                    processing_timeout_seconds=120,
                    memory_limit_mb=1024
                )
                self._concurrency_manager = ConcurrencyManager(concurrency_config)
                
                # Initialize business services
                self._ocr_service = OCRService()  # Uses default mock model
                self._translation_service = TranslationService()  # Uses default mock model
                self._food_image_service = await create_food_image_service()
                
                # Initialize processing pipeline with all services
                self._processing_pipeline = ProcessingPipeline(
                    model_manager=self._model_manager,
                    ocr_service=self._ocr_service,
                    translation_service=self._translation_service,
                    food_image_service=self._food_image_service
                )
                
                self._initialized = True
                logger.info("Service container initialization completed")
                
            except Exception as e:
                logger.error(f"Service container initialization failed: {e}", exc_info=True)
                raise
    
    async def cleanup_services(self) -> None:
        """
        Cleanup all services in reverse dependency order.
        Implements Requirement 1.1 - service lifecycle management.
        """
        logger.info("Cleaning up service container")
        
        try:
            # Cleanup in reverse order
            if self._lifecycle_manager:
                await self._lifecycle_manager.graceful_shutdown()
            
            if self._concurrency_manager:
                await self._concurrency_manager.shutdown()
            
            if self._food_image_service:
                # FoodImageService doesn't have cleanup method, just set to None
                pass
            
            # Reset all services
            self._processing_pipeline = None
            self._food_image_service = None
            self._translation_service = None
            self._ocr_service = None
            self._concurrency_manager = None
            self._lifecycle_manager = None
            self._health_monitor = None
            self._model_manager = None
            
            logger.info("Service container cleanup completed")
            
        except Exception as e:
            logger.error(f"Service container cleanup failed: {e}", exc_info=True)
        finally:
            # Always reset initialized flag
            self._initialized = False
    
    def get_model_manager(self) -> ModelManager:
        """Get model manager instance."""
        if not self._initialized or self._model_manager is None:
            raise RuntimeError("Service container not initialized")
        return self._model_manager
    
    def get_ocr_service(self) -> OCRService:
        """Get OCR service instance."""
        if not self._initialized or self._ocr_service is None:
            raise RuntimeError("Service container not initialized")
        return self._ocr_service
    
    def get_translation_service(self) -> TranslationService:
        """Get translation service instance."""
        if not self._initialized or self._translation_service is None:
            raise RuntimeError("Service container not initialized")
        return self._translation_service
    
    def get_food_image_service(self) -> FoodImageService:
        """Get food image service instance."""
        if not self._initialized or self._food_image_service is None:
            raise RuntimeError("Service container not initialized")
        return self._food_image_service
    
    def get_processing_pipeline(self) -> ProcessingPipeline:
        """Get processing pipeline instance."""
        if not self._initialized or self._processing_pipeline is None:
            raise RuntimeError("Service container not initialized")
        return self._processing_pipeline
    
    def get_concurrency_manager(self) -> ConcurrencyManager:
        """Get concurrency manager instance."""
        if not self._initialized or self._concurrency_manager is None:
            raise RuntimeError("Service container not initialized")
        return self._concurrency_manager
    
    def get_lifecycle_manager(self) -> ModelLifecycleManager:
        """Get lifecycle manager instance."""
        if not self._initialized or self._lifecycle_manager is None:
            raise RuntimeError("Service container not initialized")
        return self._lifecycle_manager
    
    def get_health_monitor(self) -> ModelHealthMonitor:
        """Get health monitor instance."""
        if not self._initialized or self._health_monitor is None:
            raise RuntimeError("Service container not initialized")
        return self._health_monitor


# Global service container
service_container = ServiceContainer()


def get_service_container(request: Request) -> ServiceContainer:
    """
    Get the service container from application state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        ServiceContainer: Application service container
        
    Raises:
        HTTPException: If service container is not available
    """
    if not hasattr(request.app.state, 'service_container'):
        logger.error("Service container not initialized")
        raise HTTPException(
            status_code=500,
            detail="Service container not available"
        )
    
    return request.app.state.service_container


def get_model_manager(
    container: ServiceContainer = Depends(get_service_container)
) -> ModelManager:
    """
    Dependency provider for ModelManager.
    Implements Requirement 1.2 - model manager dependency injection.
    
    Args:
        container: Service container instance
        
    Returns:
        ModelManager: Application model manager instance
        
    Raises:
        HTTPException: If model manager is not available
    """
    try:
        return container.get_model_manager()
    except RuntimeError as e:
        logger.error(f"Model manager not available: {e}")
        raise HTTPException(
            status_code=500,
            detail="Model manager not available"
        )


def get_ocr_service(
    container: ServiceContainer = Depends(get_service_container)
) -> OCRService:
    """
    Dependency provider for OCRService.
    Implements Requirement 1.2 - OCR service dependency injection.
    
    Args:
        container: Service container instance
        
    Returns:
        OCRService: Configured OCR service instance
        
    Raises:
        HTTPException: If OCR service is not available
    """
    try:
        return container.get_ocr_service()
    except RuntimeError as e:
        logger.error(f"OCR service not available: {e}")
        raise HTTPException(
            status_code=500,
            detail="OCR service not available"
        )


def get_translation_service(
    container: ServiceContainer = Depends(get_service_container)
) -> TranslationService:
    """
    Dependency provider for TranslationService.
    Implements Requirement 1.2 - translation service dependency injection.
    
    Args:
        container: Service container instance
        
    Returns:
        TranslationService: Configured translation service instance
        
    Raises:
        HTTPException: If translation service is not available
    """
    try:
        return container.get_translation_service()
    except RuntimeError as e:
        logger.error(f"Translation service not available: {e}")
        raise HTTPException(
            status_code=500,
            detail="Translation service not available"
        )


def get_food_image_service(
    container: ServiceContainer = Depends(get_service_container)
) -> FoodImageService:
    """
    Dependency provider for FoodImageService.
    Implements Requirement 1.2 - food image service dependency injection.
    
    Args:
        container: Service container instance
        
    Returns:
        FoodImageService: Configured food image service instance
        
    Raises:
        HTTPException: If food image service is not available
    """
    try:
        return container.get_food_image_service()
    except RuntimeError as e:
        logger.error(f"Food image service not available: {e}")
        raise HTTPException(
            status_code=500,
            detail="Food image service not available"
        )


def get_processing_pipeline(
    container: ServiceContainer = Depends(get_service_container)
) -> ProcessingPipeline:
    """
    Dependency provider for ProcessingPipeline.
    Implements Requirement 1.2 - processing pipeline dependency injection.
    
    Args:
        container: Service container instance
        
    Returns:
        ProcessingPipeline: Configured processing pipeline instance
        
    Raises:
        HTTPException: If processing pipeline is not available
    """
    try:
        return container.get_processing_pipeline()
    except RuntimeError as e:
        logger.error(f"Processing pipeline not available: {e}")
        raise HTTPException(
            status_code=500,
            detail="Processing pipeline not available"
        )


def get_concurrency_manager(
    container: ServiceContainer = Depends(get_service_container)
) -> ConcurrencyManager:
    """
    Dependency provider for ConcurrencyManager.
    Implements Requirement 1.2 - concurrency manager dependency injection.
    
    Args:
        container: Service container instance
        
    Returns:
        ConcurrencyManager: Configured concurrency manager instance
        
    Raises:
        HTTPException: If concurrency manager is not available
    """
    try:
        return container.get_concurrency_manager()
    except RuntimeError as e:
        logger.error(f"Concurrency manager not available: {e}")
        raise HTTPException(
            status_code=500,
            detail="Concurrency manager not available"
        )


def get_lifecycle_manager(
    container: ServiceContainer = Depends(get_service_container)
) -> ModelLifecycleManager:
    """
    Dependency provider for ModelLifecycleManager.
    Implements Requirement 1.3 - lifecycle manager dependency injection.
    
    Args:
        container: Service container instance
        
    Returns:
        ModelLifecycleManager: Configured lifecycle manager instance
        
    Raises:
        HTTPException: If lifecycle manager is not available
    """
    try:
        return container.get_lifecycle_manager()
    except RuntimeError as e:
        logger.error(f"Lifecycle manager not available: {e}")
        raise HTTPException(
            status_code=500,
            detail="Lifecycle manager not available"
        )


def get_health_monitor(
    container: ServiceContainer = Depends(get_service_container)
) -> ModelHealthMonitor:
    """
    Dependency provider for ModelHealthMonitor.
    Implements Requirement 1.3 - health monitor dependency injection.
    
    Args:
        container: Service container instance
        
    Returns:
        ModelHealthMonitor: Configured health monitor instance
        
    Raises:
        HTTPException: If health monitor is not available
    """
    try:
        return container.get_health_monitor()
    except RuntimeError as e:
        logger.error(f"Health monitor not available: {e}")
        raise HTTPException(
            status_code=500,
            detail="Health monitor not available"
        )


def get_request_id(request: Request) -> str:
    """
    Get request ID from request state.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Request ID
    """
    return getattr(request.state, 'request_id', 'unknown')


async def verify_api_key(request: Request) -> Optional[str]:
    """
    Verify API key if authentication is required.
    
    Args:
        request: FastAPI request object
        
    Returns:
        Optional[str]: API key if valid, None if not required
        
    Raises:
        HTTPException: If API key is required but invalid/missing
    """
    if not settings.require_api_key:
        return None
    
    api_key = request.headers.get(settings.api_key_header)
    
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required"
        )
    
    # TODO: Implement actual API key validation in later tasks
    # For now, just return the key
    return api_key


async def get_current_user(request: Request):
    """
    Get the current authenticated user from the request.
    
    This is a placeholder that will be fully implemented in auth endpoints.
    For now, returns a mock user for development.
    
    Args:
        request: The FastAPI request object
        
    Returns:
        User object if authenticated
        
    Raises:
        HTTPException: If not authenticated
    """
    from app.core.security import verify_token
    from app.models.user import User
    from app.core.db import SessionLocal
    
    # Get token from Authorization header
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        # For development, return a mock user if no auth header
        # In production, this should raise 401
        mock_user = User(id=1, email="dev@example.com", hashed_password="")
        return mock_user
    
    token = auth_header.replace("Bearer ", "")
    payload = verify_token(token)
    
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication token"
        )
    
    # Get user from database
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == payload["sub"]).first()
        if not user:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )
        return user
    finally:
        db.close()
