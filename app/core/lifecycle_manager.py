"""
Model lifecycle management service.
Implements Requirements 6.3, 6.4 for model initialization and cleanup procedures.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from contextlib import asynccontextmanager

from ..models.internal_models import ModelType, ModelStatus, ModelConfig
from .models import BaseModel, ModelManager
from .health_monitor import ModelHealthMonitor


class ModelLifecycleManager:
    """
    Manages the complete lifecycle of AI models including initialization,
    monitoring, and cleanup procedures.
    Implements Requirements 6.3, 6.4 for lifecycle management.
    """
    
    def __init__(self, model_manager: ModelManager, health_monitor: ModelHealthMonitor):
        self.model_manager = model_manager
        self.health_monitor = health_monitor
        self._initialization_callbacks: List[Callable[[ModelType, bool], None]] = []
        self._cleanup_callbacks: List[Callable[[ModelType], None]] = []
        self._shutdown_handlers_registered = False
        self._logger = logging.getLogger(__name__)
        self._startup_time = datetime.utcnow()
    
    def add_initialization_callback(self, callback: Callable[[ModelType, bool], None]) -> None:
        """
        Add callback to be notified when models are initialized.
        
        Args:
            callback: Function to call with (model_type, success) parameters
        """
        self._initialization_callbacks.append(callback)
    
    def add_cleanup_callback(self, callback: Callable[[ModelType], None]) -> None:
        """
        Add callback to be notified when models are cleaned up.
        
        Args:
            callback: Function to call with model_type parameter
        """
        self._cleanup_callbacks.append(callback)
    
    async def initialize_system(self, models_config: Dict[ModelType, Dict[str, Any]]) -> bool:
        """
        Initialize the complete model system.
        Implements Requirement 6.3 - initialization procedures.
        
        Args:
            models_config: Configuration for each model type
            
        Returns:
            bool: True if system initialization successful
        """
        self._logger.info("Starting model system initialization")
        
        try:
            # Register shutdown handlers
            self._register_shutdown_handlers()
            
            # Initialize models sequentially to avoid resource conflicts
            initialization_results = {}
            
            for model_type, config in models_config.items():
                self._logger.info(f"Initializing {model_type.value} model")
                
                try:
                    # Create model instance based on config
                    model_instance = await self._create_model_instance(model_type, config)
                    
                    if model_instance:
                        # Create model config
                        model_config = ModelConfig(
                            model_type=model_type,
                            config_path=config.get("config_path", ""),
                            weights_path=config.get("weights_path"),
                            max_failures=config.get("max_failures", 3),
                            initialization_timeout_seconds=config.get("initialization_timeout", 300),
                            health_check_interval_seconds=config.get("health_check_interval", 60)
                        )
                        
                        # Register with model manager
                        success = await self.model_manager.register_model(
                            model_type, model_instance, model_config
                        )
                        
                        initialization_results[model_type] = success
                        
                        # Notify callbacks
                        for callback in self._initialization_callbacks:
                            try:
                                callback(model_type, success)
                            except Exception as e:
                                self._logger.error(f"Initialization callback error: {e}", exc_info=True)
                        
                        if success:
                            self._logger.info(f"Successfully initialized {model_type.value} model")
                        else:
                            self._logger.error(f"Failed to initialize {model_type.value} model")
                    else:
                        initialization_results[model_type] = False
                        self._logger.error(f"Failed to create {model_type.value} model instance")
                        
                except Exception as e:
                    initialization_results[model_type] = False
                    self._logger.error(f"Exception during {model_type.value} model initialization: {e}", exc_info=True)
            
            # Start health monitoring if any models were initialized successfully
            successful_initializations = sum(1 for success in initialization_results.values() if success)
            
            if successful_initializations > 0:
                await self.health_monitor.start_monitoring(self.model_manager)
                self._logger.info(f"System initialization completed. {successful_initializations}/{len(models_config)} models initialized successfully")
                return True
            else:
                self._logger.error("System initialization failed. No models were initialized successfully")
                return False
                
        except Exception as e:
            self._logger.error(f"System initialization failed with exception: {e}", exc_info=True)
            return False
    
    async def _create_model_instance(self, model_type: ModelType, config: Dict[str, Any]) -> Optional[BaseModel]:
        """
        Create a model instance based on configuration.
        Implements Requirement 6.3 - model instantiation.
        
        Args:
            model_type: Type of model to create
            config: Model configuration
            
        Returns:
            Optional[BaseModel]: Created model instance or None if failed
        """
        try:
            # This is a factory method that would create specific model implementations
            # For now, we'll create a mock implementation for demonstration
            
            if model_type == ModelType.OCR:
                from ..services.ocr_service import OCRModelImplementation
                return OCRModelImplementation(config)
            elif model_type == ModelType.TRANSLATION:
                from ..services.translation_service import TranslationModelImplementation
                return TranslationModelImplementation(config)
            elif model_type == ModelType.NAVIGATION:
                from ..services.navigation_service import NavigationModelImplementation
                return NavigationModelImplementation(config)
            else:
                self._logger.error(f"Unknown model type: {model_type}")
                return None
                
        except ImportError as e:
            self._logger.warning(f"Model implementation not available for {model_type.value}: {e}")
            # Return a mock implementation for testing
            return MockModelImplementation(model_type, config)
        except Exception as e:
            self._logger.error(f"Failed to create {model_type.value} model instance: {e}", exc_info=True)
            return None
    
    async def graceful_shutdown(self, timeout_seconds: int = 30) -> None:
        """
        Perform graceful shutdown of the model system.
        Implements Requirement 6.3 - cleanup procedures.
        
        Args:
            timeout_seconds: Maximum time to wait for shutdown
        """
        self._logger.info("Starting graceful shutdown")
        
        try:
            # Stop health monitoring first
            await self.health_monitor.stop_monitoring()
            
            # Get list of active models for cleanup callbacks
            active_models = []
            for model_type in ModelType:
                model = await self.model_manager.get_model(model_type)
                if model:
                    active_models.append(model_type)
            
            # Cleanup all models with timeout
            cleanup_task = asyncio.create_task(self.model_manager.cleanup_all_models())
            
            try:
                await asyncio.wait_for(cleanup_task, timeout=timeout_seconds)
                self._logger.info("Model cleanup completed successfully")
            except asyncio.TimeoutError:
                self._logger.warning(f"Model cleanup timed out after {timeout_seconds} seconds")
                cleanup_task.cancel()
            
            # Notify cleanup callbacks
            for model_type in active_models:
                for callback in self._cleanup_callbacks:
                    try:
                        callback(model_type)
                    except Exception as e:
                        self._logger.error(f"Cleanup callback error: {e}", exc_info=True)
            
            self._logger.info("Graceful shutdown completed")
            
        except Exception as e:
            self._logger.error(f"Error during graceful shutdown: {e}", exc_info=True)
    
    def _register_shutdown_handlers(self) -> None:
        """
        Register signal handlers for graceful shutdown.
        Implements Requirement 6.3 - cleanup procedures.
        """
        if self._shutdown_handlers_registered:
            return
        
        def signal_handler(signum, frame):
            self._logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.graceful_shutdown())
        
        # Register handlers for common shutdown signals
        if sys.platform != "win32":
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGINT, signal_handler)
        
        self._shutdown_handlers_registered = True
        self._logger.info("Registered shutdown signal handlers")
    
    async def restart_model(self, model_type: ModelType, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Restart a specific model.
        Implements Requirement 6.3 - model restart procedures.
        
        Args:
            model_type: Type of model to restart
            config: Optional new configuration
            
        Returns:
            bool: True if restart successful
        """
        self._logger.info(f"Restarting {model_type.value} model")
        
        try:
            # Get current model for cleanup
            current_model = await self.model_manager.get_model(model_type)
            
            # Create new model instance
            if config:
                new_model = await self._create_model_instance(model_type, config)
            else:
                # Use existing configuration
                existing_config = self.model_manager._model_configs.get(model_type)
                if existing_config:
                    model_config = {
                        "config_path": existing_config.config_path,
                        "weights_path": existing_config.weights_path,
                        "max_failures": existing_config.max_failures,
                        "initialization_timeout": existing_config.initialization_timeout_seconds,
                        "health_check_interval": existing_config.health_check_interval_seconds
                    }
                    new_model = await self._create_model_instance(model_type, model_config)
                else:
                    self._logger.error(f"No configuration found for {model_type.value} model")
                    return False
            
            if not new_model:
                self._logger.error(f"Failed to create new {model_type.value} model instance")
                return False
            
            # Perform hot swap
            success = await self.model_manager.hot_swap_model(model_type, new_model)
            
            if success:
                self._logger.info(f"Successfully restarted {model_type.value} model")
                
                # Notify initialization callbacks
                for callback in self._initialization_callbacks:
                    try:
                        callback(model_type, True)
                    except Exception as e:
                        self._logger.error(f"Restart callback error: {e}", exc_info=True)
            else:
                self._logger.error(f"Failed to restart {model_type.value} model")
            
            return success
            
        except Exception as e:
            self._logger.error(f"Exception during {model_type.value} model restart: {e}", exc_info=True)
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status.
        Implements Requirement 6.4 - status monitoring.
        
        Returns:
            Dict[str, Any]: System status information
        """
        uptime = int((datetime.utcnow() - self._startup_time).total_seconds())
        
        # Get system health and convert to dict if needed
        system_health = self.health_monitor.get_system_health()
        health_dict = {
            "overall_status": system_health.overall_status,
            "uptime_seconds": system_health.uptime_seconds,
            "last_updated": system_health.last_updated.isoformat(),
        }
        
        return {
            "uptime_seconds": uptime,
            "startup_time": self._startup_time.isoformat(),
            "model_manager_healthy": self.model_manager.is_healthy(),
            "health_monitoring_active": self.health_monitor._running,
            "system_health": health_dict,
            "health_summary": self.health_monitor.get_health_summary()
        }
    
    @asynccontextmanager
    async def managed_lifecycle(self, models_config: Dict[ModelType, Dict[str, Any]]):
        """
        Context manager for complete lifecycle management.
        Implements Requirement 6.3 - lifecycle management.
        
        Args:
            models_config: Configuration for each model type
        """
        try:
            # Initialize system
            success = await self.initialize_system(models_config)
            if not success:
                raise RuntimeError("Failed to initialize model system")
            
            yield self
            
        finally:
            # Ensure cleanup happens
            await self.graceful_shutdown()


class MockModelImplementation(BaseModel):
    """
    Mock model implementation for testing and development.
    Implements the BaseModel interface for demonstration purposes.
    """
    
    def __init__(self, model_type: ModelType, config: Dict[str, Any]):
        super().__init__()
        self.model_type = model_type
        self.config_dict = config
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Mock initialization that always succeeds after a short delay."""
        await asyncio.sleep(0.1)  # Simulate initialization time
        self._initialized = True
        self._logger.info(f"Mock {self.model_type.value} model initialized")
        return True
    
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """Mock processing that returns dummy results."""
        if not self._initialized:
            raise RuntimeError("Model not initialized")
        
        await asyncio.sleep(0.05)  # Simulate processing time
        
        return {
            "model_type": self.model_type.value,
            "input_received": str(input_data)[:100],  # Truncate for logging
            "mock_result": f"Processed by mock {self.model_type.value} model",
            "processing_time_ms": 50
        }
    
    async def health_check(self) -> bool:
        """Mock health check that usually succeeds."""
        if not self._initialized:
            return False
        
        await asyncio.sleep(0.01)  # Simulate health check time
        return True  # Always healthy for mock
    
    async def cleanup(self) -> None:
        """Mock cleanup."""
        self._initialized = False
        self._logger.info(f"Mock {self.model_type.value} model cleaned up")