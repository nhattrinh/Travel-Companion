"""
Base interfaces and abstract classes for the pluggable model architecture.
Implements Requirements 1.2, 1.3, 1.4 for model management and 6.3, 6.4 for health monitoring.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
from enum import Enum
import asyncio
import logging
import time
from datetime import datetime, timedelta
from ..models.internal_models import ModelType, ModelStatus, ModelConfig


class BaseModel(ABC):
    """
    Abstract base class for all AI models in the system.
    Provides the interface for pluggable model architecture.
    Implements Requirements 1.2, 1.3, 1.4 for model interface.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.status = ModelStatus.INITIALIZING
        self.last_error: Optional[str] = None
        self.config = config
        self.initialization_time: Optional[datetime] = None
        self.last_health_check: Optional[datetime] = None
        self.health_check_failures: int = 0
        self.processing_count: int = 0
        self._logger = logging.getLogger(f"{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the model and return success status.
        Implements Requirement 1.2 - model initialization.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input and return results.
        Implements Requirement 1.2 - model processing interface.
        
        Args:
            input_data: Input data for model processing
            
        Returns:
            Dict[str, Any]: Processing results
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if model is healthy and ready for processing.
        Implements Requirement 6.4 - health monitoring.
        
        Returns:
            bool: True if model is healthy, False otherwise
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Clean up model resources.
        Implements Requirement 6.3 - resource cleanup.
        """
        pass
    
    async def _initialize_with_timeout(self, timeout_seconds: int = 300) -> bool:
        """
        Initialize model with timeout handling.
        Implements Requirement 6.3 - initialization procedures.
        
        Args:
            timeout_seconds: Maximum time to wait for initialization
            
        Returns:
            bool: True if initialization successful within timeout
        """
        try:
            self.status = ModelStatus.INITIALIZING
            self._logger.info(f"Starting initialization with {timeout_seconds}s timeout")
            
            # Use asyncio.wait_for to enforce timeout
            result = await asyncio.wait_for(
                self.initialize(), 
                timeout=timeout_seconds
            )
            
            if result:
                self.status = ModelStatus.READY
                self.initialization_time = datetime.utcnow()
                self.health_check_failures = 0
                self._logger.info("Model initialization completed successfully")
                return True
            else:
                self.status = ModelStatus.FAILED
                self.last_error = "Model initialization returned False"
                self._logger.error("Model initialization failed")
                return False
                
        except asyncio.TimeoutError:
            self.status = ModelStatus.FAILED
            self.last_error = f"Initialization timeout after {timeout_seconds} seconds"
            self._logger.error(f"Model initialization timed out after {timeout_seconds} seconds")
            return False
        except Exception as e:
            self.status = ModelStatus.FAILED
            self.last_error = str(e)
            self._logger.error(f"Model initialization failed with exception: {e}", exc_info=True)
            return False
    
    async def _process_with_monitoring(self, input_data: Any) -> Dict[str, Any]:
        """
        Process input with monitoring and error handling.
        Implements Requirement 6.3 - processing monitoring.
        
        Args:
            input_data: Input data for processing
            
        Returns:
            Dict[str, Any]: Processing results
        """
        if self.status != ModelStatus.READY:
            raise RuntimeError(f"Model not ready for processing. Status: {self.status}")
        
        start_time = time.time()
        try:
            self.processing_count += 1
            result = await self.process(input_data)
            processing_time = (time.time() - start_time) * 1000
            
            self._logger.debug(f"Processing completed in {processing_time:.2f}ms")
            return result
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.last_error = str(e)
            self._logger.error(f"Processing failed after {processing_time:.2f}ms: {e}", exc_info=True)
            raise
    
    async def _health_check_with_monitoring(self) -> bool:
        """
        Perform health check with failure tracking.
        Implements Requirement 6.4 - health monitoring.
        
        Returns:
            bool: True if model is healthy
        """
        try:
            is_healthy = await self.health_check()
            self.last_health_check = datetime.utcnow()
            
            if is_healthy:
                self.health_check_failures = 0
                self._logger.debug("Health check passed")
            else:
                self.health_check_failures += 1
                self._logger.warning(f"Health check failed (failure count: {self.health_check_failures})")
                
                # Mark as failed if too many consecutive failures
                if self.health_check_failures >= 3:
                    self.status = ModelStatus.FAILED
                    self.last_error = f"Health check failed {self.health_check_failures} consecutive times"
                    self._logger.error("Model marked as failed due to consecutive health check failures")
            
            return is_healthy
            
        except Exception as e:
            self.health_check_failures += 1
            self.last_error = str(e)
            self._logger.error(f"Health check exception (failure count: {self.health_check_failures}): {e}", exc_info=True)
            
            # Mark as failed if too many consecutive failures
            if self.health_check_failures >= 3:
                self.status = ModelStatus.FAILED
                
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        Implements Requirement 6.4 - status monitoring.
        
        Returns:
            Dict[str, Any]: Model information and metrics
        """
        return {
            "status": self.status.value,
            "last_error": self.last_error,
            "initialization_time": self.initialization_time.isoformat() if self.initialization_time else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "health_check_failures": self.health_check_failures,
            "processing_count": self.processing_count,
            "config": {
                "model_type": self.config.model_type.value if self.config else None,
                "config_path": self.config.config_path if self.config else None,
                "max_failures": self.config.max_failures if self.config else None
            } if self.config else None
        }


class ModelManager:
    """
    Central registry and lifecycle management for AI models.
    Supports hot-swapping capabilities for zero-downtime updates.
    Implements Requirements 1.2, 1.3, 1.4 for model management and 6.3, 6.4 for health monitoring.
    """
    
    def __init__(self):
        self._models: Dict[ModelType, BaseModel] = {}
        self._model_configs: Dict[ModelType, ModelConfig] = {}
        self._model_locks: Dict[ModelType, asyncio.Lock] = {
            model_type: asyncio.Lock() for model_type in ModelType
        }
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._health_monitor_running = False
        self._logger = logging.getLogger(__name__)
        self._startup_time = datetime.utcnow()
    
    async def start_health_monitoring(self, interval_seconds: int = 60) -> None:
        """
        Start periodic health monitoring for all models.
        Implements Requirement 6.4 - periodic status updates.
        
        Args:
            interval_seconds: Interval between health checks
        """
        if self._health_monitor_running:
            self._logger.warning("Health monitoring already running")
            return
        
        self._health_monitor_running = True
        self._health_monitor_task = asyncio.create_task(
            self._health_monitor_loop(interval_seconds)
        )
        self._logger.info(f"Started health monitoring with {interval_seconds}s interval")
    
    async def stop_health_monitoring(self) -> None:
        """
        Stop periodic health monitoring.
        Implements Requirement 6.3 - cleanup procedures.
        """
        self._health_monitor_running = False
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Stopped health monitoring")
    
    async def _health_monitor_loop(self, interval_seconds: int) -> None:
        """
        Main health monitoring loop.
        Implements Requirement 6.4 - periodic status updates.
        
        Args:
            interval_seconds: Interval between health checks
        """
        while self._health_monitor_running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Health monitoring loop error: {e}", exc_info=True)
                await asyncio.sleep(interval_seconds)
    
    async def _perform_health_checks(self) -> None:
        """
        Perform health checks on all registered models.
        Implements Requirement 6.4 - health monitoring.
        """
        health_check_tasks = []
        
        for model_type, model in self._models.items():
            if model.status in [ModelStatus.READY, ModelStatus.FAILED]:
                task = asyncio.create_task(
                    self._check_model_health(model_type, model)
                )
                health_check_tasks.append(task)
        
        if health_check_tasks:
            await asyncio.gather(*health_check_tasks, return_exceptions=True)
    
    async def _check_model_health(self, model_type: ModelType, model: BaseModel) -> None:
        """
        Check health of a specific model.
        Implements Requirement 6.4 - health monitoring.
        
        Args:
            model_type: Type of model to check
            model: Model instance to check
        """
        try:
            is_healthy = await model._health_check_with_monitoring()
            
            if not is_healthy and model.status == ModelStatus.READY:
                self._logger.warning(f"Model {model_type.value} health check failed")
                
            # Update config if available
            config = self._model_configs.get(model_type)
            if config:
                config.update_health_check()
                
        except Exception as e:
            await self.handle_model_failure(model_type, e)
    
    async def register_model(self, model_type: ModelType, model: BaseModel, config: Optional[ModelConfig] = None) -> bool:
        """
        Register a model with the manager.
        Implements Requirement 1.2 - model registration and storage.
        
        Args:
            model_type: Type of model being registered
            model: Model instance to register
            config: Optional model configuration
            
        Returns:
            bool: True if registration successful
        """
        async with self._model_locks[model_type]:
            try:
                # Store configuration
                if config:
                    self._model_configs[model_type] = config
                    model.config = config
                
                # Initialize the model with timeout
                timeout = config.initialization_timeout_seconds if config else 300
                if await model._initialize_with_timeout(timeout):
                    model.status = ModelStatus.READY
                    self._models[model_type] = model
                    self._logger.info(f"Successfully registered {model_type.value} model")
                    return True
                else:
                    model.status = ModelStatus.FAILED
                    self._logger.error(f"Failed to initialize {model_type.value} model")
                    return False
            except Exception as e:
                await self.handle_model_failure(model_type, e)
                return False
    
    async def get_model(self, model_type: ModelType) -> Optional[BaseModel]:
        """
        Retrieve a registered model.
        Implements Requirement 1.2 - model retrieval.
        
        Args:
            model_type: Type of model to retrieve
            
        Returns:
            Optional[BaseModel]: Model instance if available and ready
        """
        model = self._models.get(model_type)
        if model and model.status == ModelStatus.READY:
            return model
        return None
    
    async def hot_swap_model(self, model_type: ModelType, new_model: BaseModel, config: Optional[ModelConfig] = None) -> bool:
        """
        Replace an existing model without downtime.
        Implements Requirement 1.3 - hot-swapping capabilities.
        
        Args:
            model_type: Type of model to replace
            new_model: New model instance
            config: Optional new model configuration
            
        Returns:
            bool: True if swap successful
        """
        async with self._model_locks[model_type]:
            try:
                # Mark as swapping
                old_model = self._models.get(model_type)
                if old_model:
                    old_model.status = ModelStatus.SWAPPING
                
                # Store new configuration
                if config:
                    self._model_configs[model_type] = config
                    new_model.config = config
                
                # Initialize new model with timeout
                timeout = config.initialization_timeout_seconds if config else 300
                if await new_model._initialize_with_timeout(timeout):
                    new_model.status = ModelStatus.READY
                    
                    # Replace the model
                    self._models[model_type] = new_model
                    
                    # Clean up old model
                    if old_model:
                        await old_model.cleanup()
                    
                    self._logger.info(f"Successfully hot-swapped {model_type.value} model")
                    return True
                else:
                    # Restore old model status if new model fails
                    if old_model:
                        old_model.status = ModelStatus.READY
                    self._logger.error(f"Failed to hot-swap {model_type.value} model")
                    return False
                    
            except Exception as e:
                # Restore old model status on exception
                if old_model:
                    old_model.status = ModelStatus.READY
                await self.handle_model_failure(model_type, e)
                return False
    
    async def get_model_status(self, model_type: ModelType) -> ModelStatus:
        """
        Get current status of a model.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            ModelStatus: Current model status
        """
        model = self._models.get(model_type)
        return model.status if model else ModelStatus.FAILED
    
    async def handle_model_failure(self, model_type: ModelType, error: Exception) -> None:
        """
        Handle model failures with proper logging.
        Implements Requirement 1.4 - model failure handling.
        
        Args:
            model_type: Type of model that failed
            error: Exception that occurred
        """
        error_msg = f"Model {model_type.value} failed: {str(error)}"
        self._logger.error(error_msg, exc_info=True)
        
        # Update model status
        model = self._models.get(model_type)
        if model:
            model.status = ModelStatus.FAILED
            model.last_error = str(error)
        
        # Update config failure count
        config = self._model_configs.get(model_type)
        if config:
            config.increment_failure_count()
            if config.is_failure_threshold_exceeded():
                self._logger.critical(f"Model {model_type.value} exceeded failure threshold ({config.max_failures})")
    
    async def get_all_model_statuses(self) -> Dict[str, bool]:
        """
        Get status of all registered models.
        
        Returns:
            Dict[str, bool]: Model type to ready status mapping
        """
        statuses = {}
        for model_type in ModelType:
            model = self._models.get(model_type)
            statuses[model_type.value] = (
                model is not None and model.status == ModelStatus.READY
            )
        return statuses
    
    async def get_detailed_model_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed information about all models.
        Implements Requirement 6.4 - comprehensive status monitoring.
        
        Returns:
            Dict[str, Dict[str, Any]]: Detailed model information
        """
        model_info = {}
        
        for model_type in ModelType:
            model = self._models.get(model_type)
            config = self._model_configs.get(model_type)
            
            if model:
                info = model.get_model_info()
                if config:
                    info["config"].update({
                        "failure_count": config.failure_count,
                        "max_failures": config.max_failures,
                        "last_health_check": config.last_health_check.isoformat() if config.last_health_check else None,
                        "health_check_interval": config.health_check_interval_seconds
                    })
                model_info[model_type.value] = info
            else:
                model_info[model_type.value] = {
                    "status": "not_registered",
                    "last_error": None,
                    "config": config.__dict__ if config else None
                }
        
        return model_info
    
    async def cleanup_all_models(self) -> None:
        """
        Clean up all registered models and stop monitoring.
        Implements Requirement 6.3 - cleanup procedures.
        """
        self._logger.info("Starting cleanup of all models")
        
        # Stop health monitoring
        await self.stop_health_monitoring()
        
        # Clean up all models
        cleanup_tasks = []
        for model_type, model in self._models.items():
            if model:
                task = asyncio.create_task(self._cleanup_model(model_type, model))
                cleanup_tasks.append(task)
        
        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)
        
        # Clear all references
        self._models.clear()
        self._model_configs.clear()
        
        self._logger.info("Completed cleanup of all models")
    
    async def _cleanup_model(self, model_type: ModelType, model: BaseModel) -> None:
        """
        Clean up a specific model.
        Implements Requirement 6.3 - cleanup procedures.
        
        Args:
            model_type: Type of model to clean up
            model: Model instance to clean up
        """
        try:
            await model.cleanup()
            self._logger.info(f"Successfully cleaned up {model_type.value} model")
        except Exception as e:
            self._logger.error(f"Error cleaning up {model_type.value} model: {e}", exc_info=True)
    
    def get_uptime_seconds(self) -> int:
        """
        Get manager uptime in seconds.
        
        Returns:
            int: Uptime in seconds
        """
        return int((datetime.utcnow() - self._startup_time).total_seconds())
    
    def is_healthy(self) -> bool:
        """
        Check if the model manager is in a healthy state.
        Implements Requirement 6.4 - health monitoring.
        
        Returns:
            bool: True if at least one model is ready
        """
        return any(
            model.status == ModelStatus.READY 
            for model in self._models.values()
        )