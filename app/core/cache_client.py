"""
Redis cache client for the menu translation backend.

This module provides a Redis-based caching client with connection management,
error handling, and fallback mechanisms for improved performance and reliability.

Requirements addressed: 5.1, 5.3, 5.4
"""

import asyncio
import logging
from typing import Optional, Any, Dict
import json
from datetime import timedelta

try:
    # Use redis.asyncio (modern approach, replaces deprecated aioredis)
    from redis import asyncio as aioredis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    try:
        # Fallback to legacy aioredis for older environments
        import aioredis
        from aioredis import Redis
        REDIS_AVAILABLE = True
    except ImportError:
        REDIS_AVAILABLE = False
        Redis = None

from app.config.settings import settings


class CacheClientError(Exception):
    """Base exception for cache client errors"""
    pass


class ConnectionError(CacheClientError):
    """Raised when Redis connection fails"""
    pass


class CacheClient:
    """
    Redis cache client with connection management and error handling.
    
    Implements Requirements:
    - 5.1: Caching for improved performance
    - 5.3: Graceful handling of cache failures
    - 5.4: Fallback mechanisms when cache is unavailable
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize the cache client.
        
        Args:
            redis_url: Redis connection URL (optional, uses settings if not provided)
        """
        self.redis_url = redis_url or settings.redis.url
        self.redis_client: Optional[Redis] = None
        self.logger = logging.getLogger(__name__)
        self._connection_lock = asyncio.Lock()
        self._is_connected = False
        self._connection_retries = 0
        self._max_retries = 3
        self.enabled = REDIS_AVAILABLE  # Track if Redis is available
        
        if not REDIS_AVAILABLE:
            self.logger.warning("Redis dependencies not available, cache will be disabled")
    
    async def connect(self) -> bool:
        """
        Establish connection to Redis server.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not REDIS_AVAILABLE or not self.redis_url:
            self.logger.info("Redis not available or not configured, skipping connection")
            return False
        
        async with self._connection_lock:
            if self._is_connected and self.redis_client:
                return True
            
            try:
                self.logger.info(f"Connecting to Redis at {self.redis_url}")
                self.redis_client = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_timeout=5.0,
                    socket_connect_timeout=5.0,
                    retry_on_timeout=True,
                    health_check_interval=30
                )
                
                # Test the connection
                await self.redis_client.ping()
                self._is_connected = True
                self._connection_retries = 0
                self.logger.info("Successfully connected to Redis")
                return True
                
            except Exception as e:
                self._connection_retries += 1
                self.logger.error(
                    f"Failed to connect to Redis (attempt {self._connection_retries}): {str(e)}"
                )
                
                if self.redis_client:
                    try:
                        await self.redis_client.close()
                    except:
                        pass
                    self.redis_client = None
                
                self._is_connected = False
                return False
    
    async def disconnect(self) -> None:
        """Disconnect from Redis server."""
        async with self._connection_lock:
            if self.redis_client:
                try:
                    await self.redis_client.close()
                    self.logger.info("Disconnected from Redis")
                except Exception as e:
                    self.logger.warning(f"Error during Redis disconnect: {str(e)}")
                finally:
                    self.redis_client = None
                    self._is_connected = False
    
    async def get(self, key: str) -> Optional[str]:
        """
        Get value from cache.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value as string or None if not found/error
        """
        if not await self._ensure_connection():
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                self.logger.debug(f"Cache hit for key: {key}")
            else:
                self.logger.debug(f"Cache miss for key: {key}")
            return value
            
        except Exception as e:
            self.logger.warning(f"Error getting cache key '{key}': {str(e)}")
            await self._handle_connection_error()
            return None
    
    async def set(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key to set
            value: Value to cache
            ttl_seconds: Time to live in seconds (optional)
            
        Returns:
            True if successful, False otherwise
        """
        if not await self._ensure_connection():
            return False
        
        try:
            if ttl_seconds:
                result = await self.redis_client.setex(key, ttl_seconds, value)
            else:
                result = await self.redis_client.set(key, value)
            
            if result:
                self.logger.debug(f"Cache set for key: {key}")
                return True
            else:
                self.logger.warning(f"Failed to set cache key: {key}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Error setting cache key '{key}': {str(e)}")
            await self._handle_connection_error()
            return False
    
    async def setex(self, key: str, ttl_seconds: int, value: str) -> bool:
        """
        Set value in cache with expiration.
        
        Args:
            key: Cache key to set
            ttl_seconds: Time to live in seconds
            value: Value to cache
            
        Returns:
            True if successful, False otherwise
        """
        return await self.set(key, value, ttl_seconds)
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not await self._ensure_connection():
            return False
        
        try:
            result = await self.redis_client.delete(key)
            self.logger.debug(f"Cache delete for key: {key}, result: {result}")
            return result > 0
            
        except Exception as e:
            self.logger.warning(f"Error deleting cache key '{key}': {str(e)}")
            await self._handle_connection_error()
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not await self._ensure_connection():
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
            
        except Exception as e:
            self.logger.warning(f"Error checking cache key existence '{key}': {str(e)}")
            await self._handle_connection_error()
            return False
    
    async def expire(self, key: str, ttl_seconds: int) -> bool:
        """
        Set expiration time for key.
        
        Args:
            key: Cache key to set expiration for
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not await self._ensure_connection():
            return False
        
        try:
            result = await self.redis_client.expire(key, ttl_seconds)
            return result
            
        except Exception as e:
            self.logger.warning(f"Error setting expiration for cache key '{key}': {str(e)}")
            await self._handle_connection_error()
            return False
    
    async def flushdb(self) -> bool:
        """
        Clear all keys from current database.
        
        Returns:
            True if successful, False otherwise
        """
        if not await self._ensure_connection():
            return False
        
        try:
            await self.redis_client.flushdb()
            self.logger.info("Cache database flushed")
            return True
            
        except Exception as e:
            self.logger.warning(f"Error flushing cache database: {str(e)}")
            await self._handle_connection_error()
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Dictionary with server information or empty dict if unavailable
        """
        if not await self._ensure_connection():
            return {}
        
        try:
            info = await self.redis_client.info()
            return info
            
        except Exception as e:
            self.logger.warning(f"Error getting Redis info: {str(e)}")
            return {}
    
    async def ping(self) -> bool:
        """
        Ping Redis server to check connectivity.
        
        Returns:
            True if server responds, False otherwise
        """
        if not await self._ensure_connection():
            return False
        
        try:
            result = await self.redis_client.ping()
            return result is True
            
        except Exception as e:
            self.logger.warning(f"Redis ping failed: {str(e)}")
            await self._handle_connection_error()
            return False
    
    async def _ensure_connection(self) -> bool:
        """
        Ensure Redis connection is established.
        
        Returns:
            True if connected, False otherwise
        """
        if self._is_connected and self.redis_client:
            return True
        
        if self._connection_retries >= self._max_retries:
            self.logger.warning(
                f"Max connection retries ({self._max_retries}) exceeded, "
                "cache operations will be disabled"
            )
            return False
        
        return await self.connect()
    
    async def _handle_connection_error(self) -> None:
        """Handle connection errors by marking connection as failed."""
        self._is_connected = False
        if self.redis_client:
            try:
                await self.redis_client.close()
            except:
                pass
            self.redis_client = None
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to Redis."""
        return self._is_connected and self.redis_client is not None
    
    @property
    def is_available(self) -> bool:
        """Check if Redis is available for use."""
        return REDIS_AVAILABLE and self.redis_url is not None


# Global cache client instance
cache_client: Optional[CacheClient] = None


async def get_cache_client() -> Optional[CacheClient]:
    """
    Get or create the global cache client instance.
    
    Returns:
        CacheClient instance or None if Redis is not available
    """
    global cache_client
    
    if cache_client is None:
        if not REDIS_AVAILABLE or not settings.redis.url:
            return None
        
        cache_client = CacheClient()
        await cache_client.connect()
    
    return cache_client


async def close_cache_client() -> None:
    """Close the global cache client connection."""
    global cache_client
    
    if cache_client:
        await cache_client.disconnect()
        cache_client = None