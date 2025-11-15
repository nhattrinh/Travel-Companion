"""
Food Image Service for retrieving and caching food images.

This service handles searching for food images based on menu item names,
caching results for improved performance, and providing fallback mechanisms
when images are not available.

Requirements addressed: 5.1, 5.2, 5.3, 5.4
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any
import hashlib
import json
from datetime import datetime, timedelta

from app.models.internal_models import FoodImage, ErrorCode
from app.core.cache_client import CacheClient
from app.config.settings import settings


class FoodImageServiceError(Exception):
    """Base exception for food image service errors"""
    pass


class ImageSearchError(FoodImageServiceError):
    """Raised when image search fails"""
    pass


class CacheError(FoodImageServiceError):
    """Raised when cache operations fail"""
    pass


class FoodImageService:
    """
    Service for retrieving food images with caching capabilities.
    
    Implements Requirements:
    - 5.1: Search for food images by name
    - 5.2: Return most relevant images
    - 5.3: Handle unavailable images with placeholders
    - 5.4: Graceful handling of service failures
    """
    
    DEFAULT_PLACEHOLDER_URL = "/static/images/food-placeholder.jpg"
    DEFAULT_PLACEHOLDER_DESCRIPTION = "Food item placeholder image"
    MAX_IMAGES_PER_SEARCH = 10
    DEFAULT_CACHE_TTL_HOURS = 24
    
    def __init__(self, cache_client: Optional[CacheClient] = None, external_api_client=None):
        """
        Initialize the food image service.
        
        Args:
            cache_client: Redis cache client (optional)
            external_api_client: External API client for image search (optional)
        """
        self.cache_client = cache_client
        self.external_api_client = external_api_client
        self.logger = logging.getLogger(__name__)
        
        # Cache TTL from settings
        self.cache_ttl_seconds = settings.cache_ttl_seconds
        
        # In-memory fallback cache when Redis is unavailable
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
    
    async def search_food_images(
        self, 
        food_name: str, 
        limit: int = 5
    ) -> List[FoodImage]:
        """
        Search for food images by name (Requirement 5.1).
        
        Args:
            food_name: Name of the food item to search for
            limit: Maximum number of images to return
            
        Returns:
            List of FoodImage objects with search results
            
        Raises:
            ImageSearchError: When search operation fails
        """
        if not food_name or not food_name.strip():
            self.logger.warning("Empty food name provided for image search")
            return await self.handle_no_images_found(food_name)
        
        # Normalize food name for consistent searching
        normalized_name = self._normalize_food_name(food_name)
        
        try:
            # First check cache
            cached_images = await self.get_cached_images(normalized_name)
            if cached_images:
                self.logger.info(f"Retrieved {len(cached_images)} cached images for '{food_name}'")
                return cached_images[:limit]
            
            # Search using external API if available
            if self.external_api_client:
                images = await self._search_external_api(normalized_name, limit)
            else:
                # Use mock data for development/testing
                images = await self._get_mock_images(normalized_name, limit)
            
            # Cache the results
            if images:
                await self.cache_images(normalized_name, images)
                self.logger.info(f"Found and cached {len(images)} images for '{food_name}'")
            else:
                self.logger.info(f"No images found for '{food_name}', using placeholder")
                images = await self.handle_no_images_found(food_name)
            
            return images
            
        except Exception as e:
            self.logger.error(f"Error searching for images for '{food_name}': {str(e)}")
            return await self.handle_image_service_failure(food_name)
    
    async def get_most_relevant_images(
        self, 
        food_name: str, 
        limit: int = 3
    ) -> List[FoodImage]:
        """
        Get most relevant images for food item (Requirement 5.2).
        
        Args:
            food_name: Name of the food item
            limit: Maximum number of images to return
            
        Returns:
            List of most relevant FoodImage objects sorted by relevance score
        """
        try:
            # Get all available images
            all_images = await self.search_food_images(food_name, self.MAX_IMAGES_PER_SEARCH)
            
            if not all_images:
                return await self.handle_no_images_found(food_name)
            
            # Sort by relevance score (highest first) and return top results
            sorted_images = sorted(all_images, key=lambda img: img.relevance_score, reverse=True)
            most_relevant = sorted_images[:limit]
            
            self.logger.info(
                f"Selected {len(most_relevant)} most relevant images for '{food_name}' "
                f"from {len(all_images)} total images"
            )
            
            return most_relevant
            
        except Exception as e:
            self.logger.error(f"Error getting most relevant images for '{food_name}': {str(e)}")
            return await self.handle_image_service_failure(food_name)
    
    async def get_cached_images(self, food_name: str) -> Optional[List[FoodImage]]:
        """
        Retrieve cached food images.
        
        Args:
            food_name: Name of the food item
            
        Returns:
            List of cached FoodImage objects or None if not cached
        """
        cache_key = self._generate_cache_key(food_name)
        
        try:
            # Try Redis cache first
            if self.cache_client:
                cached_data = await self._get_from_redis_cache(cache_key)
                if cached_data:
                    return self._deserialize_images(cached_data)
            
            # Fallback to memory cache
            return await self._get_from_memory_cache(cache_key)
            
        except Exception as e:
            self.logger.warning(f"Error retrieving cached images for '{food_name}': {str(e)}")
            return None
    
    async def cache_images(self, food_name: str, images: List[FoodImage]) -> bool:
        """
        Cache food images for future use.
        
        Args:
            food_name: Name of the food item
            images: List of FoodImage objects to cache
            
        Returns:
            True if caching was successful, False otherwise
        """
        if not images:
            return False
        
        cache_key = self._generate_cache_key(food_name)
        
        try:
            # Try Redis cache first
            if self.cache_client:
                success = await self._cache_to_redis(cache_key, images)
                if success:
                    return True
            
            # Fallback to memory cache
            return await self._cache_to_memory(cache_key, images)
            
        except Exception as e:
            self.logger.warning(f"Error caching images for '{food_name}': {str(e)}")
            return False
    
    async def handle_no_images_found(self, food_name: str) -> List[FoodImage]:
        """
        Return placeholder when no images found (Requirement 5.3).
        
        Args:
            food_name: Name of the food item
            
        Returns:
            List containing a placeholder FoodImage
        """
        placeholder = FoodImage(
            url=self.DEFAULT_PLACEHOLDER_URL,
            description=f"{self.DEFAULT_PLACEHOLDER_DESCRIPTION} for {food_name}",
            relevance_score=0.0,
            is_placeholder=True,
            cache_key=None
        )
        
        self.logger.info(f"No images found for '{food_name}', returning placeholder")
        return [placeholder]
    
    async def handle_image_service_failure(self, food_name: str) -> List[FoodImage]:
        """
        Handle image service failures gracefully (Requirement 5.4).
        
        Args:
            food_name: Name of the food item
            
        Returns:
            List containing a placeholder FoodImage with error indication
        """
        error_placeholder = FoodImage(
            url=self.DEFAULT_PLACEHOLDER_URL,
            description=f"Image service temporarily unavailable for {food_name}",
            relevance_score=0.0,
            is_placeholder=True,
            cache_key=None
        )
        
        self.logger.warning(f"Image service failure for '{food_name}', returning error placeholder")
        return [error_placeholder]
    
    def _normalize_food_name(self, food_name: str) -> str:
        """
        Normalize food name for consistent searching and caching.
        
        Args:
            food_name: Original food name
            
        Returns:
            Normalized food name
        """
        # Convert to lowercase, strip whitespace, and remove special characters
        normalized = food_name.lower().strip()
        # Replace multiple spaces with single space
        normalized = ' '.join(normalized.split())
        return normalized
    
    def _generate_cache_key(self, food_name: str) -> str:
        """
        Generate cache key for food name.
        
        Args:
            food_name: Name of the food item
            
        Returns:
            Cache key string
        """
        # Create a hash of the normalized food name for consistent caching
        normalized_name = self._normalize_food_name(food_name)
        hash_object = hashlib.md5(normalized_name.encode())
        return f"food_images:{hash_object.hexdigest()}"
    
    async def _search_external_api(self, food_name: str, limit: int) -> List[FoodImage]:
        """
        Search for images using external API.
        
        Args:
            food_name: Name of the food item
            limit: Maximum number of images to return
            
        Returns:
            List of FoodImage objects from external API
        """
        try:
            # This would integrate with actual image search APIs like Unsplash, Pexels, etc.
            # For now, we'll simulate the API call
            await asyncio.sleep(0.1)  # Simulate API latency
            
            # Mock implementation - replace with actual API integration
            return await self._get_mock_images(food_name, limit)
            
        except Exception as e:
            self.logger.error(f"External API search failed for '{food_name}': {str(e)}")
            raise ImageSearchError(f"External image search failed: {str(e)}")
    
    async def _get_mock_images(self, food_name: str, limit: int) -> List[FoodImage]:
        """
        Generate mock images for development and testing.
        
        Args:
            food_name: Name of the food item
            limit: Maximum number of images to return
            
        Returns:
            List of mock FoodImage objects
        """
        # Generate mock images with varying relevance scores
        mock_images = []
        
        for i in range(min(limit, 5)):  # Generate up to 5 mock images
            relevance_score = max(0.5, 1.0 - (i * 0.15))  # Decreasing relevance
            
            mock_image = FoodImage(
                url=f"https://example.com/food-images/{food_name.replace(' ', '-')}-{i+1}.jpg",
                description=f"High quality image of {food_name} - variant {i+1}",
                relevance_score=relevance_score,
                is_placeholder=False,
                cache_key=self._generate_cache_key(food_name)
            )
            mock_images.append(mock_image)
        
        return mock_images
    
    async def _get_from_redis_cache(self, cache_key: str) -> Optional[str]:
        """
        Get data from Redis cache.
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            Cached data as string or None if not found
        """
        try:
            if self.cache_client and self.cache_client.is_connected:
                return await self.cache_client.get(cache_key)
            return None
        except Exception as e:
            self.logger.warning(f"Redis cache get failed for key '{cache_key}': {str(e)}")
            return None
    
    async def _cache_to_redis(self, cache_key: str, images: List[FoodImage]) -> bool:
        """
        Cache images to Redis.
        
        Args:
            cache_key: Cache key to use
            images: List of FoodImage objects to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.cache_client and self.cache_client.is_connected:
                serialized_data = self._serialize_images(images)
                return await self.cache_client.setex(cache_key, self.cache_ttl_seconds, serialized_data)
            return False
        except Exception as e:
            self.logger.warning(f"Redis cache set failed for key '{cache_key}': {str(e)}")
            return False
    
    async def _get_from_memory_cache(self, cache_key: str) -> Optional[List[FoodImage]]:
        """
        Get data from memory cache.
        
        Args:
            cache_key: Cache key to retrieve
            
        Returns:
            List of cached FoodImage objects or None if not found/expired
        """
        try:
            # Check if key exists and is not expired
            if cache_key in self._memory_cache and cache_key in self._cache_timestamps:
                cached_time = self._cache_timestamps[cache_key]
                expiry_time = cached_time + timedelta(hours=self.DEFAULT_CACHE_TTL_HOURS)
                
                if datetime.utcnow() < expiry_time:
                    cached_data = self._memory_cache[cache_key]
                    return self._deserialize_images(cached_data)
                else:
                    # Remove expired entry
                    del self._memory_cache[cache_key]
                    del self._cache_timestamps[cache_key]
            
            return None
        except Exception as e:
            self.logger.warning(f"Memory cache get failed for key '{cache_key}': {str(e)}")
            return None
    
    async def _cache_to_memory(self, cache_key: str, images: List[FoodImage]) -> bool:
        """
        Cache images to memory.
        
        Args:
            cache_key: Cache key to use
            images: List of FoodImage objects to cache
            
        Returns:
            True if successful, False otherwise
        """
        try:
            serialized_data = self._serialize_images(images)
            self._memory_cache[cache_key] = serialized_data
            self._cache_timestamps[cache_key] = datetime.utcnow()
            return True
        except Exception as e:
            self.logger.warning(f"Memory cache set failed for key '{cache_key}': {str(e)}")
            return False
    
    def _serialize_images(self, images: List[FoodImage]) -> str:
        """
        Serialize FoodImage objects to JSON string.
        
        Args:
            images: List of FoodImage objects
            
        Returns:
            JSON string representation
        """
        image_dicts = []
        for img in images:
            image_dict = {
                'url': img.url,
                'description': img.description,
                'relevance_score': img.relevance_score,
                'is_placeholder': img.is_placeholder,
                'cache_key': img.cache_key
            }
            image_dicts.append(image_dict)
        
        return json.dumps(image_dicts)
    
    def _deserialize_images(self, serialized_data: str) -> List[FoodImage]:
        """
        Deserialize JSON string to FoodImage objects.
        
        Args:
            serialized_data: JSON string representation
            
        Returns:
            List of FoodImage objects
        """
        try:
            image_dicts = json.loads(serialized_data)
            images = []
            
            for img_dict in image_dicts:
                food_image = FoodImage(
                    url=img_dict['url'],
                    description=img_dict['description'],
                    relevance_score=img_dict['relevance_score'],
                    is_placeholder=img_dict.get('is_placeholder', False),
                    cache_key=img_dict.get('cache_key')
                )
                images.append(food_image)
            
            return images
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.error(f"Error deserializing cached images: {str(e)}")
            return []
    
    async def clear_cache(self, food_name: Optional[str] = None) -> bool:
        """
        Clear cached images.
        
        Args:
            food_name: Specific food name to clear, or None to clear all
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if food_name:
                # Clear specific cache entry
                cache_key = self._generate_cache_key(food_name)
                
                # Clear from Redis
                if self.cache_client and self.cache_client.is_connected:
                    await self.cache_client.delete(cache_key)
                
                # Clear from memory cache
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                if cache_key in self._cache_timestamps:
                    del self._cache_timestamps[cache_key]
            else:
                # Clear all cache entries
                if self.cache_client and self.cache_client.is_connected:
                    await self.cache_client.flushdb()
                
                self._memory_cache.clear()
                self._cache_timestamps.clear()
            
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'memory_cache_entries': len(self._memory_cache),
            'memory_cache_timestamps': len(self._cache_timestamps),
            'redis_available': self.cache_client is not None and self.cache_client.is_available,
            'redis_connected': self.cache_client is not None and self.cache_client.is_connected,
            'external_api_available': self.external_api_client is not None,
            'cache_ttl_seconds': self.cache_ttl_seconds
        }
        
        return stats