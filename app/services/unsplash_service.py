"""
Unsplash Image Service - Fetches images from Unsplash using the official API.
"""

import asyncio
import logging
import httpx
from typing import List, Optional
import time

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Simple in-memory cache with TTL
_image_cache: dict = {}


class UnsplashImageService:
    """Service for fetching images from Unsplash using the official API."""
    
    def __init__(self):
        self.settings = get_settings().unsplash
        self.api_url = self.settings.api_url
        self.access_key = self.settings.access_key
        self.secret_key = self.settings.secret_key
        self.timeout = self.settings.timeout_seconds
        self.cache_ttl = self.settings.cache_ttl_seconds
        
        if self.access_key:
            logger.info("Unsplash API configured with access key")
        else:
            logger.warning(
                "Unsplash API key not configured. "
                "Set UNSPLASH_ACCESS_KEY in .env file."
            )
    
    def _get_headers(self) -> dict:
        """Get headers for Unsplash API requests."""
        if not self.access_key:
            return {}
        return {
            "Authorization": f"Client-ID {self.access_key}",
            "Accept-Version": "v1",
        }
    
    async def search_images(
        self,
        query: str,
        max_images: int = 3
    ) -> List[str]:
        """
        Search Unsplash for images matching the query using the official API.
        
        Args:
            query: Search term (e.g., "Golden Gate Bridge", "fried chicken")
            max_images: Maximum number of image URLs to return
            
        Returns:
            List of image URLs
        """
        # Check if API key is configured
        if not self.access_key:
            logger.warning(
                "Unsplash API key not configured. "
                "Set UNSPLASH_ACCESS_KEY environment variable."
            )
            return []
        
        # Check cache first
        cache_key = f"{query.lower()}:{max_images}"
        if cache_key in _image_cache:
            cached_time, cached_urls = _image_cache[cache_key]
            if time.time() - cached_time < self.cache_ttl:
                logger.debug(f"Cache hit for '{query}'")
                return cached_urls
        
        try:
            url = f"{self.api_url}/search/photos"
            params = {
                "query": query,
                "per_page": max_images,
                "orientation": "landscape",
            }
            
            logger.info(f"Fetching Unsplash images for: {query}")
            
            async with httpx.AsyncClient(
                headers=self._get_headers(),
                timeout=self.timeout,
            ) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 401:
                    logger.error("Unsplash API authentication failed. Check API key.")
                    return []
                
                if response.status_code == 403:
                    logger.error("Unsplash API rate limit exceeded.")
                    return []
                
                if response.status_code != 200:
                    logger.warning(
                        f"Unsplash API returned {response.status_code} for '{query}'"
                    )
                    return []
                
                data = response.json()
                results = data.get("results", [])
                
                # Extract regular-sized image URLs
                image_urls = []
                for photo in results:
                    urls = photo.get("urls", {})
                    # Prefer 'regular' size (1080px width), fallback to 'small'
                    img_url = urls.get("regular") or urls.get("small")
                    if img_url:
                        image_urls.append(img_url)
                
                # Cache the results
                _image_cache[cache_key] = (time.time(), image_urls)
                
                logger.info(f"Found {len(image_urls)} images for '{query}'")
                return image_urls
                
        except httpx.TimeoutException:
            logger.warning(f"Timeout fetching images for '{query}'")
            return []
        except Exception as e:
            logger.error(f"Error fetching Unsplash images: {e}")
            return []
    
    async def get_location_images(
        self,
        location_name: str,
        max_images: int = 3
    ) -> List[str]:
        """
        Get images for a location/landmark.
        Tries the full name first, then falls back to key terms.
        
        Args:
            location_name: Name of location (e.g., "Golden Gate Bridge")
            max_images: Maximum number of images
            
        Returns:
            List of image URLs
        """
        # Try full location name first
        images = await self.search_images(location_name, max_images)
        
        if not images:
            # Try with just key landmark terms
            # Remove common prefixes/suffixes
            simplified = location_name.replace("The ", "").replace(" Area", "")
            if simplified != location_name:
                images = await self.search_images(simplified, max_images)
        
        return images
    
    async def get_random_photo(
        self,
        query: Optional[str] = None,
        orientation: str = "landscape"
    ) -> Optional[str]:
        """
        Get a random photo from Unsplash.
        
        Args:
            query: Optional search query to filter random photo
            orientation: Photo orientation (landscape, portrait, squarish)
            
        Returns:
            Image URL or None
        """
        if not self.access_key:
            logger.warning("Unsplash API key not configured.")
            return None
        
        try:
            url = f"{self.api_url}/photos/random"
            params = {"orientation": orientation}
            if query:
                params["query"] = query
            
            async with httpx.AsyncClient(
                headers=self._get_headers(),
                timeout=self.timeout,
            ) as client:
                response = await client.get(url, params=params)
                
                if response.status_code != 200:
                    logger.warning(
                        f"Unsplash random photo returned {response.status_code}"
                    )
                    return None
                
                data = response.json()
                urls = data.get("urls", {})
                return urls.get("regular") or urls.get("small")
                
        except Exception as e:
            logger.error(f"Error fetching random Unsplash photo: {e}")
            return None


# Singleton instance
_service: Optional[UnsplashImageService] = None


def get_unsplash_service() -> UnsplashImageService:
    """Get the Unsplash image service singleton."""
    global _service
    if _service is None:
        _service = UnsplashImageService()
    return _service
