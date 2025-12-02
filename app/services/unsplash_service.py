"""
Unsplash Image Service - Scrapes images from Unsplash for landmarks and locations.
"""

import asyncio
import logging
import httpx
from bs4 import BeautifulSoup
from typing import List, Optional
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

# Simple in-memory cache with TTL
_image_cache: dict = {}
_cache_ttl = 3600  # 1 hour


class UnsplashImageService:
    """Service for fetching images from Unsplash."""
    
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    def __init__(self):
        self.timeout = 10.0
    
    async def search_images(
        self,
        query: str,
        max_images: int = 3
    ) -> List[str]:
        """
        Search Unsplash for images matching the query.
        
        Args:
            query: Search term (e.g., "Golden Gate Bridge", "fried chicken")
            max_images: Maximum number of image URLs to return
            
        Returns:
            List of image URLs
        """
        # Check cache first
        cache_key = f"{query.lower()}:{max_images}"
        if cache_key in _image_cache:
            cached_time, cached_urls = _image_cache[cache_key]
            if time.time() - cached_time < _cache_ttl:
                logger.debug(f"Cache hit for '{query}'")
                return cached_urls
        
        try:
            # Format query for URL
            q = query.replace(' ', '-').lower()
            url = f'https://unsplash.com/s/photos/{q}'
            
            logger.info(f"Fetching Unsplash images for: {query}")
            
            async with httpx.AsyncClient(
                headers=self.HEADERS,
                timeout=self.timeout,
                follow_redirects=True
            ) as client:
                response = await client.get(url)
                
                if response.status_code != 200:
                    logger.warning(
                        f"Unsplash returned {response.status_code} for '{query}'"
                    )
                    return []
                
                soup = BeautifulSoup(response.text, 'lxml')
                
                image_urls = []
                
                # Look for img tags with srcset containing photos
                imgs = soup.select('img[srcset]')
                for img in imgs:
                    srcset = img.get('srcset', '')
                    
                    # Skip if no photo in srcset (profile images, etc.)
                    if '/photo-' not in srcset:
                        continue
                    
                    # Parse srcset to find 400w version
                    # Format: "https://...?w=100 100w, https://...?w=200 200w, ..."
                    parts = srcset.split(',')
                    target_url = None
                    
                    for part in parts:
                        part = part.strip()
                        if '400w' in part:
                            # Extract URL (before the space and width)
                            target_url = part.split()[0] if ' ' in part else part
                            break
                    
                    # Fallback to first URL if no 400w
                    if not target_url and parts:
                        first_part = parts[0].strip()
                        target_url = first_part.split()[0] if ' ' in first_part else first_part
                    
                    if target_url and 'images.unsplash.com/photo-' in target_url:
                        image_urls.append(target_url)
                    
                    if len(image_urls) >= max_images * 2:  # Get extras for dedup
                        break
                
                # Deduplicate by photo ID
                seen = set()
                unique_urls = []
                for img_url in image_urls:
                    # Extract photo ID: photo-XXXXXXXXX
                    import re
                    match = re.search(r'photo-([a-zA-Z0-9_-]+)', img_url)
                    if match:
                        photo_id = match.group(1)
                        if photo_id not in seen:
                            seen.add(photo_id)
                            unique_urls.append(img_url)
                
                image_urls = unique_urls[:max_images]
                
                # Cache the results
                _image_cache[cache_key] = (time.time(), image_urls)
                
                logger.info(
                    f"Found {len(image_urls)} images for '{query}'"
                )
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


# Singleton instance
_service: Optional[UnsplashImageService] = None


def get_unsplash_service() -> UnsplashImageService:
    """Get the Unsplash image service singleton."""
    global _service
    if _service is None:
        _service = UnsplashImageService()
    return _service
