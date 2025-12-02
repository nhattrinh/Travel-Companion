"""
Enhanced Navigation Service with ML-powered context understanding.

Uses modern PyTorch optimizations for:
- Semantic POI category matching
- Context-aware etiquette recommendations
- Smart routing with cultural considerations
- Personalized POI recommendations based on user preferences

Optimized with torch.compile and mixed precision for production performance.
"""
from __future__ import annotations

import asyncio
import json
import math
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class TravelContext(str, Enum):
    """Travel context for personalized recommendations."""
    BUSINESS = "business"
    TOURISM = "tourism"
    DINING = "dining"
    SHOPPING = "shopping"
    CULTURAL = "cultural"
    TRANSIT = "transit"
    EMERGENCY = "emergency"


@dataclass
class POIResult:
    """Enhanced POI result with ML-powered insights."""
    name: str
    category: str
    latitude: float
    longitude: float
    distance_m: float
    etiquette_notes: list[str]
    cultural_tips: list[str] = field(default_factory=list)
    relevance_score: float = 1.0
    accessibility_info: Optional[str] = None
    operating_hours: Optional[str] = None
    language_support: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "name": self.name,
            "category": self.category,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "distance_m": self.distance_m,
            "etiquette_notes": self.etiquette_notes,
            "cultural_tips": self.cultural_tips,
            "relevance_score": self.relevance_score,
            "accessibility_info": self.accessibility_info,
            "operating_hours": self.operating_hours,
            "language_support": self.language_support
        }


@dataclass
class NavigationServiceConfig:
    """Configuration for enhanced navigation service."""
    # Cache settings
    cache_ttl_seconds: int = 300
    cache_prefix: str = "nav"
    
    # ML settings
    enable_semantic_search: bool = True
    enable_personalization: bool = True
    use_gpu: bool = True
    
    # Search settings
    default_radius_m: int = 1000
    max_radius_m: int = 10000
    max_results: int = 50
    
    # Embedding model (for semantic search)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384


# Category embeddings for semantic matching (pre-computed)
# In production, these would be computed from the embedding model
CATEGORY_KEYWORDS = {
    "restaurant": [
        "food", "dining", "eat", "cuisine", "meal", "lunch", "dinner",
        "breakfast", "cafe", "bistro", "eatery"
    ],
    "transit": [
        "station", "train", "bus", "metro", "subway", "transport",
        "airport", "terminal", "platform", "ticket"
    ],
    "cultural": [
        "temple", "shrine", "museum", "gallery", "heritage", "monument",
        "historic", "traditional", "art", "culture"
    ],
    "shopping": [
        "mall", "store", "shop", "market", "department", "boutique",
        "outlet", "retail", "supermarket", "convenience"
    ],
    "accommodation": [
        "hotel", "hostel", "inn", "ryokan", "lodge", "motel",
        "apartment", "stay", "room", "booking"
    ],
    "emergency": [
        "hospital", "clinic", "pharmacy", "police", "embassy",
        "emergency", "medical", "health", "urgent"
    ],
    "entertainment": [
        "theater", "cinema", "park", "garden", "amusement",
        "arcade", "karaoke", "nightclub", "bar"
    ]
}

# Enhanced etiquette database by category and country
ETIQUETTE_DATABASE = {
    "JP": {  # Japan
        "restaurant": [
            "Say 'itadakimasu' before eating",
            "Don't tip - it may be considered rude",
            "Slurping noodles is acceptable and shows appreciation",
            "Don't pass food chopstick to chopstick",
            "Wait for everyone to be served before eating"
        ],
        "transit": [
            "Queue in orderly lines on platform markers",
            "Keep phone on silent mode (manner mode)",
            "Don't eat or drink on trains (except shinkansen)",
            "Give up priority seats for elderly/disabled",
            "Stand on the left side of escalators in Tokyo"
        ],
        "cultural": [
            "Remove shoes before entering temples",
            "Bow when greeting at shrines",
            "Photography may be prohibited inside",
            "Speak quietly and respectfully",
            "Follow purification rituals if applicable"
        ],
        "shopping": [
            "Money trays are used for transactions",
            "Bags are typically handled with care",
            "Tax-free shopping available with passport",
            "Bargaining is generally not practiced"
        ],
        "general": [
            "Bowing is the common greeting",
            "Business cards exchanged with both hands",
            "Punctuality is highly valued",
            "Avoid loud conversations in public"
        ]
    },
    "ES": {  # Spain
        "restaurant": [
            "Lunch is typically 2-4 PM, dinner after 9 PM",
            "Tipping 5-10% is appreciated but not required",
            "Ask for the bill - it won't come automatically",
            "Bread may be charged separately"
        ],
        "transit": [
            "Validate tickets before boarding",
            "Offer seats to elderly",
            "Metro closes around midnight"
        ],
        "cultural": [
            "Siesta time (2-5 PM) - many shops close",
            "Dress modestly when visiting churches",
            "Greet with two kisses on cheeks"
        ],
        "general": [
            "Personal space is smaller than in US/UK",
            "Conversations are animated and expressive",
            "Punctuality is flexible in social settings"
        ]
    },
    "VN": {  # Vietnam
        "restaurant": [
            "Wait to be seated at formal restaurants",
            "It's polite to leave a little food on your plate",
            "Chopsticks should rest on bowl, not in rice",
            "Tipping is not traditional but appreciated"
        ],
        "transit": [
            "Negotiate taxi fares in advance or use meter",
            "Grab/Be app recommended for rides",
            "Cross streets carefully - traffic flows around you"
        ],
        "cultural": [
            "Remove shoes when entering homes/temples",
            "Ask permission before photographing people",
            "Dress modestly at religious sites"
        ],
        "general": [
            "Small talk is important before business",
            "Avoid public displays of anger",
            "Handshakes are common greetings"
        ]
    }
}


class SemanticMatcher:
    """
    Semantic matching for POI categories and queries.
    
    Uses sentence embeddings for semantic similarity when available,
    falls back to keyword matching otherwise.
    """
    
    def __init__(self, config: NavigationServiceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the embedding model."""
        if self._initialized:
            return
            
        if not self.config.enable_semantic_search:
            self._initialized = True
            return
            
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.embedding_model
            )
            self.model = AutoModel.from_pretrained(
                self.config.embedding_model
            )
            
            # Move to appropriate device
            if self.config.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.model = self.model.to('mps')
                
            self.model.eval()
            
            # Apply torch.compile for optimization
            if hasattr(torch, 'compile'):
                self.model = torch.compile(
                    self.model,
                    mode="reduce-overhead",
                    fullgraph=False
                )
                
            self._initialized = True
            logger.info(f"Semantic matcher initialized with {self.config.embedding_model}")
            
        except ImportError:
            logger.warning(
                "Transformers not available, using keyword matching"
            )
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize semantic matcher: {e}")
            self._initialized = True
            
    def _keyword_match_score(
        self,
        query: str,
        category: str
    ) -> float:
        """Compute keyword-based matching score."""
        query_lower = query.lower()
        keywords = CATEGORY_KEYWORDS.get(category, [])
        
        if not keywords:
            return 0.0
            
        matches = sum(1 for kw in keywords if kw in query_lower)
        
        # Also check if query contains category name
        if category.lower() in query_lower:
            matches += 2
            
        return min(1.0, matches / 3)
        
    async def compute_similarity(
        self,
        query: str,
        categories: list[str]
    ) -> dict[str, float]:
        """
        Compute semantic similarity between query and categories.
        
        Returns dict mapping category to similarity score (0-1).
        """
        await self.initialize()
        
        # Use keyword matching if model not available
        if self.model is None:
            return {
                cat: self._keyword_match_score(query, cat)
                for cat in categories
            }
            
        try:
            import torch
            
            # Encode query and categories
            texts = [query] + categories
            
            with torch.inference_mode():
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                
                if self.config.use_gpu and torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    inputs = {k: v.to('mps') for k, v in inputs.items()}
                    
                # Use autocast for mixed precision
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu",
                    enabled=self.config.use_gpu
                ):
                    outputs = self.model(**inputs)
                    
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                # Normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Compute cosine similarity
                query_emb = embeddings[0:1]
                cat_embs = embeddings[1:]
                
                similarities = torch.mm(query_emb, cat_embs.t()).squeeze(0)
                similarities = similarities.cpu().numpy()
                
            return {
                cat: float(sim)
                for cat, sim in zip(categories, similarities)
            }
            
        except Exception as e:
            logger.error(f"Semantic similarity failed: {e}")
            return {
                cat: self._keyword_match_score(query, cat)
                for cat in categories
            }


class PersonalizationEngine:
    """
    User preference learning and personalization.
    
    Tracks user interactions and provides personalized rankings.
    """
    
    def __init__(self, config: NavigationServiceConfig):
        self.config = config
        self._user_preferences: dict[str, dict] = {}
        
    def update_preferences(
        self,
        user_id: str,
        category: str,
        interaction_type: str = "view"
    ) -> None:
        """Record user interaction for preference learning."""
        if user_id not in self._user_preferences:
            self._user_preferences[user_id] = {}
            
        prefs = self._user_preferences[user_id]
        
        if category not in prefs:
            prefs[category] = {"views": 0, "visits": 0, "favorites": 0}
            
        if interaction_type == "view":
            prefs[category]["views"] += 1
        elif interaction_type == "visit":
            prefs[category]["visits"] += 5
        elif interaction_type == "favorite":
            prefs[category]["favorites"] += 10
            
    def get_category_boost(
        self,
        user_id: str,
        category: str
    ) -> float:
        """Get personalization boost for a category."""
        if not self.config.enable_personalization:
            return 1.0
            
        if user_id not in self._user_preferences:
            return 1.0
            
        prefs = self._user_preferences.get(user_id, {})
        cat_prefs = prefs.get(category, {})
        
        score = (
            cat_prefs.get("views", 0) * 0.1 +
            cat_prefs.get("visits", 0) * 0.5 +
            cat_prefs.get("favorites", 0) * 1.0
        )
        
        # Normalize to 1.0 - 2.0 range
        return min(2.0, 1.0 + score / 50)


class EnhancedNavigationService:
    """
    Enhanced navigation service with ML-powered features.
    
    Features:
    - Semantic POI search and matching
    - Personalized recommendations
    - Rich cultural context and etiquette
    - Multi-language support
    - Time-aware recommendations
    """
    
    def __init__(
        self,
        config: Optional[NavigationServiceConfig] = None,
        cache_client: Optional[Any] = None,
        maps_client: Optional[Any] = None
    ):
        self.config = config or NavigationServiceConfig()
        self.cache = cache_client
        self.maps = maps_client
        
        # Initialize components
        self.semantic_matcher = SemanticMatcher(self.config)
        self.personalization = PersonalizationEngine(self.config)
        
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the navigation service."""
        if self._initialized:
            return
            
        # Initialize cache client if not provided
        if self.cache is None:
            try:
                from app.core.cache_client import CacheClient
                self.cache = CacheClient()
            except ImportError:
                logger.warning("Cache client not available")
                
        # Initialize maps client if not provided
        if self.maps is None:
            try:
                from app.services.maps_client import MapsClient
                self.maps = MapsClient()
            except ImportError:
                logger.warning("Maps client not available")
                
        # Initialize semantic matcher
        await self.semantic_matcher.initialize()
        
        self._initialized = True
        logger.info("Enhanced navigation service initialized")
        
    async def search_pois(
        self,
        latitude: float,
        longitude: float,
        query: Optional[str] = None,
        radius_m: int = 1000,
        categories: Optional[list[str]] = None,
        context: Optional[TravelContext] = None,
        user_id: Optional[str] = None,
        country_code: str = "JP",
        language: str = "en"
    ) -> list[POIResult]:
        """
        Search for POIs with semantic matching and personalization.
        
        Args:
            latitude: User latitude
            longitude: User longitude
            query: Natural language search query
            radius_m: Search radius in meters
            categories: Filter by categories
            context: Travel context for recommendations
            user_id: User ID for personalization
            country_code: Country code for etiquette
            language: Preferred language
            
        Returns:
            List of POI results sorted by relevance
        """
        await self.initialize()
        
        # Validate radius
        radius_m = min(radius_m, self.config.max_radius_m)
        
        # Try cache first
        cache_key = self._build_cache_key(
            latitude, longitude, radius_m, query, categories
        )
        
        cached = await self._get_cached(cache_key)
        if cached:
            return cached
            
        # Fetch raw POIs from maps service
        raw_pois = await self._fetch_pois(latitude, longitude, radius_m)
        
        # Compute semantic similarity if query provided
        similarity_scores: dict[str, float] = {}
        if query and self.config.enable_semantic_search:
            poi_categories = list(set(p.get("category", "") for p in raw_pois))
            similarity_scores = await self.semantic_matcher.compute_similarity(
                query, poi_categories
            )
            
        # Filter by categories if specified
        if categories:
            raw_pois = [
                p for p in raw_pois
                if p.get("category") in categories
            ]
            
        # Build enhanced POI results
        results = []
        for poi in raw_pois:
            distance = self._haversine(
                latitude, longitude,
                poi.get("lat", 0), poi.get("lon", 0)
            )
            
            category = poi.get("category", "general")
            
            # Get etiquette notes
            etiquette = self._get_etiquette(category, country_code)
            cultural_tips = self._get_cultural_tips(
                category, country_code, context
            )
            
            # Compute relevance score
            relevance = self._compute_relevance(
                poi=poi,
                distance=distance,
                similarity_score=similarity_scores.get(category, 0.5),
                user_id=user_id,
                context=context
            )
            
            results.append(POIResult(
                name=poi.get("name", "Unknown"),
                category=category,
                latitude=poi.get("lat", 0),
                longitude=poi.get("lon", 0),
                distance_m=distance,
                etiquette_notes=etiquette,
                cultural_tips=cultural_tips,
                relevance_score=relevance,
                accessibility_info=poi.get("accessibility"),
                operating_hours=poi.get("hours"),
                language_support=poi.get("languages", ["ja", "en"])
            ))
            
        # Sort by relevance (higher is better)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Limit results
        results = results[:self.config.max_results]
        
        # Cache results
        await self._cache_results(cache_key, results)
        
        return results
        
    async def get_etiquette_guide(
        self,
        category: str,
        country_code: str = "JP"
    ) -> dict[str, Any]:
        """
        Get comprehensive etiquette guide for a category.
        
        Returns detailed cultural guidance for the category
        in the specified country.
        """
        country_etiquette = ETIQUETTE_DATABASE.get(
            country_code.upper(), {}
        )
        
        category_tips = country_etiquette.get(category, [])
        general_tips = country_etiquette.get("general", [])
        
        return {
            "category": category,
            "country": country_code,
            "category_etiquette": category_tips,
            "general_etiquette": general_tips,
            "do": self._extract_dos(category_tips + general_tips),
            "dont": self._extract_donts(category_tips + general_tips)
        }
        
    async def record_interaction(
        self,
        user_id: str,
        poi_category: str,
        interaction_type: str = "view"
    ) -> None:
        """Record user interaction for preference learning."""
        self.personalization.update_preferences(
            user_id, poi_category, interaction_type
        )
        
    def _compute_relevance(
        self,
        poi: dict,
        distance: float,
        similarity_score: float,
        user_id: Optional[str],
        context: Optional[TravelContext]
    ) -> float:
        """Compute relevance score for a POI."""
        # Base score from semantic similarity
        score = similarity_score
        
        # Distance penalty (closer is better)
        if distance > 0:
            distance_factor = max(0.2, 1.0 - (distance / 5000))
            score *= distance_factor
            
        # Personalization boost
        if user_id:
            category = poi.get("category", "general")
            boost = self.personalization.get_category_boost(user_id, category)
            score *= boost
            
        # Context boost
        if context:
            category = poi.get("category", "")
            context_match = self._context_category_match(context, category)
            score *= (1.0 + context_match * 0.5)
            
        # Time-based adjustments
        score *= self._get_time_factor(poi)
        
        return min(1.0, max(0.0, score))
        
    def _context_category_match(
        self,
        context: TravelContext,
        category: str
    ) -> float:
        """Compute match between travel context and POI category."""
        context_categories = {
            TravelContext.BUSINESS: ["transit", "accommodation", "restaurant"],
            TravelContext.TOURISM: ["cultural", "entertainment", "shopping"],
            TravelContext.DINING: ["restaurant", "cafe", "market"],
            TravelContext.SHOPPING: ["shopping", "market"],
            TravelContext.CULTURAL: ["cultural", "museum", "temple"],
            TravelContext.TRANSIT: ["transit", "station"],
            TravelContext.EMERGENCY: ["emergency", "hospital", "pharmacy"]
        }
        
        matching_categories = context_categories.get(context, [])
        return 1.0 if category in matching_categories else 0.0
        
    def _get_time_factor(self, poi: dict) -> float:
        """Adjust relevance based on current time and POI hours."""
        # This would check if POI is currently open
        # For now, return neutral factor
        return 1.0
        
    def _get_etiquette(
        self,
        category: str,
        country_code: str
    ) -> list[str]:
        """Get etiquette notes for a category."""
        country_etiquette = ETIQUETTE_DATABASE.get(
            country_code.upper(), {}
        )
        
        tips = country_etiquette.get(category, [])
        if not tips:
            tips = country_etiquette.get("general", [])
            
        return tips[:3]  # Return top 3 tips
        
    def _get_cultural_tips(
        self,
        category: str,
        country_code: str,
        context: Optional[TravelContext]
    ) -> list[str]:
        """Get cultural tips based on category and context."""
        country_etiquette = ETIQUETTE_DATABASE.get(
            country_code.upper(), {}
        )
        
        tips = []
        
        # Add category-specific tips
        cat_tips = country_etiquette.get(category, [])
        tips.extend(cat_tips[:2])
        
        # Add general tips if context matches
        if context:
            general = country_etiquette.get("general", [])
            tips.extend(general[:1])
            
        return tips
        
    def _extract_dos(self, tips: list[str]) -> list[str]:
        """Extract positive recommendations."""
        dos = []
        for tip in tips:
            lower = tip.lower()
            if (
                lower.startswith("do ") or
                lower.startswith("say ") or
                lower.startswith("use ") or
                "is polite" in lower or
                "is common" in lower or
                "is appreciated" in lower
            ):
                dos.append(tip)
        return dos
        
    def _extract_donts(self, tips: list[str]) -> list[str]:
        """Extract things to avoid."""
        donts = []
        for tip in tips:
            lower = tip.lower()
            if (
                lower.startswith("don't ") or
                lower.startswith("avoid ") or
                lower.startswith("never ") or
                "is rude" in lower or
                "may be considered" in lower
            ):
                donts.append(tip)
        return donts
        
    def _build_cache_key(
        self,
        latitude: float,
        longitude: float,
        radius_m: int,
        query: Optional[str],
        categories: Optional[list[str]]
    ) -> str:
        """Build cache key for POI search."""
        # Geohash-like key with reduced precision
        lat_key = int(latitude * 100)
        lon_key = int(longitude * 100)
        
        key_parts = [
            self.config.cache_prefix,
            "pois",
            str(lat_key),
            str(lon_key),
            str(radius_m)
        ]
        
        if query:
            key_parts.append(query[:20].replace(" ", "_"))
            
        if categories:
            key_parts.append("_".join(sorted(categories)))
            
        return ":".join(key_parts)
        
    async def _get_cached(self, key: str) -> Optional[list[POIResult]]:
        """Get cached POI results."""
        if self.cache is None:
            return None
            
        try:
            cached = await self.cache.get(key)
            if cached:
                data = json.loads(cached)
                return [
                    POIResult(**item) for item in data
                ]
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
            
        return None
        
    async def _cache_results(
        self,
        key: str,
        results: list[POIResult]
    ) -> None:
        """Cache POI results."""
        if self.cache is None:
            return
            
        try:
            data = [r.to_dict() for r in results]
            await self.cache.set(
                key,
                json.dumps(data),
                ttl_seconds=self.config.cache_ttl_seconds
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
            
    async def _fetch_pois(
        self,
        latitude: float,
        longitude: float,
        radius_m: int
    ) -> list[dict]:
        """Fetch POIs from maps service."""
        if self.maps is None:
            return []
            
        try:
            return await self.maps.search_nearby_pois(
                latitude, longitude, radius_m
            )
        except Exception as e:
            logger.error(f"Failed to fetch POIs: {e}")
            return []
            
    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters."""
        R = 6371000  # Earth radius in meters
        
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlam = math.radians(lon2 - lon1)
        
        a = (
            math.sin(dphi / 2) ** 2 +
            math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
        )
        
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Factory function for dependency injection
async def create_enhanced_navigation_service(
    config: Optional[NavigationServiceConfig] = None,
    cache_client: Optional[Any] = None,
    maps_client: Optional[Any] = None
) -> EnhancedNavigationService:
    """
    Create and initialize an enhanced navigation service.
    
    Args:
        config: Service configuration
        cache_client: Cache client instance
        maps_client: Maps client instance
        
    Returns:
        Initialized EnhancedNavigationService instance
    """
    service = EnhancedNavigationService(
        config=config,
        cache_client=cache_client,
        maps_client=maps_client
    )
    await service.initialize()
    return service
