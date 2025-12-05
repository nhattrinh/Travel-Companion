"""
Navigation Tool Executors - Real API implementations.

Uses:
- OpenStreetMap Overpass API for nearby places/POIs
- OSRM (Open Source Routing Machine) for routing
- Nominatim for geocoding

All APIs are free and don't require API keys.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
OSRM_API_URL = "https://router.project-osrm.org"
NOMINATIM_API_URL = "https://nominatim.openstreetmap.org"

# Rate limiting - be respectful to free APIs
REQUEST_TIMEOUT = 30.0
USER_AGENT = "TravelCompanion/1.0 (navigation assistant)"


# =============================================================================
# Data Models
# =============================================================================

class PlaceCategory(str, Enum):
    """Supported place categories mapped to OSM tags."""
    RESTAURANT = "restaurant"
    CAFE = "cafe"
    BAR = "bar"
    FAST_FOOD = "fast_food"
    HOTEL = "hotel"
    HOSTEL = "hostel"
    MUSEUM = "museum"
    ATTRACTION = "attraction"
    PHARMACY = "pharmacy"
    HOSPITAL = "hospital"
    ATM = "atm"
    BANK = "bank"
    SUPERMARKET = "supermarket"
    CONVENIENCE = "convenience"
    BUS_STATION = "bus_station"
    TRAIN_STATION = "station"
    SUBWAY = "subway_entrance"
    PARKING = "parking"
    TOILET = "toilets"


# Map user-friendly categories to OSM amenity/tourism tags
CATEGORY_TO_OSM = {
    "restaurant": [("amenity", "restaurant")],
    "cafe": [("amenity", "cafe")],
    "bar": [("amenity", "bar"), ("amenity", "pub")],
    "fast_food": [("amenity", "fast_food")],
    "food": [
        ("amenity", "restaurant"),
        ("amenity", "cafe"),
        ("amenity", "fast_food"),
    ],
    "hotel": [("tourism", "hotel")],
    "hostel": [("tourism", "hostel")],
    "accommodation": [
        ("tourism", "hotel"),
        ("tourism", "hostel"),
        ("tourism", "guest_house"),
    ],
    "museum": [("tourism", "museum")],
    "attraction": [("tourism", "attraction"), ("tourism", "viewpoint")],
    "pharmacy": [("amenity", "pharmacy")],
    "hospital": [("amenity", "hospital"), ("amenity", "clinic")],
    "medical": [("amenity", "hospital"), ("amenity", "clinic"), ("amenity", "pharmacy")],
    "atm": [("amenity", "atm")],
    "bank": [("amenity", "bank")],
    "supermarket": [("shop", "supermarket")],
    "convenience": [("shop", "convenience")],
    "shopping": [("shop", "supermarket"), ("shop", "convenience"), ("shop", "mall")],
    "bus": [("amenity", "bus_station"), ("highway", "bus_stop")],
    "train": [("railway", "station")],
    "subway": [("railway", "subway_entrance"), ("station", "subway")],
    "transit": [
        ("amenity", "bus_station"),
        ("railway", "station"),
        ("railway", "subway_entrance"),
    ],
    "parking": [("amenity", "parking")],
    "toilet": [("amenity", "toilets")],
    "ramen": [("amenity", "restaurant"), ("cuisine", "ramen")],
    "korean": [("amenity", "restaurant"), ("cuisine", "korean")],
    "vietnamese": [("amenity", "restaurant"), ("cuisine", "vietnamese")],
    "japanese": [("amenity", "restaurant"), ("cuisine", "japanese")],
}


@dataclass
class Place:
    """A point of interest."""
    id: str
    name: str
    lat: float
    lon: float
    category: str
    distance_m: Optional[float] = None
    tags: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "category": self.category,
            "distance_m": self.distance_m,
            "opening_hours": self.tags.get("opening_hours"),
            "phone": self.tags.get("phone"),
            "website": self.tags.get("website"),
            "cuisine": self.tags.get("cuisine"),
        }


@dataclass
class RouteStep:
    """A step in a route."""
    instruction: str
    distance_m: float
    duration_s: float
    name: str = ""
    
    def to_dict(self) -> dict:
        return {
            "instruction": self.instruction,
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
            "road_name": self.name,
        }


@dataclass
class Route:
    """A route between two points."""
    distance_m: float
    duration_s: float
    steps: list[RouteStep]
    geometry: Optional[str] = None  # Encoded polyline
    
    def to_dict(self) -> dict:
        return {
            "distance_m": self.distance_m,
            "duration_s": self.duration_s,
            "duration_min": round(self.duration_s / 60, 1),
            "steps": [s.to_dict() for s in self.steps],
        }


# =============================================================================
# Overpass API - Nearby Places
# =============================================================================

def _build_overpass_query(
    lat: float,
    lon: float,
    radius_m: int,
    categories: list[str],
) -> str:
    """Build Overpass QL query for nearby places."""
    
    # Collect all OSM tag filters
    filters = []
    for cat in categories:
        cat_lower = cat.lower()
        if cat_lower in CATEGORY_TO_OSM:
            filters.extend(CATEGORY_TO_OSM[cat_lower])
        else:
            # Default to amenity search
            filters.append(("amenity", cat_lower))
    
    # Build query parts for each filter
    query_parts = []
    for key, value in filters:
        query_parts.append(
            f'node["{key}"="{value}"](around:{radius_m},{lat},{lon});'
        )
        query_parts.append(
            f'way["{key}"="{value}"](around:{radius_m},{lat},{lon});'
        )
    
    union = "\n  ".join(query_parts)
    
    query = f"""
[out:json][timeout:25];
(
  {union}
);
out center body;
"""
    return query


def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in meters."""
    import math
    R = 6371000  # Earth radius in meters
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c


def _parse_overpass_response(
    data: dict,
    origin_lat: float,
    origin_lon: float,
) -> list[Place]:
    """Parse Overpass API response into Place objects."""
    places = []
    
    for element in data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        
        if not name:
            continue  # Skip unnamed places
        
        # Get coordinates (nodes have lat/lon, ways have center)
        if element["type"] == "node":
            lat = element["lat"]
            lon = element["lon"]
        elif "center" in element:
            lat = element["center"]["lat"]
            lon = element["center"]["lon"]
        else:
            continue
        
        # Determine category from tags
        category = (
            tags.get("amenity")
            or tags.get("tourism")
            or tags.get("shop")
            or tags.get("railway")
            or "place"
        )
        
        # Calculate distance
        distance = _haversine_distance(origin_lat, origin_lon, lat, lon)
        
        place = Place(
            id=f"osm:{element['type']}:{element['id']}",
            name=name,
            lat=lat,
            lon=lon,
            category=category,
            distance_m=round(distance, 1),
            tags=tags,
        )
        places.append(place)
    
    # Sort by distance
    places.sort(key=lambda p: p.distance_m or float("inf"))
    
    return places


async def get_nearby_places(
    lat: float,
    lon: float,
    radius_m: int = 1000,
    categories: Optional[list[str]] = None,
    limit: int = 10,
    language: str = "en",
) -> dict[str, Any]:
    """
    Search for nearby places using OpenStreetMap Overpass API.
    
    Args:
        lat: Latitude of search center
        lon: Longitude of search center
        radius_m: Search radius in meters (default 1000)
        categories: List of place categories to search for
        limit: Maximum number of results
        language: Preferred language for results
    
    Returns:
        Dictionary with places list and metadata
    """
    if categories is None:
        categories = ["restaurant", "cafe", "attraction"]
    
    query = _build_overpass_query(lat, lon, radius_m, categories)
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.post(
                OVERPASS_API_URL,
                data={"data": query},
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
            
        except httpx.TimeoutException:
            logger.error("Overpass API timeout")
            return {"error": "Search timed out", "places": []}
        except httpx.HTTPStatusError as e:
            logger.error(f"Overpass API error: {e}")
            return {"error": f"API error: {e.response.status_code}", "places": []}
        except Exception as e:
            logger.error(f"Overpass request failed: {e}")
            return {"error": str(e), "places": []}
    
    places = _parse_overpass_response(data, lat, lon)
    places = places[:limit]
    
    return {
        "places": [p.to_dict() for p in places],
        "total_found": len(places),
        "search_center": {"lat": lat, "lon": lon},
        "search_radius_m": radius_m,
        "categories": categories,
    }


# =============================================================================
# OSRM - Routing
# =============================================================================

def _parse_osrm_maneuver(maneuver: dict) -> str:
    """Convert OSRM maneuver to human-readable instruction."""
    maneuver_type = maneuver.get("type", "")
    modifier = maneuver.get("modifier", "")
    
    instructions = {
        "depart": "Start",
        "arrive": "Arrive at destination",
        "turn": f"Turn {modifier}",
        "continue": "Continue straight",
        "merge": f"Merge {modifier}",
        "fork": f"Take the {modifier} fork",
        "roundabout": f"Take the roundabout, exit {modifier}",
        "rotary": f"Enter the rotary",
        "new name": f"Continue onto",
        "end of road": f"At end of road, turn {modifier}",
    }
    
    return instructions.get(maneuver_type, f"{maneuver_type} {modifier}".strip())


async def get_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    mode: str = "walking",
    language: str = "en",
) -> dict[str, Any]:
    """
    Get route between two points using OSRM.
    
    Args:
        start_lat: Starting point latitude
        start_lon: Starting point longitude
        end_lat: Destination latitude
        end_lon: Destination longitude
        mode: Travel mode - "walking", "driving", or "cycling"
        language: Language for instructions
    
    Returns:
        Dictionary with route details
    """
    # Map mode to OSRM profile
    profile_map = {
        "walking": "foot",
        "walk": "foot",
        "foot": "foot",
        "driving": "car",
        "drive": "car",
        "car": "car",
        "cycling": "bike",
        "bike": "bike",
        "bicycle": "bike",
    }
    profile = profile_map.get(mode.lower(), "foot")
    
    # OSRM expects lon,lat order
    coords = f"{start_lon},{start_lat};{end_lon},{end_lat}"
    url = f"{OSRM_API_URL}/route/v1/{profile}/{coords}"
    
    params = {
        "overview": "full",
        "steps": "true",
        "geometries": "polyline",
    }
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
            
        except httpx.TimeoutException:
            logger.error("OSRM API timeout")
            return {"error": "Routing timed out"}
        except httpx.HTTPStatusError as e:
            logger.error(f"OSRM API error: {e}")
            return {"error": f"Routing error: {e.response.status_code}"}
        except Exception as e:
            logger.error(f"OSRM request failed: {e}")
            return {"error": str(e)}
    
    if data.get("code") != "Ok":
        return {"error": data.get("message", "Routing failed")}
    
    route_data = data["routes"][0]
    
    # Parse steps
    steps = []
    for leg in route_data.get("legs", []):
        for step in leg.get("steps", []):
            maneuver = step.get("maneuver", {})
            instruction = _parse_osrm_maneuver(maneuver)
            road_name = step.get("name", "")
            
            if road_name:
                instruction = f"{instruction} {road_name}"
            
            steps.append(RouteStep(
                instruction=instruction,
                distance_m=step.get("distance", 0),
                duration_s=step.get("duration", 0),
                name=road_name,
            ))
    
    route = Route(
        distance_m=route_data["distance"],
        duration_s=route_data["duration"],
        steps=steps,
        geometry=route_data.get("geometry"),
    )
    
    return {
        "route": route.to_dict(),
        "start": {"lat": start_lat, "lon": start_lon},
        "end": {"lat": end_lat, "lon": end_lon},
        "mode": mode,
    }


# =============================================================================
# Nominatim - Geocoding
# =============================================================================

async def geocode(
    query: str,
    language: str = "en",
    limit: int = 5,
) -> dict[str, Any]:
    """
    Geocode an address or place name using Nominatim.
    
    Args:
        query: Address or place name to search
        language: Preferred language
        limit: Maximum results
    
    Returns:
        Dictionary with geocoding results
    """
    url = f"{NOMINATIM_API_URL}/search"
    
    params = {
        "q": query,
        "format": "json",
        "limit": limit,
        "accept-language": language,
        "addressdetails": 1,
    }
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
            
        except Exception as e:
            logger.error(f"Nominatim request failed: {e}")
            return {"error": str(e), "results": []}
    
    results = []
    for item in data:
        results.append({
            "name": item.get("display_name"),
            "lat": float(item.get("lat", 0)),
            "lon": float(item.get("lon", 0)),
            "type": item.get("type"),
            "importance": item.get("importance"),
        })
    
    return {"results": results, "query": query}


async def reverse_geocode(
    lat: float,
    lon: float,
    language: str = "en",
) -> dict[str, Any]:
    """
    Reverse geocode coordinates to an address.
    
    Args:
        lat: Latitude
        lon: Longitude
        language: Preferred language
    
    Returns:
        Dictionary with address details
    """
    url = f"{NOMINATIM_API_URL}/reverse"
    
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "accept-language": language,
        "addressdetails": 1,
    }
    
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        try:
            response = await client.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            data = response.json()
            
        except Exception as e:
            logger.error(f"Nominatim reverse geocode failed: {e}")
            return {"error": str(e)}
    
    address = data.get("address", {})
    
    return {
        "display_name": data.get("display_name"),
        "address": {
            "road": address.get("road"),
            "city": address.get("city") or address.get("town") or address.get("village"),
            "state": address.get("state"),
            "country": address.get("country"),
            "postcode": address.get("postcode"),
        },
        "lat": lat,
        "lon": lon,
    }


# =============================================================================
# Tool Executor - Maps tool calls to real API calls
# =============================================================================

TOOL_REGISTRY = {
    "get_nearby_places": get_nearby_places,
    "get_route": get_route,
    "geocode": geocode,
    "reverse_geocode": reverse_geocode,
}


async def execute_tool(name: str, arguments: dict) -> dict[str, Any]:
    """
    Execute a navigation tool by name.
    
    Args:
        name: Tool name (e.g., "get_nearby_places")
        arguments: Tool arguments
    
    Returns:
        Tool execution result
    """
    if name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {name}"}
    
    tool_fn = TOOL_REGISTRY[name]
    
    try:
        result = await tool_fn(**arguments)
        return result
    except TypeError as e:
        logger.error(f"Tool {name} argument error: {e}")
        return {"error": f"Invalid arguments: {e}"}
    except Exception as e:
        logger.error(f"Tool {name} execution failed: {e}")
        return {"error": f"Execution failed: {e}"}


def execute_tool_sync(name: str, arguments: dict) -> dict[str, Any]:
    """Synchronous wrapper for execute_tool."""
    return asyncio.run(execute_tool(name, arguments))


# =============================================================================
# Convenience functions for menu/local info
# =============================================================================

async def get_menu_item_info(
    item_name: str,
    language: str = "en",
) -> dict[str, Any]:
    """
    Get information about a menu item or dish.
    
    Note: This uses a simple knowledge base. For production,
    integrate with a food database API.
    """
    # Simple knowledge base for common dishes
    DISH_INFO = {
        "pho": {
            "name": "Phở",
            "description": "Vietnamese noodle soup with beef or chicken broth",
            "origin": "Vietnam",
            "ingredients": ["rice noodles", "broth", "herbs", "meat"],
            "dietary": ["gluten-free-option"],
        },
        "ramen": {
            "name": "ラーメン (Ramen)",
            "description": "Japanese noodle soup with various broths and toppings",
            "origin": "Japan",
            "ingredients": ["wheat noodles", "broth", "chashu", "egg", "nori"],
            "dietary": ["contains-gluten", "contains-egg"],
        },
        "bibimbap": {
            "name": "비빔밥 (Bibimbap)",
            "description": "Korean mixed rice with vegetables, meat, and gochujang",
            "origin": "South Korea",
            "ingredients": ["rice", "vegetables", "meat", "egg", "gochujang"],
            "dietary": ["can-be-vegetarian"],
        },
        "banh mi": {
            "name": "Bánh mì",
            "description": "Vietnamese sandwich on French baguette",
            "origin": "Vietnam",
            "ingredients": ["baguette", "pâté", "meat", "pickled vegetables", "cilantro"],
            "dietary": ["contains-gluten"],
        },
    }
    
    item_lower = item_name.lower().replace(" ", "").replace("-", "")
    
    for key, info in DISH_INFO.items():
        if key.replace(" ", "") in item_lower or item_lower in key.replace(" ", ""):
            return {"found": True, "item": info}
    
    return {
        "found": False,
        "message": f"No information found for '{item_name}'",
        "suggestion": "Try asking about pho, ramen, bibimbap, or banh mi",
    }


async def get_local_etiquette(
    country: str,
    topic: Optional[str] = None,
    language: str = "en",
) -> dict[str, Any]:
    """
    Get local etiquette and customs information.
    
    Note: Uses a simple knowledge base. For production,
    integrate with a cultural database.
    """
    ETIQUETTE = {
        "japan": {
            "greetings": "Bow when greeting. Depth indicates respect level.",
            "dining": "Say 'itadakimasu' before eating. Don't stick chopsticks upright in rice.",
            "temples": "Remove shoes before entering. Be quiet and respectful.",
            "transit": "Queue orderly. Give up seats to elderly. Keep quiet on trains.",
            "tipping": "Tipping is not customary and can be considered rude.",
        },
        "korea": {
            "greetings": "Bow slightly when greeting. Use both hands when giving/receiving.",
            "dining": "Wait for eldest to start. Pour drinks for others, not yourself.",
            "temples": "Remove shoes. Dress modestly. No photos during ceremonies.",
            "transit": "Give up priority seats. Avoid eating on subway.",
            "tipping": "Tipping is not expected. Service is included.",
        },
        "vietnam": {
            "greetings": "Slight bow or handshake. Use both hands for business cards.",
            "dining": "Wait for host. Leave a little food on plate when full.",
            "temples": "Remove shoes. Cover shoulders and knees. No pointing at Buddha.",
            "transit": "Traffic is chaotic. Walk steadily, don't make sudden moves.",
            "tipping": "Small tips appreciated but not required. Round up bills.",
        },
    }
    
    country_lower = country.lower()
    
    if country_lower not in ETIQUETTE:
        return {
            "found": False,
            "message": f"No etiquette info for '{country}'",
            "available_countries": list(ETIQUETTE.keys()),
        }
    
    info = ETIQUETTE[country_lower]
    
    if topic:
        topic_lower = topic.lower()
        if topic_lower in info:
            return {
                "found": True,
                "country": country,
                "topic": topic,
                "advice": info[topic_lower],
            }
        return {
            "found": False,
            "message": f"No info on '{topic}' for {country}",
            "available_topics": list(info.keys()),
        }
    
    return {
        "found": True,
        "country": country,
        "etiquette": info,
    }


# Add to tool registry
TOOL_REGISTRY["get_menu_item_info"] = get_menu_item_info
TOOL_REGISTRY["get_local_etiquette"] = get_local_etiquette


# =============================================================================
# CLI for testing
# =============================================================================

if __name__ == "__main__":
    import sys
    
    async def test_tools():
        print("Testing Navigation Tools\n" + "=" * 40)
        
        # Test nearby places
        print("\n1. Testing get_nearby_places (Tokyo - ramen)...")
        result = await get_nearby_places(
            lat=35.6762,
            lon=139.6503,
            radius_m=500,
            categories=["ramen", "restaurant"],
            limit=5,
        )
        print(f"   Found {result.get('total_found', 0)} places")
        for place in result.get("places", [])[:3]:
            print(f"   - {place['name']} ({place['distance_m']}m)")
        
        # Test routing
        print("\n2. Testing get_route (Tokyo Station to Shibuya)...")
        result = await get_route(
            start_lat=35.6812,
            start_lon=139.7671,
            end_lat=35.6580,
            end_lon=139.7016,
            mode="walking",
        )
        if "route" in result:
            route = result["route"]
            print(f"   Distance: {route['distance_m']}m")
            print(f"   Duration: {route['duration_min']} min")
            print(f"   Steps: {len(route['steps'])}")
        else:
            print(f"   Error: {result.get('error')}")
        
        # Test geocoding
        print("\n3. Testing geocode...")
        result = await geocode("Eiffel Tower, Paris")
        if result.get("results"):
            loc = result["results"][0]
            print(f"   {loc['name'][:50]}...")
            print(f"   Coords: {loc['lat']}, {loc['lon']}")
        
        # Test reverse geocoding
        print("\n4. Testing reverse_geocode...")
        result = await reverse_geocode(lat=35.6762, lon=139.6503)
        print(f"   {result.get('display_name', 'N/A')[:60]}...")
        
        # Test menu info
        print("\n5. Testing get_menu_item_info...")
        result = await get_menu_item_info("pho")
        if result.get("found"):
            item = result["item"]
            print(f"   {item['name']}: {item['description'][:50]}...")
        
        print("\n" + "=" * 40)
        print("All tests completed!")
    
    asyncio.run(test_tools())
