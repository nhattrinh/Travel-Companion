"""
Input validation utilities for security hardening
"""
import re
from typing import Optional
from fastapi import HTTPException, status


class ValidationError(Exception):
    """Custom validation error"""
    pass


def validate_email(email: str) -> str:
    """
    Validate email format
    
    Args:
        email: Email address to validate
        
    Returns:
        Validated email in lowercase
        
    Raises:
        ValidationError: If email format is invalid
    """
    email = email.strip().lower()
    
    # Basic email regex pattern
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(pattern, email):
        raise ValidationError("Invalid email format")
    
    if len(email) > 255:
        raise ValidationError("Email too long (max 255 characters)")
    
    return email


def validate_language_code(lang: str) -> str:
    """
    Validate language code (ISO 639-1 format)
    
    Args:
        lang: Language code (2-8 characters)
        
    Returns:
        Validated language code in lowercase
        
    Raises:
        ValidationError: If language code is invalid
    """
    lang = lang.strip().lower()
    
    if not re.match(r'^[a-z]{2,8}$', lang):
        raise ValidationError("Invalid language code format (expected 2-8 lowercase letters)")
    
    return lang


def validate_latitude(lat: float) -> float:
    """
    Validate latitude coordinate
    
    Args:
        lat: Latitude value
        
    Returns:
        Validated latitude
        
    Raises:
        ValidationError: If latitude is out of range
    """
    if not -90.0 <= lat <= 90.0:
        raise ValidationError(f"Latitude {lat} out of range (must be -90 to 90)")
    
    return lat


def validate_longitude(lon: float) -> float:
    """
    Validate longitude coordinate
    
    Args:
        lon: Longitude value
        
    Returns:
        Validated longitude
        
    Raises:
        ValidationError: If longitude is out of range
    """
    if not -180.0 <= lon <= 180.0:
        raise ValidationError(f"Longitude {lon} out of range (must be -180 to 180)")
    
    return lon


def validate_radius(radius: float, max_radius: float = 50000.0) -> float:
    """
    Validate search radius (in meters)
    
    Args:
        radius: Search radius
        max_radius: Maximum allowed radius (default 50km)
        
    Returns:
        Validated radius
        
    Raises:
        ValidationError: If radius is invalid
    """
    if radius <= 0:
        raise ValidationError("Radius must be positive")
    
    if radius > max_radius:
        raise ValidationError(f"Radius {radius}m exceeds maximum {max_radius}m")
    
    return radius


def validate_text_length(text: str, max_length: int, field_name: str = "Text") -> str:
    """
    Validate text length
    
    Args:
        text: Text to validate
        max_length: Maximum allowed length
        field_name: Name of field for error message
        
    Returns:
        Validated text
        
    Raises:
        ValidationError: If text is too long or empty
    """
    if not text or not text.strip():
        raise ValidationError(f"{field_name} cannot be empty")
    
    if len(text) > max_length:
        raise ValidationError(
            f"{field_name} too long ({len(text)} chars, max {max_length})"
        )
    
    return text.strip()


def validate_password_strength(password: str) -> str:
    """
    Validate password meets security requirements
    
    Args:
        password: Password to validate
        
    Returns:
        Validated password
        
    Raises:
        ValidationError: If password doesn't meet requirements
    """
    if len(password) < 8:
        raise ValidationError("Password must be at least 8 characters")
    
    if len(password) > 128:
        raise ValidationError("Password too long (max 128 characters)")
    
    # Check for at least one letter and one number
    has_letter = any(c.isalpha() for c in password)
    has_digit = any(c.isdigit() for c in password)
    
    if not (has_letter and has_digit):
        raise ValidationError(
            "Password must contain at least one letter and one number"
        )
    
    return password


def sanitize_destination(destination: str) -> str:
    """
    Sanitize destination string
    
    Args:
        destination: Destination name
        
    Returns:
        Sanitized destination
        
    Raises:
        ValidationError: If destination is invalid
    """
    destination = destination.strip()
    
    if not destination:
        raise ValidationError("Destination cannot be empty")
    
    if len(destination) > 255:
        raise ValidationError("Destination too long (max 255 characters)")
    
    # Remove potentially dangerous characters
    # Allow letters, numbers, spaces, commas, hyphens, and basic punctuation
    if not re.match(r'^[a-zA-Z0-9\s,.\-()]+$', destination):
        raise ValidationError(
            "Destination contains invalid characters (only letters, numbers, spaces, commas, periods, hyphens, and parentheses allowed)"
        )
    
    return destination


def validate_confidence_score(confidence: Optional[int]) -> Optional[int]:
    """
    Validate confidence score (0-100 scale)
    
    Args:
        confidence: Confidence score
        
    Returns:
        Validated confidence score
        
    Raises:
        ValidationError: If confidence is out of range
    """
    if confidence is None:
        return None
    
    if not 0 <= confidence <= 100:
        raise ValidationError(f"Confidence score {confidence} out of range (must be 0-100)")
    
    return confidence


def validate_context_category(context: str) -> str:
    """
    Validate context category
    
    Args:
        context: Context category
        
    Returns:
        Validated context
        
    Raises:
        ValidationError: If context is invalid
    """
    valid_contexts = {"restaurant", "transit", "lodging", "general"}
    context = context.strip().lower()
    
    if context not in valid_contexts:
        raise ValidationError(
            f"Invalid context '{context}' (must be one of: {', '.join(valid_contexts)})"
        )
    
    return context
