"""
Edge case tests for boundary conditions and security validation.
Implements T114 - Comprehensive edge case testing.
"""
import pytest
from app.core.validation import (
    validate_email,
    validate_language_code,
    validate_latitude,
    validate_longitude,
    validate_radius,
    validate_text_length,
    validate_password_strength,
    sanitize_destination,
    validate_confidence_score,
    validate_context_category,
    ValidationError
)


class TestEmailValidation:
    """Test email validation edge cases."""
    
    def test_valid_emails(self):
        """Valid email formats should pass."""
        assert validate_email("user@example.com") == "user@example.com"
        assert validate_email("User@Example.Com") == "user@example.com"  # Lowercase conversion
        assert validate_email("user.name+tag@example.co.uk") == "user.name+tag@example.co.uk"
        assert validate_email("user_name@sub-domain.example.com") == "user_name@sub-domain.example.com"
    
    def test_invalid_email_format(self):
        """Invalid email formats should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid email format"):
            validate_email("not-an-email")
        
        with pytest.raises(ValidationError, match="Invalid email format"):
            validate_email("missing@domain")
        
        with pytest.raises(ValidationError, match="Invalid email format"):
            validate_email("@missing-local.com")
        
        with pytest.raises(ValidationError, match="Invalid email format"):
            validate_email("spaces in@email.com")
    
    def test_email_max_length(self):
        """Emails exceeding 255 characters should be rejected."""
        long_email = "a" * 250 + "@example.com"  # 262 characters
        with pytest.raises(ValidationError, match="too long"):
            validate_email(long_email)
    
    def test_sql_injection_attempt(self):
        """SQL injection attempts in email should be rejected."""
        with pytest.raises(ValidationError, match="Invalid email format"):
            validate_email("admin'--@example.com")
        
        with pytest.raises(ValidationError, match="Invalid email format"):
            validate_email("user@example.com; DROP TABLE users;--")


class TestLanguageCodeValidation:
    """Test language code validation edge cases."""
    
    def test_valid_language_codes(self):
        """Valid ISO 639-1 codes should pass."""
        assert validate_language_code("en") == "en"
        assert validate_language_code("ja") == "ja"
        assert validate_language_code("ES") == "es"  # Lowercase conversion
        assert validate_language_code("zh-CN") == "zh-cn"  # Extended format
    
    def test_invalid_language_codes(self):
        """Invalid language codes should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid language code"):
            validate_language_code("e")  # Too short
        
        with pytest.raises(ValidationError, match="Invalid language code"):
            validate_language_code("eng123")  # Contains digits
        
        with pytest.raises(ValidationError, match="Invalid language code"):
            validate_language_code("en-US-extra")  # Too long


class TestCoordinateValidation:
    """Test latitude/longitude validation edge cases."""
    
    def test_valid_coordinates(self):
        """Valid coordinates should pass."""
        assert validate_latitude(0.0) == 0.0
        assert validate_latitude(45.5) == 45.5
        assert validate_latitude(-45.5) == -45.5
        assert validate_latitude(90.0) == 90.0
        assert validate_latitude(-90.0) == -90.0
        
        assert validate_longitude(0.0) == 0.0
        assert validate_longitude(180.0) == 180.0
        assert validate_longitude(-180.0) == -180.0
    
    def test_latitude_out_of_range(self):
        """Latitudes outside [-90, 90] should raise ValidationError."""
        with pytest.raises(ValidationError, match="between -90 and 90"):
            validate_latitude(91.0)
        
        with pytest.raises(ValidationError, match="between -90 and 90"):
            validate_latitude(-91.0)
        
        with pytest.raises(ValidationError, match="between -90 and 90"):
            validate_latitude(180.0)
    
    def test_longitude_out_of_range(self):
        """Longitudes outside [-180, 180] should raise ValidationError."""
        with pytest.raises(ValidationError, match="between -180 and 180"):
            validate_longitude(181.0)
        
        with pytest.raises(ValidationError, match="between -180 and 180"):
            validate_longitude(-181.0)
        
        with pytest.raises(ValidationError, match="between -180 and 180"):
            validate_longitude(360.0)


class TestRadiusValidation:
    """Test radius validation edge cases."""
    
    def test_valid_radius(self):
        """Valid radius values should pass."""
        assert validate_radius(100.0) == 100.0
        assert validate_radius(5000.0) == 5000.0
        assert validate_radius(50000.0) == 50000.0  # Max 50km
    
    def test_negative_radius(self):
        """Negative radius should raise ValidationError."""
        with pytest.raises(ValidationError, match="must be positive"):
            validate_radius(-100.0)
        
        with pytest.raises(ValidationError, match="must be positive"):
            validate_radius(0.0)
    
    def test_radius_exceeds_max(self):
        """Radius exceeding 50km should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot exceed 50000 meters"):
            validate_radius(50001.0)
        
        with pytest.raises(ValidationError, match="cannot exceed 50000 meters"):
            validate_radius(100000.0)


class TestTextLengthValidation:
    """Test text length validation edge cases."""
    
    def test_valid_text(self):
        """Valid text should pass."""
        assert validate_text_length("Hello", max_length=10, field_name="text") == "Hello"
        assert validate_text_length("A" * 100, max_length=100, field_name="text") == "A" * 100
    
    def test_empty_text(self):
        """Empty text should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_text_length("", max_length=10, field_name="text")
        
        with pytest.raises(ValidationError, match="cannot be empty"):
            validate_text_length("   ", max_length=10, field_name="text")
    
    def test_text_exceeds_max_length(self):
        """Text exceeding max length should raise ValidationError."""
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            validate_text_length("A" * 101, max_length=100, field_name="description")


class TestPasswordStrength:
    """Test password strength validation edge cases."""
    
    def test_valid_passwords(self):
        """Valid passwords should pass."""
        assert validate_password_strength("password123") == "password123"
        assert validate_password_strength("StrongP@ss1") == "StrongP@ss1"
        assert validate_password_strength("a1" + "b" * 126) == "a1" + "b" * 126  # Max 128
    
    def test_password_too_short(self):
        """Passwords shorter than 8 characters should be rejected."""
        with pytest.raises(ValidationError, match="at least 8 characters"):
            validate_password_strength("pass123")
        
        with pytest.raises(ValidationError, match="at least 8 characters"):
            validate_password_strength("abc")
    
    def test_password_too_long(self):
        """Passwords longer than 128 characters should be rejected."""
        with pytest.raises(ValidationError, match="cannot exceed 128 characters"):
            validate_password_strength("a1" + "b" * 127)  # 129 characters
    
    def test_password_missing_letter(self):
        """Passwords without letters should be rejected."""
        with pytest.raises(ValidationError, match="at least one letter"):
            validate_password_strength("12345678")
        
        with pytest.raises(ValidationError, match="at least one letter"):
            validate_password_strength("!@#$%^&*123")
    
    def test_password_missing_digit(self):
        """Passwords without digits should be rejected."""
        with pytest.raises(ValidationError, match="at least one digit"):
            validate_password_strength("password")
        
        with pytest.raises(ValidationError, match="at least one digit"):
            validate_password_strength("NoNumbers!")


class TestDestinationSanitization:
    """Test destination sanitization for injection prevention."""
    
    def test_valid_destinations(self):
        """Valid destination names should pass."""
        assert sanitize_destination("Tokyo") == "Tokyo"
        assert sanitize_destination("New York, NY") == "New York, NY"
        assert sanitize_destination("Saint-Pierre (France)") == "Saint-Pierre (France)"
        assert sanitize_destination("Café-Restaurant") == "Café-Restaurant"
    
    def test_empty_destination(self):
        """Empty destination should raise ValidationError."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            sanitize_destination("")
        
        with pytest.raises(ValidationError, match="cannot be empty"):
            sanitize_destination("   ")
    
    def test_destination_too_long(self):
        """Destinations exceeding 255 characters should be rejected."""
        with pytest.raises(ValidationError, match="too long"):
            sanitize_destination("A" * 256)
    
    def test_sql_injection_blocked(self):
        """SQL injection attempts should be blocked."""
        with pytest.raises(ValidationError, match="contains invalid characters"):
            sanitize_destination("Tokyo'; DROP TABLE users;--")
        
        with pytest.raises(ValidationError, match="contains invalid characters"):
            sanitize_destination("Tokyo' OR '1'='1")
    
    def test_xss_injection_blocked(self):
        """XSS injection attempts should be blocked."""
        with pytest.raises(ValidationError, match="contains invalid characters"):
            sanitize_destination("<script>alert('xss')</script>")
        
        with pytest.raises(ValidationError, match="contains invalid characters"):
            sanitize_destination("Tokyo<img src=x onerror=alert(1)>")
    
    def test_special_chars_blocked(self):
        """Dangerous special characters should be blocked."""
        with pytest.raises(ValidationError, match="contains invalid characters"):
            sanitize_destination("Tokyo;rm -rf /")
        
        with pytest.raises(ValidationError, match="contains invalid characters"):
            sanitize_destination("Tokyo|cat /etc/passwd")


class TestConfidenceScoreValidation:
    """Test confidence score validation edge cases."""
    
    def test_valid_confidence_scores(self):
        """Valid confidence scores should pass."""
        assert validate_confidence_score(0.0) == 0.0
        assert validate_confidence_score(50.0) == 50.0
        assert validate_confidence_score(100.0) == 100.0
        assert validate_confidence_score(None) is None  # Optional
    
    def test_confidence_out_of_range(self):
        """Confidence scores outside [0, 100] should be rejected."""
        with pytest.raises(ValidationError, match="between 0 and 100"):
            validate_confidence_score(-0.1)
        
        with pytest.raises(ValidationError, match="between 0 and 100"):
            validate_confidence_score(100.1)
        
        with pytest.raises(ValidationError, match="between 0 and 100"):
            validate_confidence_score(200.0)


class TestContextCategoryValidation:
    """Test context category validation edge cases."""
    
    def test_valid_categories(self):
        """Valid context categories should pass."""
        assert validate_context_category("restaurant") == "restaurant"
        assert validate_context_category("transit") == "transit"
        assert validate_context_category("lodging") == "lodging"
        assert validate_context_category("general") == "general"
    
    def test_invalid_categories(self):
        """Invalid context categories should be rejected."""
        with pytest.raises(ValidationError, match="Invalid context category"):
            validate_context_category("shopping")
        
        with pytest.raises(ValidationError, match="Invalid context category"):
            validate_context_category("RESTAURANT")  # Case sensitive
        
        with pytest.raises(ValidationError, match="Invalid context category"):
            validate_context_category("")


class TestBoundaryConditions:
    """Test various boundary conditions."""
    
    def test_null_inputs(self):
        """Null/None inputs should be handled properly."""
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            validate_email(None)
        
        with pytest.raises((ValidationError, TypeError, AttributeError)):
            validate_language_code(None)
        
        # Confidence score allows None
        assert validate_confidence_score(None) is None
    
    def test_whitespace_handling(self):
        """Whitespace should be handled appropriately."""
        # Email trimming not implemented, should fail with spaces
        with pytest.raises(ValidationError):
            validate_email(" user@example.com ")
        
        # Destination allows internal spaces
        assert "New York" in sanitize_destination("New York")
        
        # Empty after strip should fail
        with pytest.raises(ValidationError):
            validate_text_length("   ", max_length=10, field_name="text")
    
    def test_unicode_handling(self):
        """Unicode characters should be handled properly."""
        # Email with unicode
        with pytest.raises(ValidationError):
            validate_email("用户@example.com")
        
        # Destination with unicode (should be blocked by alphanumeric check)
        with pytest.raises(ValidationError):
            sanitize_destination("東京")
    
    def test_exact_boundary_values(self):
        """Test exact boundary values."""
        # Latitude boundaries
        assert validate_latitude(90.0) == 90.0
        assert validate_latitude(-90.0) == -90.0
        
        # Longitude boundaries
        assert validate_longitude(180.0) == 180.0
        assert validate_longitude(-180.0) == -180.0
        
        # Radius boundary
        assert validate_radius(50000.0) == 50000.0
        
        # Confidence boundaries
        assert validate_confidence_score(0.0) == 0.0
        assert validate_confidence_score(100.0) == 100.0
        
        # Password length boundaries
        assert validate_password_strength("pass1234") == "pass1234"  # Exactly 8
        assert len(validate_password_strength("a1" + "b" * 126)) == 128  # Exactly 128
