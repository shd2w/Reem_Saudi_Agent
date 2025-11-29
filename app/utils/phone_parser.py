import re
from typing import Optional, Tuple

def extract_saudi_phone(text: str) -> Optional[str]:
    """Extracts a Saudi Arabian phone number from a string.

    Handles formats like:
    - 05xxxxxxxx
    - +9665xxxxxxxx
    - 9665xxxxxxxx
    - 5xxxxxxxx

    Args:
        text: The text to search for a phone number.

    Returns:
        The extracted phone number in the format '9665xxxxxxxx', or None.
    """
    # Regex to find Saudi phone numbers in various formats
    # It looks for a 5 followed by 8 digits, with optional prefixes like 0, +966, or 966.
    pattern = re.compile(r'(?:\+?966)?(0)?(5[0-9]{8})')
    
    match = pattern.search(text)
    
    if match:
        # The last group captures the core '5xxxxxxxx' part
        phone_part = match.groups()[-1]
        return f"966{phone_part}"
        
    return None


def extract_generic_phone(text: str) -> Optional[str]:
    """Extract a generic phone-like sequence of digits (9-15 digits).

    Returns the first occurrence of 9-15 consecutive digits, or None.
    """
    if not text:
        return None
    m = re.search(r"(\d{9,15})", text)
    return m.group(1) if m else None


def normalize_phone_digits(phone: str) -> Optional[str]:
    """Normalize a phone to digits-only and return its last 10 digits if available.

    This is a pragmatic, country-agnostic comparison helper. For correlation, we
    often only need to check if two numbers refer to the same line by comparing
    their last 10 digits.
    """
    if not phone:
        return None
    digits = re.sub(r"\D", "", phone)
    if not digits:
        return None
    return digits[-10:] if len(digits) > 10 else digits


def is_valid_phone(phone: str) -> Tuple[bool, Optional[str]]:
    """
    Validate if phone number is likely a valid Saudi/Egyptian number.
    
    Args:
        phone: Phone number to validate
        
    Returns:
        Tuple of (is_valid, reason)
        - (True, None) if valid
        - (False, reason) if invalid
    
    Examples:
        "9665012345678" → (True, None)
        "201234567890" → (True, None) 
        "220796299538458" → (False, "Too long (15 digits)")
        "123" → (False, "Too short (3 digits)")
    """
    if not phone:
        return False, "Empty phone number"
    
    # Remove all non-digits
    digits = re.sub(r"\D", "", phone)
    
    if not digits:
        return False, "No digits found"
    
    digit_count = len(digits)
    
    # Check length constraints
    if digit_count < 9:
        return False, f"Too short ({digit_count} digits, min 9)"
    
    if digit_count > 13:
        return False, f"Too long ({digit_count} digits, max 13)"
    
    # Saudi Arabia validation (9665XXXXXXXX or 05XXXXXXXX)
    if digits.startswith("966"):
        # Should be 12 digits total: 966 + 9 digits
        if digit_count != 12:
            return False, f"Invalid Saudi format ({digit_count} digits, expected 12)"
        # Next digit after 966 should be 5
        if not digits[3:].startswith("5"):
            return False, "Saudi numbers must start with 5 after 966"
        return True, None
    
    # Egypt validation (20XXXXXXXXXX)
    if digits.startswith("20"):
        # Should be 12-13 digits total: 20 + 10-11 digits
        if digit_count < 12 or digit_count > 13:
            return False, f"Invalid Egypt format ({digit_count} digits, expected 12-13)"
        return True, None
    
    # Local Saudi format (05XXXXXXXX or 5XXXXXXXX)
    if digits.startswith("5") or digits.startswith("05"):
        # Should be 9-10 digits
        if digit_count == 9 or digit_count == 10:
            return True, None
        return False, f"Invalid local Saudi format ({digit_count} digits, expected 9-10)"
    
    # Local Egypt format (0XXXXXXXXXX)
    if digits.startswith("0") and digit_count >= 10 and digit_count <= 11:
        return True, None
    
    # Unknown format but reasonable length (9-12 digits)
    if digit_count >= 9 and digit_count <= 12:
        return True, None  # Give it a chance
    
    return False, f"Unknown format ({digit_count} digits)"


def remove_country_code(phone: str) -> str:
    """Remove country code from phone number for database search.
    
    Handles:
    - Saudi Arabia: +966, 0966, 966 → returns local format (05xxxxxxxx)
    - Egypt: +20, 020, 20 → returns local format (0xxxxxxxxx)
    
    Args:
        phone: Phone number with or without country code
    
    Returns:
        Phone number without country code in local format
    
    Examples:
        "966501234567" → "0501234567"
        "201234567890" → "01234567890"
        "0501234567" → "0501234567" (already local)
    """
    if not phone:
        return phone
    
    # Remove all non-digits
    digits = re.sub(r"\D", "", phone)
    
    if not digits:
        return phone
    
    # Saudi Arabia: 966 country code
    if digits.startswith("966"):
        # Remove 966, add 0 prefix for local format
        local = digits[3:]  # Remove "966"
        return f"{local}"
    
    # Egypt: 20 country code (10 digits after 20)
    if digits.startswith("20") and len(digits) >= 11:
        # Remove 20, add 0 prefix for local format
        local = digits[2:]  # Remove "20"
        return f"{local}"
    
    # Already in local format or unknown country
    return digits
