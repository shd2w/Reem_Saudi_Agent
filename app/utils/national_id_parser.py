"""
Saudi National ID Parser
========================
Extract information from Saudi National ID numbers.

Saudi National ID Format (10 digits):
- Digit 1: Type (1=Saudi citizen, 2=Resident)
- Digits 2-3: Birth year (YY) - Hijri calendar
- Digits 4-5: Birth month (MM) - Hijri calendar  
- Digits 6-7: Birth day (DD) - Hijri calendar
- Digits 8-10: Sequence number

Note: Birth date is in Hijri (Islamic) calendar, needs conversion to Gregorian.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from loguru import logger


def hijri_to_gregorian_approximate(hijri_year: int, hijri_month: int, hijri_day: int) -> Optional[str]:
    """
    Approximate conversion from Hijri to Gregorian date.
    
    Note: This is approximate! Hijri calendar is lunar and complex.
    For production, use a proper library like hijri-converter.
    
    Approximation: Hijri year ≈ Gregorian year - 622 + (Hijri year * 0.97)
    """
    try:
        # Convert 2-digit year to 4-digit
        # Assume 00-50 = 1400-1450 (2000s-2020s)
        # Assume 51-99 = 1351-1399 (1900s-2000s)
        if hijri_year <= 50:
            full_hijri_year = 1400 + hijri_year
        else:
            full_hijri_year = 1300 + hijri_year
        
        # Approximate Gregorian year
        # Hijri year 1443 ≈ Gregorian year 2021-2022
        gregorian_year = int(full_hijri_year - 1443 + 2021)
        
        # Ensure reasonable bounds (1950-2010 for adults)
        if gregorian_year < 1950:
            gregorian_year += 100
        elif gregorian_year > 2010:
            gregorian_year -= 100
        
        # Validate month and day
        if not (1 <= hijri_month <= 12):
            logger.warning(f"Invalid Hijri month: {hijri_month}")
            return None
        
        if not (1 <= hijri_day <= 30):  # Hijri months are 29-30 days
            logger.warning(f"Invalid Hijri day: {hijri_day}")
            return None
        
        # Use approximate Gregorian date
        # (Hijri months don't map directly, so we use rough estimate)
        gregorian_month = hijri_month
        gregorian_day = min(hijri_day, 28)  # Safe day that exists in all months
        
        # Validate and create date
        try:
            date_obj = datetime(gregorian_year, gregorian_month, gregorian_day)
            return date_obj.strftime("%Y-%m-%d")
        except ValueError:
            # If date invalid, use middle of year as fallback
            logger.warning(f"Invalid date: {gregorian_year}-{gregorian_month}-{gregorian_day}, using fallback")
            return f"{gregorian_year}-06-15"
    
    except Exception as e:
        logger.error(f"Error converting Hijri date: {e}")
        return None


def parse_saudi_national_id(national_id: str) -> Optional[Dict]:
    """
    Parse Saudi National ID to extract birth date and other info.
    
    Args:
        national_id: 10-digit Saudi national ID
        
    Returns:
        Dictionary with:
        - type: "citizen" or "resident"
        - birth_date: Gregorian date string (YYYY-MM-DD)
        - hijri_birth: Original Hijri date
        - sequence: Sequence number
    """
    try:
        # Validate format
        if not national_id or not national_id.isdigit():
            logger.warning(f"Invalid national ID format: {national_id}")
            return None
        
        if len(national_id) != 10:
            logger.warning(f"National ID must be 10 digits, got {len(national_id)}: {national_id}")
            return None
        
        # Parse components
        id_type = int(national_id[0])
        birth_year = int(national_id[1:3])
        birth_month = int(national_id[3:5])
        birth_day = int(national_id[5:7])
        sequence = national_id[7:10]
        
        # Determine type
        type_str = "citizen" if id_type == 1 else "resident"
        
        # Convert birth date
        gregorian_date = hijri_to_gregorian_approximate(birth_year, birth_month, birth_day)
        
        if not gregorian_date:
            logger.warning(f"Could not extract birth date from ID: {national_id}")
            return None
        
        result = {
            "type": type_str,
            "birth_date": gregorian_date,
            "hijri_birth": f"{birth_year:02d}/{birth_month:02d}/{birth_day:02d}",
            "sequence": sequence
        }
        
        logger.info(f"✅ Parsed national ID: Type={type_str}, BirthDate={gregorian_date}")
        return result
    
    except Exception as e:
        logger.error(f"❌ Error parsing national ID {national_id}: {e}")
        return None


def get_birth_date_from_national_id(national_id: str, fallback: str = "1990-01-01") -> str:
    """
    Extract birth date from Saudi national ID, with fallback.
    
    Args:
        national_id: Saudi national ID
        fallback: Fallback date if extraction fails
        
    Returns:
        Birth date string (YYYY-MM-DD)
    """
    parsed = parse_saudi_national_id(national_id)
    
    if parsed and parsed.get("birth_date"):
        return parsed["birth_date"]
    else:
        logger.warning(f"⚠️ Could not parse birth date from ID {national_id}, using fallback: {fallback}")
        return fallback
