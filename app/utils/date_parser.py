"""
Arabic Date Parser - Extract dates from Arabic phrases
"""
from datetime import datetime, timedelta
from typing import Optional
import re

def parse_arabic_date(text: str, current_time: Optional[datetime] = None) -> Optional[str]:
    """
    Parse Arabic date phrases and convert to YYYY-MM-DD format.
    
    Args:
        text: User message containing date phrase
        current_time: Current datetime (defaults to now)
    
    Returns:
        Date in YYYY-MM-DD format or None if no date found
    """
    if current_time is None:
        current_time = datetime.now()
    
    text_lower = text.lower().strip()
    
    # Today - اليوم
    if any(word in text_lower for word in ["اليوم", "الیوم"]):
        return current_time.strftime("%Y-%m-%d")
    
    # Tomorrow - بكرة، غدا، بكره
    if any(word in text_lower for word in ["بكرة", "بكره", "غدا", "غداً", "بكرا"]):
        tomorrow = current_time + timedelta(days=1)
        return tomorrow.strftime("%Y-%m-%d")
    
    # Day after tomorrow - بعد بكرة، بعد غدا
    if any(phrase in text_lower for phrase in ["بعد بكرة", "بعد بكره", "بعد غدا", "بعد غداً"]):
        day_after = current_time + timedelta(days=2)
        return day_after.strftime("%Y-%m-%d")
    
    # Specific weekdays in Arabic
    weekdays_ar = {
        "السبت": 5,  # Saturday
        "الأحد": 6,  # Sunday
        "الاثنين": 0,  # Monday
        "الثلاثاء": 1,  # Tuesday
        "الأربعاء": 2,  # Wednesday
        "الخميس": 3,  # Thursday
        "الجمعة": 4,  # Friday
    }
    
    for day_name, day_num in weekdays_ar.items():
        if day_name in text_lower:
            # Find next occurrence of this weekday
            current_weekday = current_time.weekday()
            days_ahead = (day_num - current_weekday) % 7
            if days_ahead == 0:
                days_ahead = 7  # Next week if same day
            target_date = current_time + timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d")
    
    # Extract dates in format DD-MM-YYYY, DD/MM/YYYY, or YYYY-MM-DD
    date_patterns = [
        r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
        r'(\d{1,2})-(\d{1,2})-(\d{4})',  # DD-MM-YYYY
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # DD/MM/YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:  # YYYY-MM-DD
                year, month, day = groups
            else:  # DD-MM-YYYY or DD/MM/YYYY
                day, month, year = groups
            
            try:
                parsed_date = datetime(int(year), int(month), int(day))
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                continue
    
    return None


def extract_date_from_context(message: str, conversation_history: list = None, current_time: Optional[datetime] = None) -> Optional[str]:
    """
    Extract date from current message or recent conversation history.
    
    Args:
        message: Current user message
        conversation_history: List of recent messages
        current_time: Current datetime
    
    Returns:
        Date in YYYY-MM-DD format or None
    """
    # First try current message
    date = parse_arabic_date(message, current_time)
    if date:
        return date
    
    # Check last 3 messages in history
    if conversation_history:
        for msg in reversed(conversation_history[-3:]):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content", "")
                date = parse_arabic_date(content, current_time)
                if date:
                    return date
    
    return None
