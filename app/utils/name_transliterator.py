"""
Name Transliteration for Arabic Context
=========================================
Handles English/Latin names in Arabic conversation to maintain professionalism.

Features:
- Detects Latin/English names
- Transliterates common names to Arabic
- Falls back to generic greetings
- Maintains natural Saudi dialect

Author: Agent Orchestrator Team
Version: 1.0.0
"""

import re
from typing import Optional


# Common English to Arabic name mappings (Saudi pronunciation)
COMMON_NAME_TRANSLATIONS = {
    # Male names
    "ahmed": "أحمد",
    "ahmad": "أحمد",
    "mohamed": "محمد",
    "mohammed": "محمد",
    "muhammad": "محمد",
    "ali": "علي",
    "omar": "عمر",
    "umar": "عمر",
    "khalid": "خالد",
    "khaled": "خالد",
    "abdullah": "عبدالله",
    "abdallah": "عبدالله",
    "abdulrahman": "عبدالرحمن",
    "abdelrahman": "عبدالرحمن",
    "faisal": "فيصل",
    "fahad": "فهد",
    "fahd": "فهد",
    "salman": "سلمان",
    "saud": "سعود",
    "sultan": "سلطان",
    "turki": "تركي",
    "nawaf": "نواف",
    "bandar": "بندر",
    "abdulaziz": "عبدالعزيز",
    "abdelaziz": "عبدالعزيز",
    "waleed": "وليد",
    "walid": "وليد",
    "majed": "ماجد",
    "majid": "ماجد",
    "rayan": "ريان",
    "rayyan": "ريان",
    "yazeed": "يزيد",
    "yazid": "يزيد",
    "talal": "طلال",
    "nasser": "ناصر",
    "nasir": "ناصر",
    "mansour": "منصور",
    "mansur": "منصور",
    "hamza": "حمزة",
    "yousef": "يوسف",
    "youssef": "يوسف",
    "yusuf": "يوسف",
    "hassan": "حسن",
    "hussein": "حسين",
    "husain": "حسين",
    "osama": "أسامة",
    "usama": "أسامة",
    "tariq": "طارق",
    "tarik": "طارق",
    "adel": "عادل",
    "adil": "عادل",
    "saeed": "سعيد",
    "said": "سعيد",
    "rashid": "راشد",
    "rashed": "راشد",
    
    # Female names
    "fatima": "فاطمة",
    "fatma": "فاطمة",
    "aisha": "عائشة",
    "aysha": "عائشة",
    "sarah": "سارة",
    "sara": "سارة",
    "maryam": "مريم",
    "mariam": "مريم",
    "layla": "ليلى",
    "laila": "ليلى",
    "nora": "نورة",
    "noura": "نورة",
    "huda": "هدى",
    "reem": "ريم",
    "rim": "ريم",
    "amal": "أمل",
    "hanan": "حنان",
    "lama": "لمى",
    "lama": "لمى",
    "salma": "سلمى",
    "dina": "دينا",
    "dana": "دانا",
    "rana": "رنا",
    "rania": "رانيا",
    "lina": "لينا",
    "maha": "مها",
    "nada": "ندى",
    "hala": "هالة",
    "hala": "هالة",
    "amira": "أميرة",
    "amina": "أمينة",
    "khadija": "خديجة",
    "khadijah": "خديجة",
    
    # International names (phonetic)
    "john": "جون",
    "david": "ديفيد",
    "michael": "مايكل",
    "james": "جيمس",
    "robert": "روبرت",
    "william": "ويليام",
    "mary": "ماري",
    "lisa": "ليزا",
    "jennifer": "جينيفر",
    "linda": "ليندا",
    "shady": "شادي",
    "tony": "توني",
    "sam": "سام",
    "max": "ماكس",
    
    # Doctor names (common)
    "batoul": "بتول",
    "battoul": "بتول",
    "heba": "هبة",
    "fehan": "فيحان",
    "omar": "عمر",
}


def is_latin_name(name: str) -> bool:
    """
    Check if a name contains only Latin/English characters.
    
    Args:
        name: Name to check
        
    Returns:
        True if name is in Latin script, False if it contains Arabic
    """
    if not name or name in ["Unknown", "None", ""]:
        return False
    
    # Check if contains Arabic characters
    has_arabic = bool(re.search(r'[\u0600-\u06FF]', name))
    
    # Check if contains Latin characters
    has_latin = bool(re.search(r'[A-Za-z]', name))
    
    return has_latin and not has_arabic


def transliterate_name(name: str) -> Optional[str]:
    """
    Transliterate English name to Arabic.
    
    Args:
        name: English name to transliterate
        
    Returns:
        Arabic transliteration if found, None otherwise
    """
    if not name or not is_latin_name(name):
        return None
    
    # Normalize and lookup
    name_lower = name.lower().strip()
    
    # Direct match
    if name_lower in COMMON_NAME_TRANSLATIONS:
        return COMMON_NAME_TRANSLATIONS[name_lower]
    
    # Try without common suffixes
    for suffix in [" jr", " sr", " ii", " iii"]:
        if name_lower.endswith(suffix):
            base_name = name_lower[:-len(suffix)].strip()
            if base_name in COMMON_NAME_TRANSLATIONS:
                return COMMON_NAME_TRANSLATIONS[base_name]
    
    return None


def transliterate_full_name(name: str) -> str:
    """
    Transliterate a full name (first + last) word by word.
    
    Args:
        name: Full name in English (e.g., "Batoul Fehan")
        
    Returns:
        Arabic transliteration or original if not found
    """
    if not name or not is_latin_name(name):
        return name
    
    words = name.split()
    arabic_words = []
    
    for word in words:
        word_lower = word.lower()
        if word_lower in COMMON_NAME_TRANSLATIONS:
            arabic_words.append(COMMON_NAME_TRANSLATIONS[word_lower])
        else:
            # Keep original if no translation
            arabic_words.append(word)
    
    return " ".join(arabic_words)


def get_arabic_name_or_fallback(name: str, use_generic: bool = True) -> str:
    """
    Get Arabic name or fallback to generic greeting.
    
    Args:
        name: Original name (might be English/Latin)
        use_generic: If True, return generic term when no translation found
        
    Returns:
        Arabic name or generic greeting
    """
    if not name or name in ["Unknown", "None", ""]:
        return ""  # Let LLM handle naturally
    
    # If already Arabic, return as-is
    if not is_latin_name(name):
        return name
    
    # Try to transliterate
    arabic_name = transliterate_name(name)
    if arabic_name:
        return arabic_name
    
    # No transliteration found - return empty, let LLM handle
    return ""


def format_greeting_with_name(name: str, greeting_style: str = "formal") -> str:
    """
    Format a complete greeting with proper name handling.
    
    Args:
        name: User's name (English or Arabic)
        greeting_style: "formal", "casual", or "simple"
        
    Returns:
        Complete greeting with name in Arabic
    """
    arabic_name = get_arabic_name_or_fallback(name, use_generic=True)
    
    if greeting_style == "formal":
        return f"حياك الله يا {arabic_name}"
    elif greeting_style == "casual":
        return f"هلا يا {arabic_name}"
    else:  # simple
        return f"يا {arabic_name}"


# Singleton instance
_instance = None

def get_name_transliterator():
    """Get singleton instance (for consistency)"""
    global _instance
    if _instance is None:
        _instance = {
            "is_latin": is_latin_name,
            "transliterate": transliterate_name,
            "get_arabic_or_fallback": get_arabic_name_or_fallback,
            "format_greeting": format_greeting_with_name
        }
    return _instance
