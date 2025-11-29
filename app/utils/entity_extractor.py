"""
Entity Extraction / NER Module
================================
Extracts names, phone numbers, and other entities from user messages.

Author: Agent Orchestrator Team
Version: 1.0.0
"""

import re
from typing import Dict, Optional
from loguru import logger


class EntityExtractor:
    """
    Extract entities from user messages using regex and patterns.
    """
    
    def __init__(self):
        # Phone patterns for different formats
        self.phone_patterns = [
            r'(\+?968\d{8})',  # Oman format with country code
            r'(05\d{8})',      # Saudi format
            r'(\d{8})',        # 8 digits
            r'(96897\d{6})',   # Oman mobile
        ]
        
        # Phrases indicating "my phone" or "this number"
        self.phone_indicators = [
            'معك', 'معاك', 'نفس الرقم', 'جوالي', 'رقمي',
            'my phone', 'this number', 'same number', 'عالواتس'
        ]
    
    def extract_name_and_phone(
        self, 
        message: str, 
        session_phone: Optional[str] = None,
        context: Optional[str] = None,
        current_step: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Extract name and phone from user message.
        Only extracts name in registration context or with explicit markers.
        
        CRITICAL: Step-aware extraction to prevent false positives (Issue: ID extracted as phone)
        
        Args:
            message: User message
            session_phone: WhatsApp session phone number
            context: Current context (e.g., 'registration', 'booking')
            current_step: Current workflow step (e.g., 'awaiting_name', 'registration_id')
            
        Returns:
            Dict with 'name' and/or 'phone' keys
        """
        extracted = {}
        message_lower = message.lower()
        
        # CRITICAL: Skip phone extraction if we're collecting national ID
        # National IDs (10 digits) look like phone numbers and cause false positives
        skip_phone_extraction = current_step in [
            'registration_id', 'awaiting_id', 'awaiting_national_id', 'awaiting_gender'
        ] if current_step else False
        
        # ADDITIONAL CHECK: If message is exactly 10 digits, likely national ID not phone
        is_likely_national_id = bool(re.match(r'^\d{10}$', message.strip()))
        
        if skip_phone_extraction:
            logger.debug(f"⏭️ Skipping phone extraction (step={current_step}) to avoid ID false positive")
        elif is_likely_national_id and context == 'registration':
            logger.debug(f"⏭️ Skipping phone extraction (10-digit number in registration context = likely national ID)")
            skip_phone_extraction = True
        
        # Greetings and commands that should NOT be extracted as names
        BLACKLIST = [
            'هلا', 'مرحبا', 'السلام', 'صباح', 'مساء', 'اهلا', 'أهلا',
            'احجز', 'أحجز', 'اريد', 'أريد', 'ابي', 'ابغى', 'عندي',
            'hello', 'hi', 'hey', 'good', 'morning', 'evening',
            'book', 'booking', 'appointment', 'want', 'need'
        ]
        
        # Name markers that indicate explicit name mention
        NAME_MARKERS = [
            'اسمي', 'اسمى', 'انا', 'أنا', 'my name', 'i am', "i'm", 'call me'
        ]
        
        # Negative phrases that block name extraction
        NEGATIVE_PHRASES = ['ما سجلت', 'مو سجلت', 'ما عندي', 'ليس', 'مو']
        
        # Extract phone number (ONLY if not in ID collection step)
        if not skip_phone_extraction:
            for pattern in self.phone_patterns:
                match = re.search(pattern, message)
                if match:
                    candidate_phone = match.group(1)
                    
                    # CRITICAL: Validate phone format (Issue: False positives from IDs)
                    # Saudi phone must start with 05 or 5, or Oman format
                    if self._is_valid_phone_format(candidate_phone):
                        extracted['phone'] = candidate_phone
                        logger.info(f"✅ Extracted phone: {extracted['phone']}")
                        break
                    else:
                        logger.debug(f"⚠️ Rejected invalid phone format: {candidate_phone}")
            
            # Check if user says "my phone" or "same number"
            if not extracted.get('phone') and session_phone:
                if any(indicator in message_lower for indicator in self.phone_indicators):
                    extracted['phone'] = session_phone
                    extracted['phone_source'] = 'session'
                    logger.info(f"✅ Using session phone: {session_phone}")
        
        # Extract name ONLY if in registration context OR explicit name marker present
        should_extract_name = False
        
        # BLOCK name extraction if negative phrase detected
        if any(neg in message_lower for neg in NEGATIVE_PHRASES):
            logger.debug(f"⚠️ Blocked name extraction: negative phrase detected")
            should_extract_name = False
        # Check if in registration context
        elif context in ['registration', 'patient_registration']:
            should_extract_name = True
        # Check if message contains explicit name marker
        elif any(marker in message_lower for marker in NAME_MARKERS):
            should_extract_name = True
        
        if should_extract_name:
            # Extract name (Arabic or English)
            cleaned = message
            
            # Remove phone if found
            if 'phone' in extracted:
                cleaned = cleaned.replace(extracted['phone'], '')
            
            # Remove common stop words
            stop_words = [
                'اسمي', 'اسمى', 'انا', 'أنا', 'my name', 'i am', "i'm",
                'جوالي', 'رقمي', 'و', 'معك', 'معاك', 'عالواتس',
                'is', 'the', 'and', 'call', 'me'
            ]
            
            for word in stop_words:
                cleaned = re.sub(rf'\b{word}\b', '', cleaned, flags=re.IGNORECASE)
            
            # Extract remaining text as name (Arabic or English, 2-30 chars)
            name_match = re.search(r'([أ-يa-zA-Z\s]{2,30})', cleaned)
            if name_match:
                name = name_match.group(1).strip()
                
                # Validate name: not in blacklist, min length, valid chars
                if (len(name) >= 2 and 
                    name.lower() not in BLACKLIST and
                    not any(blacklisted in name.lower() for blacklisted in BLACKLIST)):
                    
                    # Additional validation: name should not be a command
                    if not any(cmd in name.lower() for cmd in ['احجز', 'اريد', 'ابي', 'book', 'want']):
                        extracted['name'] = name
                        extracted['name_confidence'] = 'high' if any(marker in message_lower for marker in NAME_MARKERS) else 'medium'
                        logger.info(f"✅ Extracted name: {extracted['name']} (confidence: {extracted['name_confidence']})")
                    else:
                        logger.debug(f"⚠️ Rejected name candidate (command-like): '{name}'")
                else:
                    logger.debug(f"⚠️ Rejected name candidate (blacklisted/short): '{name}'")
        
        # Sanitize extracted name to prevent prompt injection
        if extracted.get('name'):
            extracted['name'] = self._sanitize_name(extracted['name'])
            
            # Detect message language
            message_language = self._detect_language(message)
            
            # Validate name language matches message language
            if not self.validate_name_language(extracted['name'], message_language):
                # Reject the name
                logger.info(f"❌ Name rejected: Latin alphabet not allowed in Arabic context")
                extracted.pop('name')
                extracted.pop('name_confidence', None)
        
        # Log extraction summary
        if extracted:
            logger.info(f"📊 Entity extraction: {list(extracted.keys())} from '{message[:50]}'")
        else:
            logger.debug(f"⚠️ No entities extracted from: '{message[:50]}'")
        
        return extracted
    
    def _detect_language(self, text: str) -> str:
        """
        Detect if text is Arabic or English.
        
        Args:
            text: Text to analyze
            
        Returns:
            'ar' for Arabic, 'en' for English
        """
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_chars = sum(1 for c in text if c.isalpha())
        
        if total_chars == 0:
            return 'ar'  # Default to Arabic
        
        arabic_ratio = arabic_chars / total_chars
        return 'ar' if arabic_ratio > 0.3 else 'en'
    
    def _sanitize_name(self, name: str) -> str:
        """
        Sanitize name to prevent prompt injection and normalize encoding.
        Validates that name is in Arabic if provided in Arabic context.
        
        Args:
            name: Raw name string
            
        Returns:
            Sanitized name
        """
        import unicodedata
        
        # Normalize to Unicode NFC (canonical composition)
        name = unicodedata.normalize('NFC', name)
        
        # Remove control characters and special chars that could be injection vectors
        dangerous_chars = ['\n', '\r', '\t', '{', '}', '[', ']', '<', '>', '|', '\\', '`', '$', ';']
        for char in dangerous_chars:
            name = name.replace(char, '')
        
        # Remove multiple spaces
        name = ' '.join(name.split())
        
        # Trim to reasonable length (prevent overflow)
        if len(name) > 50:
            name = name[:50]
            logger.debug(f"⚠️ Name truncated to 50 chars")
        
        return name.strip()
    
    def _is_valid_phone_format(self, phone: str) -> bool:
        """
        Validate phone number format to prevent false positives.
        
        Args:
            phone: Candidate phone number string
            
        Returns:
            True if valid phone format, False otherwise
        """
        import re
        
        # Remove any non-digits for validation
        digits_only = re.sub(r'\D', '', phone)
        
        # Saudi mobile: Must start with 5 (after country code 966)
        # Or start with 05 (local format)
        # Format: 05XXXXXXXX (10 digits) or 5XXXXXXXX (9 digits) or 9665XXXXXXXX (12 digits)
        if re.match(r'^(966)?5\d{8}$', digits_only):
            return True
        
        # Also accept 05 format (Saudi local)
        if re.match(r'^05\d{8}$', phone):
            return True
        
        # Oman format: +968XXXXXXXX
        if re.match(r'^(968)?\d{8}$', digits_only):
            return True
        
        # If it's just 8-9 random digits without proper prefix, reject
        # This prevents matching parts of national IDs
        logger.debug(f"Invalid phone format (no valid prefix): {phone}")
        return False
    
    def validate_name_language(self, name: str, message_language: str) -> bool:
        """
        Validate that name language matches message language.
        If message is in Arabic, reject Latin alphabet names.
        
        Args:
            name: Extracted name
            message_language: Language of the message ('ar' or 'en')
            
        Returns:
            True if valid, False if rejected
        """
        import unicodedata
        
        # Count Arabic vs Latin characters
        arabic_chars = 0
        latin_chars = 0
        
        for char in name:
            if '\u0600' <= char <= '\u06FF':  # Arabic Unicode range
                arabic_chars += 1
            elif char.isalpha() and 'a' <= char.lower() <= 'z':  # Latin alphabet
                latin_chars += 1
        
        # If message is in Arabic but name is Latin, reject
        if message_language == 'ar' and latin_chars > arabic_chars:
            logger.warning(f"⚠️ Rejected Latin name '{name}' in Arabic context")
            return False
        
        return True
    
    def extract_date_time(self, message: str) -> Dict[str, str]:
        """
        Extract date and time references from message.
        
        Args:
            message: User message
            
        Returns:
            Dict with 'date' and/or 'time' keys
        """
        extracted = {}
        message_lower = message.lower()
        
        # Date keywords
        if any(word in message_lower for word in ['بكرة', 'بكره', 'غدا', 'غداً', 'tomorrow']):
            extracted['date'] = 'tomorrow'
        elif any(word in message_lower for word in ['اليوم', 'today']):
            extracted['date'] = 'today'
        elif any(word in message_lower for word in ['بعد بكرة', 'بعد غد']):
            extracted['date'] = 'day_after_tomorrow'
        
        # Time keywords
        if any(word in message_lower for word in ['صباح', 'الصباح', 'صبح', 'morning']):
            extracted['time'] = 'morning'
        elif any(word in message_lower for word in ['ظهر', 'الظهر', 'noon']):
            extracted['time'] = 'noon'
        elif any(word in message_lower for word in ['عصر', 'العصر', 'afternoon']):
            extracted['time'] = 'afternoon'
        elif any(word in message_lower for word in ['مساء', 'المساء', 'evening']):
            extracted['time'] = 'evening'
        
        # Extract specific hour with VALIDATION (Issue: 62 o'clock extracted from ID)
        time_match = re.search(r'(\d{1,2})\s*(am|pm|ص|م)?', message_lower)
        if time_match:
            hour = int(time_match.group(1))
            period = time_match.group(2)
            
            # CRITICAL: Validate hour is in valid range (0-23)
            # Prevents national IDs like "628404738" from being extracted as "hour: 62"
            if 0 <= hour <= 23:
                extracted['hour'] = hour
                if period:
                    extracted['period'] = period
                logger.debug(f"✅ Extracted valid hour: {hour}")
            else:
                logger.debug(f"⚠️ Rejected invalid hour: {hour} (must be 0-23)")
        
        return extracted
    
    def extract_service_keywords(self, message: str) -> list:
        """
        Extract service-related keywords.
        
        Args:
            message: User message
            
        Returns:
            List of service keywords found
        """
        keywords = []
        message_lower = message.lower()
        
        service_terms = {
            'ليزر': 'laser',
            'استشارة': 'consultation',
            'فحص': 'checkup',
            'تنظيف': 'cleaning',
            'حشو': 'filling',
            'تقويم': 'braces',
            'زراعة': 'implant'
        }
        
        for arabic, english in service_terms.items():
            if arabic in message_lower or english in message_lower:
                keywords.append(arabic)
        
        return keywords


# Singleton instance
_entity_extractor = None

def get_entity_extractor() -> EntityExtractor:
    """Get singleton entity extractor instance."""
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
    return _entity_extractor
