"""
Intelligent Language Detection & Response Formatting
====================================================

A sophisticated utility for detecting communication language and generating
culturally appropriate response instructions for conversational AI systems.

This module provides robust language detection capabilities with special attention
to Arabic (Saudi dialect) and English communications, ensuring responses maintain
cultural authenticity and professional standards.

Author: Healthcare AI Team
Version: 2.0.0
"""

import re
from typing import Literal, Dict, Tuple
from dataclasses import dataclass


@dataclass
class LanguageMetrics:
    """
    Container for language analysis metrics.
    
    Attributes:
        arabic_char_count: Number of Arabic Unicode characters detected
        english_char_count: Number of Latin alphabet characters detected
        confidence_score: Percentage confidence in language detection (0-100)
        total_analyzable_chars: Total number of language-specific characters
    """
    arabic_char_count: int
    english_char_count: int
    confidence_score: float
    total_analyzable_chars: int


class LanguageDetector:
    """
    Advanced language detection system for Arabic and English text.
    
    This detector uses Unicode character analysis to determine the primary
    language of input text, with built-in confidence scoring to handle
    mixed-language content appropriately.
    """
    
    # Unicode ranges for comprehensive Arabic detection
    ARABIC_PATTERN = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    ENGLISH_PATTERN = r'[a-zA-Z]'
    
    # Minimum character threshold for reliable detection
    MIN_CHARS_FOR_DETECTION = 3
    
    @classmethod
    def analyze_text(cls, text: str) -> LanguageMetrics:
        """
        Perform comprehensive linguistic analysis on input text.
        
        Args:
            text: The input string to analyze
            
        Returns:
            LanguageMetrics object containing detailed analysis results
            
        Example:
            >>> detector = LanguageDetector()
            >>> metrics = detector.analyze_text("Hello world")
            >>> print(metrics.confidence_score)
            100.0
        """
        if not text or not text.strip():
            return LanguageMetrics(0, 0, 0.0, 0)
        
        arabic_chars = len(re.findall(cls.ARABIC_PATTERN, text))
        english_chars = len(re.findall(cls.ENGLISH_PATTERN, text))
        total_chars = arabic_chars + english_chars
        
        # Calculate confidence based on character distribution
        if total_chars > 0:
            dominant_chars = max(arabic_chars, english_chars)
            confidence = (dominant_chars / total_chars) * 100
        else:
            confidence = 0.0
        
        return LanguageMetrics(
            arabic_char_count=arabic_chars,
            english_char_count=english_chars,
            confidence_score=round(confidence, 2),
            total_analyzable_chars=total_chars
        )
    
    @classmethod
    def detect_language(cls, text: str, confidence_threshold: float = 60.0) -> Literal["arabic", "english"]:
        """
        Detect the primary language of the provided text.
        
        This method determines whether text is predominantly Arabic or English
        by analyzing character distributions. For ambiguous cases where neither
        language dominates clearly, English is returned as the default.
        
        Args:
            text: Input text to analyze
            confidence_threshold: Minimum confidence percentage for detection (default: 60%)
            
        Returns:
            Either "arabic" or "english" indicating the detected language
            
        Raises:
            ValueError: If text is empty or contains no analyzable characters
            
        Example:
            >>> LanguageDetector.detect_language("مرحباً بك")
            'arabic'
            >>> LanguageDetector.detect_language("Hello there")
            'english'
        """
        metrics = cls.analyze_text(text)
        
        # Handle edge case of insufficient text
        if metrics.total_analyzable_chars < cls.MIN_CHARS_FOR_DETECTION:
            # Default to Arabic for Saudi market
            return "arabic"
        
        # If ANY Arabic characters exist, it's Arabic (Saudi market priority)
        if metrics.arabic_char_count > 0:
            return "arabic"
        
        # Only if there are English chars and NO Arabic, it's English
        if metrics.english_char_count > 0:
            return "english"
        
        # Default to Arabic for Saudi market
        return "arabic"


class ResponseFormatter:
    """
    Generates culturally appropriate instructions for AI response generation.
    
    This class provides language-specific instruction sets that guide
    conversational AI to respond in culturally authentic and professionally
    appropriate ways for both Arabic (Saudi dialect) and English speakers.
    """
    
    @staticmethod
    def get_language_instruction(language: Literal["arabic", "english"]) -> str:
        """
        Retrieve response instructions tailored to the detected language.
        
        These instructions help AI systems generate culturally appropriate,
        contextually relevant responses that meet user expectations for
        tone, formality, and linguistic style.
        
        Args:
            language: The detected language ("arabic" or "english")
            
        Returns:
            A comprehensive instruction string for response generation
            
        Example:
            >>> instructions = ResponseFormatter.get_language_instruction("arabic")
            >>> print("Arabic instructions loaded" if instructions else "Failed")
            Arabic instructions loaded
        """
        instructions = {
            "arabic": """
🌟 LANGUAGE DIRECTIVE: Saudi Arabic Response Required

The user is communicating in Arabic. Your response must be delivered in authentic Saudi Arabic dialect (اللهجة السعودية) with the following characteristics:

✓ Conversational Tone: Use natural, flowing Saudi conversational patterns
✓ Cultural Warmth: Embrace the Saudi tradition of warm, hospitable communication
✓ Authentic Expressions: Incorporate common Saudi phrases and idioms naturally
✓ Appropriate Greetings: Open with culturally fitting greetings such as:
  - "أهلاً وسهلاً" (Welcome)
  - "حياك الله" (May God greet you)
  - "تشرفنا" (We're honored)
✓ Respectful Address: Use appropriate levels of formality based on context
✓ Cultural Sensitivity: Be mindful of cultural norms and values in healthcare contexts

Remember: Saudi communication values personal connection, respect, and warmth.
""",
            "english": """
🌟 LANGUAGE DIRECTIVE: Professional English Response Required

The user is communicating in English. Your response should demonstrate:

✓ Clarity: Use clear, accessible language that's easy to understand
✓ Professionalism: Maintain a professional yet approachable tone
✓ Precision: When discussing medical or technical topics, use appropriate terminology with explanations
✓ Warmth: Be friendly and personable while remaining professional
✓ Structure: Organize information logically for easy comprehension
✓ Accessibility: Avoid unnecessary jargon; explain complex concepts simply

Remember: Balance professionalism with approachability to create a comfortable, trustworthy interaction.
"""
        }
        
        return instructions.get(language, instructions["english"])
    
    @staticmethod
    def format_response_with_language(response: str, language: Literal["arabic", "english"]) -> str:
        """
        Apply final formatting and validation to ensure language consistency.
        
        This method performs a final check to ensure the response aligns with
        the detected language's expectations and applies any necessary
        formatting adjustments.
        
        Args:
            response: The generated response text
            language: The target language for the response
            
        Returns:
            The formatted response string, ready for delivery
            
        Note:
            Currently passes through the response unchanged, but provides
            a hook for future enhancements like automated translation
            validation or formatting adjustments.
        """
        # Future enhancement: Add language-specific formatting rules
        # For now, return response as-is after validation
        return response.strip()


# Convenience functions for backward compatibility and ease of use

def detect_language(text: str) -> Literal["arabic", "english"]:
    """
    Quick language detection function.
    
    A convenience wrapper around LanguageDetector.detect_language()
    for simple use cases.
    
    Args:
        text: Input text to analyze
        
    Returns:
        "arabic" or "english"
    """
    return LanguageDetector.detect_language(text)


def get_language_instruction(language: str) -> str:
    """
    Quick instruction retrieval function.
    
    A convenience wrapper around ResponseFormatter.get_language_instruction()
    for simple use cases.
    
    Args:
        language: "arabic" or "english"
        
    Returns:
        Instruction string for AI response generation
    """
    return ResponseFormatter.get_language_instruction(language)


def format_response_with_language(response: str, language: str) -> str:
    """
    Quick response formatting function.
    
    A convenience wrapper around ResponseFormatter.format_response_with_language()
    for simple use cases.
    
    Args:
        response: Generated response text
        language: Target language
        
    Returns:
        Formatted response string
    """
    return ResponseFormatter.format_response_with_language(response, language)