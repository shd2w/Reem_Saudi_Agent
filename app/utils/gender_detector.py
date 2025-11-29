"""
LLM-Based Gender Detection from Arabic Names
=============================================

Uses OpenAI LLM to intelligently detect gender from Arabic names.
Falls back to 'male' if detection fails.
"""

from loguru import logger
from typing import Optional
from openai import AsyncOpenAI
from ..config import get_settings


async def detect_gender_from_name(name: str) -> str:
    """
    Detect gender from Arabic or English name using LLM.
    
    Args:
        name: Person's name (Arabic or English)
        
    Returns:
        "male" or "female" (defaults to "male" if uncertain)
    """
    if not name or not name.strip():
        logger.warning("⚠️ Empty name provided for gender detection, defaulting to male")
        return "male"
    
    try:
        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        
        # LLM prompt for gender detection
        prompt = f"""Detect the gender of this Arabic or English name: "{name}"

Rules:
- Respond with ONLY one word: "male" or "female"
- If uncertain, respond with "male" (default)
- Consider Arabic name patterns and endings
- Common female endings: ة، اء، ى
- Common male patterns: عبد، محمد، أحمد، etc.

Name: {name}
Gender:"""

        logger.debug(f"🤖 Detecting gender for name: '{name}'")
        
        # Call LLM with timeout
        import asyncio
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=settings.openai_model or "gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a gender detection expert for Arabic and English names. Respond with only 'male' or 'female'."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=10,  # Only need one word
                timeout=10.0  # 10 second timeout
            ),
            timeout=15.0
        )
        
        # Extract gender from response
        detected = response.choices[0].message.content.strip().lower()
        
        # Validate response
        if "female" in detected or "أنثى" in detected:
            logger.info(f"✅ LLM detected gender: female for name '{name}'")
            return "female"
        elif "male" in detected or "ذكر" in detected:
            logger.info(f"✅ LLM detected gender: male for name '{name}'")
            return "male"
        else:
            logger.warning(f"⚠️ LLM returned unexpected response: '{detected}', defaulting to male")
            return "male"
            
    except asyncio.TimeoutError:
        logger.error(f"⚠️ LLM timeout detecting gender for '{name}', defaulting to male")
        return "male"
    except Exception as e:
        logger.error(f"❌ Error detecting gender for '{name}': {e}, defaulting to male")
        return "male"


def detect_gender_from_name_sync(name: str) -> str:
    """
    Synchronous wrapper for gender detection.
    Uses asyncio.run() to call async function.
    
    Args:
        name: Person's name
        
    Returns:
        "male" or "female"
    """
    import asyncio
    try:
        return asyncio.run(detect_gender_from_name(name))
    except Exception as e:
        logger.error(f"❌ Sync gender detection failed for '{name}': {e}, defaulting to male")
        return "male"
