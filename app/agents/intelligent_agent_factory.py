# -*- coding: utf-8 -*-
"""
Intelligent Agent Factory - Singleton for IntelligentBookingAgent
=================================================================
"""

from loguru import logger
from .intelligent_booking_agent import IntelligentBookingAgent
from ..config import settings
from ..api.agent_api import AgentApiClient


_intelligent_agent_instance = None


def get_intelligent_agent() -> IntelligentBookingAgent:
    """
    Get or create singleton instance of IntelligentBookingAgent.
    
    Returns:
        IntelligentBookingAgent instance
    """
    global _intelligent_agent_instance
    
    if _intelligent_agent_instance is None:
        logger.info("🤖 Initializing IntelligentBookingAgent (first time)")
        
        # Get OpenAI API key
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        api_key = settings.openai_api_key.get_secret_value()
        
        # Create API client
        api_client = AgentApiClient()
        
        # Create intelligent agent
        _intelligent_agent_instance = IntelligentBookingAgent(
            api_client=api_client,
            openai_api_key=api_key
        )
        
        logger.info("✅ IntelligentBookingAgent singleton created")
    
    return _intelligent_agent_instance
