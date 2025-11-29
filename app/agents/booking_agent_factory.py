"""
Booking Agent Factory with Feature Flags

Enables gradual rollout of LangGraph implementation.
Allows A/B testing and safe migration.
"""
import hashlib
from typing import Union
from loguru import logger

from ..config import settings
from .booking_agent import BookingAgent
from .booking_agent_langgraph import BookingAgentLangGraph


class BookingAgentFactory:
    """
    Factory to create booking agent based on feature flag.
    
    Allows gradual rollout of LangGraph version with multiple strategies:
    - Global toggle
    - Session whitelist
    - Percentage-based rollout
    """
    
    # Track which version is being used (for metrics)
    _version_usage = {"original": 0, "langgraph": 0}
    
    @staticmethod
    def create(session_id: str, api_client=None) -> Union[BookingAgent, BookingAgentLangGraph]:
        """
        Create booking agent based on feature flag configuration.
        
        Args:
            session_id: Session identifier
            api_client: Optional API client instance
            
        Returns:
            BookingAgent or BookingAgentLangGraph instance
            
        Feature flag priority:
        1. Global flag (USE_LANGGRAPH)
        2. Session whitelist (LANGGRAPH_SESSIONS)
        3. Percentage rollout (LANGGRAPH_ROLLOUT_PERCENTAGE)
        4. Default to original
        """
        
        # Strategy 1: Global flag (highest priority)
        # Check both use_langgraph (Settings field) and USE_LANGGRAPH (backward compatibility)
        logger.debug(f"🔍 [FACTORY] Checking USE_LANGGRAPH: settings.use_langgraph={settings.use_langgraph}")
        if settings.use_langgraph or getattr(settings, 'USE_LANGGRAPH', False):
            logger.info(f"🆕 [FACTORY] Using LangGraph agent for {session_id} (global flag enabled)")
            BookingAgentFactory._version_usage["langgraph"] += 1
            # BookingAgentLangGraph.__new__ only accepts session_id (like BookingAgent)
            return BookingAgentLangGraph(session_id)
        
        # Strategy 2: Session whitelist
        langgraph_sessions = settings.langgraph_sessions or getattr(settings, 'LANGGRAPH_SESSIONS', [])
        if langgraph_sessions and session_id in langgraph_sessions:
            logger.info(f"🆕 [FACTORY] Using LangGraph agent for {session_id} (session whitelisted)")
            BookingAgentFactory._version_usage["langgraph"] += 1
            return BookingAgentLangGraph(session_id)
        
        # Strategy 3: Percentage-based rollout
        rollout_percentage = getattr(settings, 'LANGGRAPH_ROLLOUT_PERCENTAGE', 0)
        if rollout_percentage > 0:
            # Deterministic hash-based selection (same session always gets same version)
            hash_val = int(hashlib.md5(session_id.encode()).hexdigest(), 16)
            if (hash_val % 100) < rollout_percentage:
                logger.info(f"🆕 [FACTORY] Using LangGraph agent for {session_id} ({rollout_percentage}% rollout)")
                BookingAgentFactory._version_usage["langgraph"] += 1
                return BookingAgentLangGraph(session_id)
        
        # Strategy 4: Default to original agent
        # BookingAgent.__new__() only accepts session_id, so we can't pass api_client
        # The agent creates its own api_client in __init__
        logger.info(f"📦 [FACTORY] Using original agent for {session_id} (default)")
        BookingAgentFactory._version_usage["original"] += 1
        return BookingAgent(session_id)  # __new__ only takes session_id
    
    @classmethod
    def get_usage_stats(cls) -> dict:
        """
        Get usage statistics for monitoring.
        
        Returns:
            dict with original and langgraph usage counts
        """
        total = sum(cls._version_usage.values())
        if total == 0:
            return {
                "original": {"count": 0, "percentage": 0},
                "langgraph": {"count": 0, "percentage": 0},
                "total": 0
            }
        
        return {
            "original": {
                "count": cls._version_usage["original"],
                "percentage": (cls._version_usage["original"] / total) * 100
            },
            "langgraph": {
                "count": cls._version_usage["langgraph"],
                "percentage": (cls._version_usage["langgraph"] / total) * 100
            },
            "total": total
        }
    
    @classmethod
    def reset_stats(cls):
        """Reset usage statistics"""
        cls._version_usage = {"original": 0, "langgraph": 0}
