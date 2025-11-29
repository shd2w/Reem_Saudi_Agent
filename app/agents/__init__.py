"""
Agent module exports
"""

# Lazy imports to avoid loading heavy dependencies during verification
# Import directly from module when needed:
#   from app.agents.reem_agent import ReemAgent
#   from app.agents.intelligent_booking_agent import IntelligentBookingAgent

__all__ = [
    "ReemAgent",
    "IntelligentBookingAgent",
]


def __getattr__(name):
    """Lazy import agents to avoid loading OpenAI dependencies prematurely"""
    if name == "IntelligentBookingAgent":
        from .intelligent_booking_agent import IntelligentBookingAgent
        return IntelligentBookingAgent
    elif name == "ReemAgent":
        from .reem_agent import ReemAgent
        return ReemAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
