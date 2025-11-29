"""
Agent Response Models
=====================
Data structures for agent responses and function calling.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ResponseType(Enum):
    """Type of agent response"""
    DIRECT = "direct"              # Direct response, no action needed
    FUNCTION_CALL = "function_call"  # Needs to call a function
    WORKFLOW = "workflow"          # Needs to execute workflow
    ERROR = "error"                # Error occurred


@dataclass
class FunctionCall:
    """
    Represents a function call requested by the agent.
    
    Example:
        FunctionCall(
            name="execute_workflow",
            arguments={"workflow": "booking", "params": {"service_id": 7}}
        )
    """
    name: str
    arguments: Dict[str, Any]
    call_id: Optional[str] = None


@dataclass
class AgentResponse:
    """
    Response from Reem (Intelligent Agent).
    
    This is what Reem returns after processing a user message.
    
    Attributes:
        content: Natural language response to user (Arabic)
        response_type: Type of response
        function_call: If action needed, which function to call
        context_updates: Updates to conversation context
        metadata: Additional metadata (confidence, tokens, etc.)
    
    Example:
        AgentResponse(
            content="تمام عزيزي! حجزت لك موعد يوم السبت...",
            response_type=ResponseType.DIRECT,
            function_call=None,
            context_updates={"last_intent": "booking_complete"},
            metadata={"tokens": 150, "confidence": 0.95}
        )
    """
    content: str
    response_type: ResponseType = ResponseType.DIRECT
    function_call: Optional[FunctionCall] = None
    context_updates: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_function_call(self) -> bool:
        """Check if response requires function execution"""
        return self.function_call is not None
    
    def is_workflow(self) -> bool:
        """Check if response requires workflow execution"""
        return (
            self.has_function_call() and 
            self.function_call.name == "execute_workflow"
        )
    
    def is_error(self) -> bool:
        """Check if response is an error"""
        return self.response_type == ResponseType.ERROR


@dataclass
class Message:
    """
    Single message in conversation history.
    
    Compatible with OpenAI chat format.
    """
    role: str  # "user", "assistant", "system", "function"
    content: str
    name: Optional[str] = None  # Function name if role="function"
    function_call: Optional[Dict[str, Any]] = None  # If assistant called function
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for OpenAI API"""
        data = {"role": self.role, "content": self.content}
        if self.name:
            data["name"] = self.name
        if self.function_call:
            data["function_call"] = self.function_call
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create from dict"""
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            name=data.get("name"),
            function_call=data.get("function_call")
        )
