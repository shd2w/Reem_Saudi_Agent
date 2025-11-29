"""
Conversation Context Models
============================
Data structures for managing conversation state.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime

from .agent_response import Message
from .workflow_result import PatientInfo, ServiceInfo


@dataclass
class BookingState:
    """
    Represents current booking state - separate from conversation!
    
    This tracks the booking workflow progress independently,
    allowing it to be paused, resumed, or cancelled at any time.
    
    Attributes:
        status: Current booking status
        progress: Which steps are completed
        paused_at_step: Which step we paused at (if paused)
        collected_data: Data collected so far
        started_at: When booking started
        paused_at: When booking was paused
    """
    status: str = "idle"  # idle, active, paused, cancelled, completed
    progress: Dict[str, bool] = field(default_factory=dict)
    paused_at_step: Optional[str] = None
    collected_data: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[float] = None
    paused_at: Optional[float] = None
    
    def is_active(self) -> bool:
        """Check if booking is currently active"""
        return self.status == "active"
    
    def is_paused(self) -> bool:
        """Check if booking is paused"""
        return self.status == "paused"
    
    def can_resume(self) -> bool:
        """Check if booking can be resumed"""
        return self.status == "paused" and self.paused_at_step is not None
    
    def get_next_step(self) -> Optional[str]:
        """Get the next step that needs to be completed"""
        steps = ["collect_service", "check_availability", "select_slot", "confirm_booking"]
        for step in steps:
            if not self.progress.get(step, False):
                return step
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "status": self.status,
            "progress": self.progress,
            "paused_at_step": self.paused_at_step,
            "collected_data": self.collected_data,
            "started_at": self.started_at,
            "paused_at": self.paused_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BookingState":
        """Create from dictionary"""
        return cls(
            status=data.get("status", "idle"),
            progress=data.get("progress", {}),
            paused_at_step=data.get("paused_at_step"),
            collected_data=data.get("collected_data", {}),
            started_at=data.get("started_at"),
            paused_at=data.get("paused_at")
        )


@dataclass
class ConversationContext:
    """
    Full conversation context for a session.
    
    This is the complete state of the conversation that gets
    passed to Reem on each turn.
    
    NOW INCLUDES: Enhanced booking state management for dynamic control
    
    Attributes:
        session_id: Unique session identifier (e.g., "whatsapp:966123456789")
        phone_number: User's phone number
        patient: Patient information (if registered)
        conversation_history: Full message history
        selected_service: Currently selected service (if any)
        booking_state: Current booking workflow state (ENHANCED!)
        turn: Current turn number
        language: User's language (detected)
        metadata: Additional context (timestamps, flags, etc.)
    """
    session_id: str
    phone_number: str
    patient: Optional[PatientInfo] = None
    conversation_history: List[Message] = field(default_factory=list)
    selected_service: Optional[ServiceInfo] = None
    booking_state: BookingState = field(default_factory=BookingState)
    turn: int = 0
    language: str = "arabic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # NEW: Context awareness fields for tracking conversation topics
    last_discussed_service: Optional[str] = None  # e.g., "بوتوكس", "ميزوثيرابي"
    last_discussed_service_id: Optional[int] = None  # Service ID from API
    conversation_topics: List[str] = field(default_factory=list)  # Track all topics discussed
    
    def is_registered(self) -> bool:
        """Check if patient is registered"""
        return self.patient is not None and self.patient.already_registered
    
    def has_service_selected(self) -> bool:
        """Check if service is selected"""
        return self.selected_service is not None
    
    def is_in_booking_flow(self) -> bool:
        """Check if currently in booking workflow"""
        return self.booking_state.is_active() or self.booking_state.is_paused()
    
    def get_patient_name(self) -> str:
        """Get patient FIRST name for personalization (more natural/friendly)"""
        if self.patient and self.patient.name:
            # Extract first name only (e.g., "محمد علي" → "محمد")
            full_name = self.patient.name.strip()
            first_name = full_name.split()[0] if full_name else full_name
            return first_name
        return "عزيزي" if self.get_patient_gender() == "male" else "عزيزتي"
    
    def get_patient_gender(self) -> str:
        """Get patient gender or default to male"""
        if self.patient and self.patient.gender:
            return self.patient.gender
        return "male"
    
    def add_message(self, message: Message):
        """Add message to history"""
        self.conversation_history.append(message)
    
    def get_last_n_messages(self, n: int = 10) -> List[Message]:
        """Get last N messages from history"""
        return self.conversation_history[-n:] if len(self.conversation_history) > n else self.conversation_history
    
    def update_from_dict(self, updates: Dict[str, Any]):
        """Update context from dictionary"""
        if "patient" in updates and isinstance(updates["patient"], dict):
            self.patient = PatientInfo.from_dict(updates["patient"])
        
        if "selected_service" in updates and isinstance(updates["selected_service"], dict):
            self.selected_service = ServiceInfo(**updates["selected_service"])
        
        if "booking_state" in updates:
            self.booking_state = updates["booking_state"]
        
        if "turn" in updates:
            self.turn = updates["turn"]
        
        if "language" in updates:
            self.language = updates["language"]
        
        if "metadata" in updates:
            self.metadata.update(updates["metadata"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "session_id": self.session_id,
            "phone_number": self.phone_number,
            "patient": self.patient.to_dict() if self.patient else None,
            "conversation_history": [msg.to_dict() for msg in self.conversation_history],
            "selected_service": self.selected_service.to_dict() if self.selected_service else None,
            "booking_state": self.booking_state.to_dict(),
            "turn": self.turn,
            "language": self.language,
            "metadata": self.metadata,
            "last_discussed_service": self.last_discussed_service,
            "last_discussed_service_id": self.last_discussed_service_id,
            "conversation_topics": self.conversation_topics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationContext":
        """Create from dictionary"""
        patient = PatientInfo.from_dict(data["patient"]) if data.get("patient") else None
        
        history = []
        for msg_data in data.get("conversation_history", []):
            history.append(Message.from_dict(msg_data))
        
        selected_service = None
        if data.get("selected_service"):
            selected_service = ServiceInfo(**data["selected_service"])
        
        # Parse booking state
        booking_state_data = data.get("booking_state")
        if booking_state_data and isinstance(booking_state_data, dict):
            booking_state = BookingState.from_dict(booking_state_data)
        else:
            booking_state = BookingState()
        
        return cls(
            session_id=data["session_id"],
            phone_number=data["phone_number"],
            patient=patient,
            conversation_history=history,
            selected_service=selected_service,
            booking_state=booking_state,
            turn=data.get("turn", 0),
            language=data.get("language", "arabic"),
            metadata=data.get("metadata", {}),
            last_discussed_service=data.get("last_discussed_service"),
            last_discussed_service_id=data.get("last_discussed_service_id"),
            conversation_topics=data.get("conversation_topics", [])
        )
    
    def get_summary_for_llm(self) -> str:
        """
        Generate a summary of context for LLM system prompt.
        
        This is injected into Reem's system prompt to provide context.
        """
        summary_parts = []
        
        # Patient info
        if self.is_registered():
            summary_parts.append(f"👤 العميل: {self.patient.name} (مسجل)")
            summary_parts.append(f"📱 الهاتف: {self.patient.phone}")
            summary_parts.append(f"🔢 رقم الهوية: {self.patient.national_id or 'غير متوفر'}")
            summary_parts.append(f"⚧ الجنس: {self.patient.gender}")
        else:
            summary_parts.append("👤 العميل: غير مسجل (يحتاج تسجيل)")
        
        # Selected service
        if self.has_service_selected():
            summary_parts.append(f"💆 الخدمة المختارة: {self.selected_service.name_ar}")
            if self.selected_service.price:
                summary_parts.append(f"💰 السعر: {self.selected_service.price} ريال")
        
        # Last discussed service (CRITICAL FOR CONTEXT AWARENESS)
        if self.last_discussed_service:
            summary_parts.append(f"💬 آخر خدمة تم الحديث عنها: {self.last_discussed_service}")
            if self.last_discussed_service_id:
                summary_parts.append(f"🔢 معرف الخدمة: {self.last_discussed_service_id}")
        
        # Conversation topics
        if self.conversation_topics:
            recent_topics = self.conversation_topics[-3:]  # Last 3 topics
            summary_parts.append(f"📝 المواضيع الأخيرة: {', '.join(recent_topics)}")
        
        # Booking state
        if self.is_in_booking_flow():
            status_map = {
                "active": "نشط - جاري الحجز",
                "paused": f"متوقف مؤقتاً عند: {self.booking_state.paused_at_step}",
                "cancelled": "تم الإلغاء",
                "completed": "تم بنجاح"
            }
            status_text = status_map.get(self.booking_state.status, self.booking_state.status)
            summary_parts.append(f"📅 حالة الحجز: {status_text}")
            
            # Show collected data
            if self.booking_state.collected_data:
                summary_parts.append(f"📊 البيانات المجمعة: {len(self.booking_state.collected_data)} عنصر")
        
        # Turn info
        summary_parts.append(f"🔄 رقم المحادثة: {self.turn}")
        
        return "\n".join(summary_parts)


@dataclass
class SessionMetrics:
    """
    Metrics for monitoring session performance.
    """
    session_id: str
    total_turns: int = 0
    total_tokens: int = 0
    total_function_calls: int = 0
    total_workflows: int = 0
    average_response_time: float = 0.0
    first_message_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    conversion_status: Optional[str] = None  # "booked", "abandoned", "in_progress"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "total_turns": self.total_turns,
            "total_tokens": self.total_tokens,
            "total_function_calls": self.total_function_calls,
            "total_workflows": self.total_workflows,
            "average_response_time": self.average_response_time,
            "first_message_at": self.first_message_at.isoformat() if self.first_message_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "conversion_status": self.conversion_status
        }
