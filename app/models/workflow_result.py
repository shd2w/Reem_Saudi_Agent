"""
Workflow Result Models
======================
Data structures for workflow execution results.
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class WorkflowStatus(Enum):
    """Status of workflow execution"""
    SUCCESS = "success"
    PARTIAL = "partial"      # Partially complete, needs more input
    FAILED = "failed"
    PENDING = "pending"      # Waiting for user input


@dataclass
class WorkflowResult:
    """
    Result from workflow execution (LangGraph backend).
    
    This is what LangGraph returns after executing a workflow.
    NO conversation text - only structured data!
    
    Attributes:
        success: Whether workflow completed successfully
        status: Current status of workflow
        data: Structured data returned by workflow
        next_step: Next step required (if not complete)
        errors: List of errors encountered
        metadata: Execution metrics (time, nodes visited, etc.)
    
    Examples:
        # Successful booking
        WorkflowResult(
            success=True,
            status=WorkflowStatus.SUCCESS,
            data={
                "booking_id": "BK12345",
                "confirmation_code": "ABC123",
                "appointment_date": "2025-10-26",
                "appointment_time": "14:00",
                "doctor_name": "د. سارة الأحمد"
            },
            next_step=None,
            errors=[],
            metadata={"execution_time": 1.2, "nodes_visited": 5}
        )
        
        # Partial registration (needs more input)
        WorkflowResult(
            success=False,
            status=WorkflowStatus.PENDING,
            data={
                "name": "شادي سالم",
                "national_id": None  # Still missing
            },
            next_step="awaiting_national_id",
            errors=[],
            metadata={"current_step": "registration_id"}
        )
    """
    success: bool
    status: WorkflowStatus
    data: Dict[str, Any] = field(default_factory=dict)
    next_step: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_complete(self) -> bool:
        """Check if workflow is complete"""
        return self.status == WorkflowStatus.SUCCESS
    
    def is_pending(self) -> bool:
        """Check if workflow is waiting for input"""
        return self.status == WorkflowStatus.PENDING
    
    def has_errors(self) -> bool:
        """Check if workflow encountered errors"""
        return len(self.errors) > 0
    
    def get_error_message(self) -> Optional[str]:
        """Get first error message"""
        return self.errors[0] if self.errors else None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Used by ReemAgent.wrap_result() to serialize workflow results.
        """
        return {
            "success": self.success,
            "status": self.status.value if isinstance(self.status, WorkflowStatus) else str(self.status),
            "data": self.data,
            "next_step": self.next_step,
            "errors": self.errors,
            "metadata": self.metadata
        }


@dataclass
class PatientInfo:
    """
    Patient information from database.
    """
    id: int
    name: str
    phone: str
    national_id: Optional[str] = None
    gender: Optional[str] = None
    email: Optional[str] = None
    city: Optional[str] = None
    country_code: Optional[str] = None
    already_registered: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "phone": self.phone,
            "national_id": self.national_id,
            "gender": self.gender,
            "email": self.email,
            "city": self.city,
            "country_code": self.country_code,
            "already_registered": self.already_registered
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatientInfo":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            name=data["name"],
            phone=data["phone"],
            national_id=data.get("national_id"),
            gender=data.get("gender"),
            email=data.get("email"),
            city=data.get("city"),
            country_code=data.get("country_code"),
            already_registered=data.get("already_registered", True)
        )


@dataclass
class ServiceInfo:
    """
    Service information.
    """
    id: int
    name: str
    name_ar: str
    category: str
    price: Optional[float] = None
    duration: Optional[int] = None  # in minutes
    description: Optional[str] = None
    requires_doctor: bool = False
    requires_specialist: bool = False
    requires_device: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "name_ar": self.name_ar,
            "category": self.category,
            "price": self.price,
            "duration": self.duration,
            "description": self.description,
            "requires_doctor": self.requires_doctor,
            "requires_specialist": self.requires_specialist,
            "requires_device": self.requires_device
        }
