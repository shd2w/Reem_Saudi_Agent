"""
API Operations Helper - Complete Coverage
==========================================
Provides easy-to-use functions for all backend API operations.

Supported Operations:
- Bookings: GET all, GET single, POST create, POST reschedule
- Patients: GET all, GET single, POST create, PUT update
- Services: GET all, GET single
- Doctors: GET all, GET single
- Specialists: GET all, GET single
- Devices: GET all, GET single
- Slots: GET slots
- Rating: GET all, GET single, POST create
- Wallet: GET wallet, POST create
- Memo: POST create, GET/DELETE

Author: Agent Orchestrator Team
Version: 1.0.0
"""

from typing import Dict, Any, Optional, List
from loguru import logger
from .agent_api import AgentApiClient


class ApiOperations:
    """
    Complete API operations for all endpoints.
    """
    
    def __init__(self):
        self.client = AgentApiClient()
    
    # ==================== BOOKING OPERATIONS ====================
    
    async def get_all_bookings(self, params: Optional[Dict] = None) -> List[Dict]:
        """
        Get all bookings with optional filters.
        
        Args:
            params: Query parameters (e.g., {"patient_id": 123, "status": "confirmed"})
            
        Returns:
            List of booking objects
        """
        try:
            result = await self.client.get("/bookings", params=params)
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get all bookings error: {e}")
            return []
    
    async def get_booking(self, booking_id: int) -> Optional[Dict]:
        """
        Get single booking by ID.
        
        Args:
            booking_id: Booking ID
            
        Returns:
            Booking object or None
        """
        try:
            result = await self.client.get(f"/bookings/{booking_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Get booking {booking_id} error: {e}")
            return None
    
    async def create_booking(
        self,
        patient_id: int,
        service_id: int,
        appointment_date: str,
        appointment_time: str,
        doctor_id: Optional[int] = None,
        notes: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create a new booking.
        
        Args:
            patient_id: Patient ID
            service_id: Service ID
            appointment_date: Date in YYYY-MM-DD format
            appointment_time: Time in HH:MM format
            doctor_id: Optional doctor ID
            notes: Optional booking notes
            
        Returns:
            Created booking object or None
        """
        try:
            data = {
                "patient_id": patient_id,
                "service_id": service_id,
                "appointment_date": appointment_date,
                "appointment_time": appointment_time,
                "status": "confirmed"
            }
            
            if doctor_id:
                data["doctor_id"] = doctor_id
            if notes:
                data["notes"] = notes
            
            result = await self.client.post("/bookings", json_body=data)
            logger.info(f"✅ Booking created successfully: {result.get('id')}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Create booking error: {e}")
            return None
    
    async def reschedule_booking(
        self,
        booking_id: int,
        new_date: str,
        new_time: str
    ) -> Optional[Dict]:
        """
        Reschedule an existing booking.
        
        Args:
            booking_id: Booking ID to reschedule
            new_date: New date in YYYY-MM-DD format
            new_time: New time in HH:MM format
            
        Returns:
            Updated booking object or None
        """
        try:
            data = {
                "booking_id": booking_id,
                "new_date": new_date,
                "new_time": new_time
            }
            
            result = await self.client.post("/bookings/reschedule", json_body=data)
            logger.info(f"✅ Booking {booking_id} rescheduled successfully")
            return result.get("data")
        except Exception as e:
            logger.error(f"Reschedule booking error: {e}")
            return None
    
    # ==================== PATIENT OPERATIONS ====================
    
    async def get_all_patients(self, params: Optional[Dict] = None) -> List[Dict]:
        """Get all patients with optional filters."""
        try:
            result = await self.client.get("/patients", params=params)
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get all patients error: {e}")
            return []
    
    async def get_patient(self, patient_id: int) -> Optional[Dict]:
        """Get single patient by ID."""
        try:
            result = await self.client.get(f"/patients/{patient_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Get patient {patient_id} error: {e}")
            return None
    
    async def get_patient_by_phone(self, phone: str) -> Optional[Dict]:
        """Get patient by phone number."""
        try:
            result = await self.client.get("/patients", params={"phone": phone})
            patients = result.get("data", [])
            return patients[0] if patients else None
        except Exception as e:
            logger.error(f"Get patient by phone error: {e}")
            return None
    
    async def create_patient(
        self,
        name: str,
        phone: str,
        gender: Optional[str] = None,
        date_of_birth: Optional[str] = None,
        email: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create a new patient.
        
        Args:
            name: Patient full name
            phone: Phone number
            gender: Optional gender (male/female)
            date_of_birth: Optional DOB in YYYY-MM-DD format
            email: Optional email
            
        Returns:
            Created patient object or None
        """
        try:
            # CRITICAL: API expects specific field names (400 error fix)
            data = {
                "name": name,
                "patient_phone": phone,  # API expects "patient_phone" not "phone"
                "identification_id": ""  # API requires this field (empty if not provided)
            }
            
            if gender:
                data["gender"] = gender
            if date_of_birth:
                data["date_of_birth"] = date_of_birth
            if email:
                data["email"] = email
            
            # CRITICAL: Use /customer/create endpoint, not /patients (405 error fix)
            result = await self.client.post("/customer/create", json_body=data)
            logger.info(f"✅ Patient created successfully: {name}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Create patient error: {e}")
            return None
    
    async def update_patient(
        self,
        patient_id: int,
        updates: Dict[str, Any]
    ) -> Optional[Dict]:
        """
        Update patient information.
        
        Args:
            patient_id: Patient ID to update
            updates: Dictionary of fields to update
            
        Returns:
            Updated patient object or None
        """
        try:
            # CRITICAL: Use /customer/update/{id} endpoint, not /patients/{id}
            result = await self.client.put(f"/customer/update/{patient_id}", json_body=updates)
            logger.info(f"✅ Patient {patient_id} updated successfully")
            return result.get("data")
        except Exception as e:
            logger.error(f"Update patient {patient_id} error: {e}")
            return None
    
    # ==================== SERVICE OPERATIONS ====================
    
    async def get_all_services(self) -> List[Dict]:
        """Get all services."""
        try:
            result = await self.client.get("/services")
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get all services error: {e}")
            return []
    
    async def get_service(self, service_id: int) -> Optional[Dict]:
        """Get single service by ID."""
        try:
            result = await self.client.get(f"/services/{service_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Get service {service_id} error: {e}")
            return None
    
    # ==================== DOCTOR OPERATIONS ====================
    
    async def get_all_doctors(self) -> List[Dict]:
        """Get all doctors."""
        try:
            result = await self.client.get("/doctors")
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get all doctors error: {e}")
            return []
    
    async def get_doctor(self, doctor_id: int) -> Optional[Dict]:
        """Get single doctor by ID."""
        try:
            result = await self.client.get(f"/doctors/{doctor_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Get doctor {doctor_id} error: {e}")
            return None
    
    # ==================== SPECIALIST OPERATIONS ====================
    
    async def get_all_specialists(self) -> List[Dict]:
        """Get all specialists."""
        try:
            result = await self.client.get("/specialists")
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get all specialists error: {e}")
            return []
    
    async def get_specialist(self, specialist_id: int) -> Optional[Dict]:
        """Get single specialist by ID."""
        try:
            result = await self.client.get(f"/specialists/{specialist_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Get specialist {specialist_id} error: {e}")
            return None
    
    # ==================== DEVICE OPERATIONS ====================
    
    async def get_all_devices(self) -> List[Dict]:
        """Get all devices."""
        try:
            result = await self.client.get("/devices")
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get all devices error: {e}")
            return []
    
    async def get_device(self, device_id: int) -> Optional[Dict]:
        """Get single device by ID."""
        try:
            result = await self.client.get(f"/devices/{device_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Get device {device_id} error: {e}")
            return None
    
    # ==================== SLOT OPERATIONS ====================
    
    async def get_slots(
        self,
        service_id: Optional[int] = None,
        doctor_id: Optional[int] = None,
        date: Optional[str] = None
    ) -> List[Dict]:
        """
        Get available time slots.
        
        Args:
            service_id: Optional service filter
            doctor_id: Optional doctor filter
            date: Optional date filter (YYYY-MM-DD)
            
        Returns:
            List of available slots
        """
        try:
            params = {}
            if service_id:
                params["service_id"] = service_id
            if doctor_id:
                params["doctor_id"] = doctor_id
            if date:
                params["date"] = date
            
            result = await self.client.get("/slots", params=params)
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get slots error: {e}")
            return []
    
    # ==================== RATING OPERATIONS ====================
    
    async def get_all_ratings(self, params: Optional[Dict] = None) -> List[Dict]:
        """Get all ratings with optional filters."""
        try:
            result = await self.client.get("/ratings", params=params)
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get all ratings error: {e}")
            return []
    
    async def get_rating(self, rating_id: int) -> Optional[Dict]:
        """Get single rating by ID."""
        try:
            result = await self.client.get(f"/ratings/{rating_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Get rating {rating_id} error: {e}")
            return None
    
    async def create_rating(
        self,
        booking_id: int,
        rating: int,
        comment: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create a new rating.
        
        Args:
            booking_id: Booking ID to rate
            rating: Rating value (1-5)
            comment: Optional comment
            
        Returns:
            Created rating object or None
        """
        try:
            data = {
                "booking_id": booking_id,
                "rating": rating
            }
            
            if comment:
                data["comment"] = comment
            
            result = await self.client.post("/ratings", json_body=data)
            logger.info(f"✅ Rating created for booking {booking_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Create rating error: {e}")
            return None
    
    # ==================== WALLET OPERATIONS ====================
    
    async def get_wallet(self, patient_id: int) -> Optional[Dict]:
        """
        Get patient wallet information.
        
        Args:
            patient_id: Patient ID
            
        Returns:
            Wallet object or None
        """
        try:
            result = await self.client.get(f"/wallet/{patient_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Get wallet error: {e}")
            return None
    
    async def create_wallet(
        self,
        patient_id: int,
        initial_balance: float = 0.0
    ) -> Optional[Dict]:
        """
        Create a new wallet for patient.
        
        Args:
            patient_id: Patient ID
            initial_balance: Starting balance
            
        Returns:
            Created wallet object or None
        """
        try:
            data = {
                "patient_id": patient_id,
                "balance": initial_balance
            }
            
            result = await self.client.post("/wallet", json_body=data)
            logger.info(f"✅ Wallet created for patient {patient_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Create wallet error: {e}")
            return None
    
    # ==================== MEMO OPERATIONS ====================
    
    async def create_memo(
        self,
        patient_id: int,
        content: str,
        memo_type: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Create a new memo for patient.
        
        Args:
            patient_id: Patient ID
            content: Memo content
            memo_type: Optional memo type/category
            
        Returns:
            Created memo object or None
        """
        try:
            data = {
                "patient_id": patient_id,
                "content": content
            }
            
            if memo_type:
                data["type"] = memo_type
            
            result = await self.client.post("/memos", json_body=data)
            logger.info(f"✅ Memo created for patient {patient_id}")
            return result.get("data")
        except Exception as e:
            logger.error(f"Create memo error: {e}")
            return None
    
    async def get_memos(self, patient_id: int) -> List[Dict]:
        """Get all memos for a patient."""
        try:
            result = await self.client.get(f"/memos", params={"patient_id": patient_id})
            return result.get("data", [])
        except Exception as e:
            logger.error(f"Get memos error: {e}")
            return []
    
    async def delete_memo(self, memo_id: int) -> bool:
        """Delete a memo."""
        try:
            await self.client.delete(f"/memos/{memo_id}")
            logger.info(f"✅ Memo {memo_id} deleted")
            return True
        except Exception as e:
            logger.error(f"Delete memo error: {e}")
            return False


# Singleton instance
_api_operations = None

def get_api_operations() -> ApiOperations:
    """Get singleton API operations instance."""
    global _api_operations
    if _api_operations is None:
        _api_operations = ApiOperations()
    return _api_operations
