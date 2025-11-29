"""
Booking Helper - High-level API wrapper for smooth booking workflows.

This module provides intelligent wrappers around the AgentApiClient to handle:
- Service category → subcategory resolution
- Resource requirement detection (doctor/specialist/device)
- Complete booking workflows with validation
- Error handling and retries

Usage:
    helper = BookingHelper(api_client)
    booking = await helper.create_booking_smart(
        patient_id=123,
        service_name="Botox",
        date="2025-10-25"
    )
"""

from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class BookingHelper:
    """High-level helper for booking operations."""
    
    def __init__(self, api_client):
        """
        Initialize helper with API client.
        
        Args:
            api_client: AgentApiClient instance
        """
        self.api = api_client
        
    async def find_bookable_service(
        self, 
        service_name: Optional[str] = None,
        category_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a bookable service (subcategory) by name or category.
        
        Args:
            service_name: Service name to search for (optional)
            category_id: Category ID to get subcategories from (optional)
            
        Returns:
            First bookable service found, or None
            
        Example:
            service = await helper.find_bookable_service(service_name="Botox")
            service_id = service['id']
        """
        try:
            if category_id:
                # Get subcategories from specific category
                subcategories = await self.api.get_subcategory_services(category_id)
                if subcategories.get("results"):
                    return subcategories["results"][0]
            else:
                # Get first category, then its subcategories
                categories = await self.api.get_services(limit=20)
                
                for category in categories.get("results", []):
                    # Try to get subcategories
                    try:
                        subcategories = await self.api.get_subcategory_services(
                            category["id"], 
                            limit=10
                        )
                        
                        results = subcategories.get("results", [])
                        if results:
                            # If searching by name, filter
                            if service_name:
                                for svc in results:
                                    if service_name.lower() in svc.get("name", "").lower():
                                        return svc
                            else:
                                # Return first subcategory
                                return results[0]
                    except:
                        continue
                        
            return None
            
        except Exception as e:
            logger.error(f"Error finding bookable service: {e}")
            return None
    
    async def get_service_requirements(
        self, 
        service_id: int
    ) -> Dict[str, bool]:
        """
        Get resource requirements for a service.
        
        Args:
            service_id: Service ID
            
        Returns:
            Dict with 'requires_doctor', 'requires_specialist', 'requires_device' flags
            
        Example:
            reqs = await helper.get_service_requirements(122)
            if reqs['requires_doctor']:
                # Need to provide doctor_id
        """
        try:
            service = await self.api.get_service(service_id)
            
            return {
                "requires_doctor": (
                    service.get("requires_doctor") or 
                    service.get("require_doctor") or 
                    service.get("doctor_required", False)
                ),
                "requires_specialist": (
                    service.get("requires_specialist") or 
                    service.get("require_specialist") or 
                    service.get("specialist_required", False)
                ),
                "requires_device": (
                    service.get("requires_device") or 
                    service.get("require_device") or 
                    service.get("device_required", False)
                ),
                "duration_minutes": service.get("duration_minutes", 30)
            }
        except Exception as e:
            logger.error(f"Error getting service requirements: {e}")
            return {
                "requires_doctor": False,
                "requires_specialist": False,
                "requires_device": False,
                "duration_minutes": 30
            }
    
    async def find_available_resource(
        self,
        requires_doctor: bool = False,
        requires_specialist: bool = False,
        requires_device: bool = False
    ) -> Tuple[Optional[str], Optional[int]]:
        """
        Find an available resource based on requirements.
        
        Args:
            requires_doctor: Service requires doctor
            requires_specialist: Service requires specialist
            requires_device: Service requires device
            
        Returns:
            Tuple of (resource_type, resource_id) or (None, None)
            resource_type is one of: 'doctor_id', 'specialist_id', 'device_id'
            
        Example:
            resource_type, resource_id = await helper.find_available_resource(
                requires_doctor=True
            )
            # resource_type='doctor_id', resource_id=3
        """
        try:
            if requires_doctor:
                doctors = await self.api.get_doctors(limit=10)
                if doctors.get("results"):
                    return ("doctor_id", doctors["results"][0]["id"])
                    
            elif requires_specialist:
                specialists = await self.api.get_specialists(limit=10)
                if specialists.get("results"):
                    return ("specialist_id", specialists["results"][0]["id"])
                    
            elif requires_device:
                devices = await self.api.get_devices(limit=10)
                if devices.get("results"):
                    return ("device_id", devices["results"][0]["id"])
                    
            return (None, None)
            
        except Exception as e:
            logger.error(f"Error finding resource: {e}")
            return (None, None)
    
    async def get_first_available_slot(
        self,
        service_id: int,
        date: str,
        resource_type: str,
        resource_id: int,
        timeout: float = 120.0
    ) -> Optional[Dict[str, Any]]:
        """
        Get first available slot for a service on a specific date.
        
        Args:
            service_id: Subcategory service ID
            date: Date in YYYY-MM-DD format
            resource_type: 'doctor_id', 'specialist_id', or 'device_id'
            resource_id: Resource ID
            timeout: Request timeout
            
        Returns:
            First available slot dict with 'id' field, or None
            
        Example:
            slot = await helper.get_first_available_slot(
                service_id=122,
                date="2025-10-25",
                resource_type="doctor_id",
                resource_id=3
            )
            slot_id = slot['id']
        """
        try:
            kwargs = {
                "service_id": service_id,
                "date": date,
                "timeout": timeout
            }
            kwargs[resource_type] = resource_id
            
            slots = await self.api.get_available_slots(**kwargs)
            
            if slots.get("results"):
                return slots["results"][0]
                
            return None
            
        except Exception as e:
            logger.error(f"Error getting available slots: {e}")
            return None
    
    async def create_booking_smart(
        self,
        patient_id: int,
        service_name: Optional[str] = None,
        service_id: Optional[int] = None,
        date: Optional[str] = None,
        days_ahead: int = 2,
        auto_find_resource: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Smart booking creation with automatic service/resource resolution.
        
        This method handles the complete booking workflow:
        1. Find bookable service (if only name provided)
        2. Get service requirements
        3. Find available resource
        4. Get available slots
        5. Create booking
        
        Args:
            patient_id: Patient ID
            service_name: Service name to search for (if service_id not provided)
            service_id: Subcategory service ID (if known)
            date: Date in YYYY-MM-DD format (default: days_ahead from now)
            days_ahead: Days ahead to book (default: 2)
            auto_find_resource: Automatically find required resource
            
        Returns:
            Created booking dict with 'id' field, or None on failure
            
        Example:
            # Simple - just provide patient and service name
            booking = await helper.create_booking_smart(
                patient_id=123,
                service_name="Botox"
            )
            
            # Advanced - specify service ID and date
            booking = await helper.create_booking_smart(
                patient_id=123,
                service_id=122,
                date="2025-10-25"
            )
        """
        try:
            # Step 1: Find bookable service if needed
            if not service_id:
                logger.info(f"Finding bookable service for: {service_name}")
                service = await self.find_bookable_service(service_name=service_name)
                if not service:
                    logger.error("Could not find bookable service")
                    return None
                service_id = service["id"]
                logger.info(f"✅ Found service ID: {service_id}")
            
            # Step 2: Get service requirements
            logger.info(f"Getting service requirements for ID: {service_id}")
            requirements = await self.get_service_requirements(service_id)
            duration = requirements["duration_minutes"]
            logger.info(f"✅ Service requirements: {requirements}")
            
            # Step 3: Find required resource
            if auto_find_resource:
                logger.info("Finding available resource...")
                resource_type, resource_id = await self.find_available_resource(
                    requires_doctor=requirements["requires_doctor"],
                    requires_specialist=requirements["requires_specialist"],
                    requires_device=requirements["requires_device"]
                )
                
                if not resource_type or not resource_id:
                    logger.error("Could not find required resource")
                    return None
                    
                logger.info(f"✅ Found resource: {resource_type}={resource_id}")
            else:
                logger.error("Auto resource finding is disabled")
                return None
            
            # Step 4: Determine booking date
            if not date:
                date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            logger.info(f"Booking date: {date}")
            
            # Step 5: Get available slots
            logger.info("Getting available slots...")
            slot = await self.get_first_available_slot(
                service_id=service_id,
                date=date,
                resource_type=resource_type,
                resource_id=resource_id
            )
            
            if not slot:
                logger.error(f"No available slots for {date}")
                return None
                
            slot_id = slot["id"]
            logger.info(f"✅ Found slot ID: {slot_id}")
            
            # Step 6: Create booking
            logger.info("Creating booking...")
            booking_params = {
                "patient_id": patient_id,
                "service_id": service_id,
                "start_date": date,
                "slot_choice_id": slot_id,
                "duration_minutes": duration
            }
            booking_params[resource_type] = resource_id
            
            logger.info(f"Booking params: {booking_params}")
            booking = await self.api.create_booking(**booking_params)
            
            if booking and "id" in booking:
                logger.info(f"✅ Created booking ID: {booking['id']}")
                return booking
            else:
                logger.error("Booking created but no ID returned")
                return None
                
        except Exception as e:
            logger.error(f"Error in smart booking creation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def find_or_create_patient(
        self,
        phone: str,
        name: str,
        email: Optional[str] = None,
        city: str = "Riyadh"
    ) -> Optional[int]:
        """
        Find patient by phone or create new one.
        
        Args:
            phone: Patient phone number
            name: Patient name
            email: Patient email (optional)
            city: Patient city (default: Riyadh)
            
        Returns:
            Patient ID or None on failure
            
        Example:
            patient_id = await helper.find_or_create_patient(
                phone="0501234567",
                name="Ahmed Ali"
            )
        """
        try:
            # Try to find existing patient
            try:
                patient = await self.api.get_patient_by_phone(phone)
                if patient and "id" in patient:
                    logger.info(f"✅ Found existing patient ID: {patient['id']}")
                    return patient["id"]
            except:
                pass
            
            # Create new patient
            logger.info(f"Creating new patient: {name}")
            patient_data = {
                "name": name,
                "phone": phone,
                "city": city
            }
            if email:
                patient_data["email"] = email
                
            patient = await self.api.create_patient(**patient_data)
            
            if patient and "id" in patient:
                logger.info(f"✅ Created patient ID: {patient['id']}")
                return patient["id"]
                
            return None
            
        except Exception as e:
            logger.error(f"Error finding/creating patient: {e}")
            return None
