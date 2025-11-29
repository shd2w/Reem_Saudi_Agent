"""
Appointment Reminder Service
=============================
Automated reminder system for upcoming appointments.

Features:
- Scheduled reminders (24h, 2h before appointment)
- WhatsApp notification sending
- Reminder status tracking
- Configurable reminder times

Author: Agent Orchestrator Team
Version: 1.0.0
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger

from ..api.agent_api import AgentApiClient
from ..api.wasender_client import WaSenderClient
from ..memory.session_manager import SessionManager


class ReminderService:
    """
    Professional reminder service for appointment notifications.
    
    Sends automated reminders via WhatsApp at configured intervals.
    """
    
    def __init__(self):
        self.api_client = AgentApiClient()
        self.wasender_client = WaSenderClient()
        self.session_manager = SessionManager()
        
        # Reminder intervals (hours before appointment)
        self.reminder_intervals = [24, 2]  # 24 hours and 2 hours before
        
        logger.info("Reminder Service initialized")
    
    async def check_and_send_reminders(self) -> Dict[str, Any]:
        """
        Check for upcoming appointments and send reminders.
        
        Returns:
            Summary of reminders sent
        """
        try:
            logger.info("🔔 Checking for appointments needing reminders...")
            
            # Get upcoming appointments
            now = datetime.now()
            tomorrow = now + timedelta(days=2)
            
            bookings_result = await self.api_client.get("/booking", params={
                "state": "confirmed",
                "date_from": now.strftime("%Y-%m-%d"),
                "date_to": tomorrow.strftime("%Y-%m-%d"),
                "limit": 100
            })
            
            if not bookings_result or not bookings_result.get("data"):
                logger.info("No upcoming appointments found")
                return {
                    "checked": True,
                    "reminders_sent": 0,
                    "appointments_checked": 0
                }
            
            bookings = bookings_result["data"]
            reminders_sent = 0
            
            for booking in bookings:
                sent = await self._process_booking_reminders(booking, now)
                if sent:
                    reminders_sent += 1
            
            logger.info(f"✅ Sent {reminders_sent} reminders for {len(bookings)} appointments")
            
            return {
                "checked": True,
                "reminders_sent": reminders_sent,
                "appointments_checked": len(bookings),
                "timestamp": now.isoformat()
            }
            
        except Exception as exc:
            logger.error(f"Reminder check error: {exc}", exc_info=True)
            return {
                "checked": False,
                "error": str(exc)
            }
    
    async def _process_booking_reminders(self, booking: Dict[str, Any], now: datetime) -> bool:
        """
        Process reminders for a single booking.
        
        Args:
            booking: Booking data
            now: Current datetime
            
        Returns:
            True if reminder was sent
        """
        try:
            booking_id = booking.get("id")
            appointment_date = booking.get("date")
            appointment_time = booking.get("time", "00:00")
            
            if not appointment_date:
                return False
            
            # Parse appointment datetime
            appointment_dt = datetime.strptime(
                f"{appointment_date} {appointment_time}",
                "%Y-%m-%d %H:%M"
            )
            
            # Calculate time until appointment
            time_until = appointment_dt - now
            hours_until = time_until.total_seconds() / 3600
            
            # Check if we should send a reminder
            reminder_to_send = None
            for interval in self.reminder_intervals:
                # Send reminder if we're within 30 minutes of the interval
                if abs(hours_until - interval) <= 0.5:
                    reminder_to_send = interval
                    break
            
            if not reminder_to_send:
                return False
            
            # Check if we already sent this reminder
            reminder_key = f"reminder_{booking_id}_{reminder_to_send}h"
            if self._reminder_already_sent(reminder_key):
                return False
            
            # Get patient info
            patient = booking.get("patient", {})
            patient_name = patient.get("name", "Patient")
            patient_phone = patient.get("phone")
            
            if not patient_phone:
                logger.warning(f"No phone number for booking {booking_id}")
                return False
            
            # Get service info
            service = booking.get("service", {})
            service_name = service.get("name", "appointment")
            
            # Format reminder message
            if reminder_to_send == 24:
                message = f"""🔔 Appointment Reminder

Hello {patient_name}!

This is a reminder that you have an appointment tomorrow:

📅 Service: {service_name}
🕐 Date & Time: {appointment_date} at {appointment_time}
🆔 Booking ID: {booking_id}

Please arrive 10 minutes early. If you need to cancel or reschedule, please let us know at least 2 hours in advance.

See you tomorrow!
Wajan Medical Center"""
            else:  # 2 hour reminder
                message = f"""🔔 Appointment in 2 Hours!

Hello {patient_name}!

Your appointment is coming up soon:

📅 Service: {service_name}
🕐 Time: {appointment_time}
🆔 Booking ID: {booking_id}

Please arrive 10 minutes early.

See you soon!
Wajan Medical Center"""
            
            # Send reminder via WhatsApp
            send_result = await self.wasender_client.send_message(
                phone_number=patient_phone,
                message=message
            )
            
            if send_result:
                # Mark reminder as sent
                self._mark_reminder_sent(reminder_key)
                logger.info(f"✅ Sent {reminder_to_send}h reminder for booking {booking_id}")
                return True
            
            return False
            
        except Exception as exc:
            logger.error(f"Process booking reminder error: {exc}", exc_info=True)
            return False
    
    def _reminder_already_sent(self, reminder_key: str) -> bool:
        """Check if reminder was already sent"""
        try:
            session_data = self.session_manager.get("reminders_sent") or {}
            return reminder_key in session_data.get("sent", [])
        except:
            return False
    
    def _mark_reminder_sent(self, reminder_key: str) -> None:
        """Mark reminder as sent"""
        try:
            session_data = self.session_manager.get("reminders_sent") or {}
            
            if "sent" not in session_data:
                session_data["sent"] = []
            
            session_data["sent"].append(reminder_key)
            
            # Keep only last 1000 reminders
            session_data["sent"] = session_data["sent"][-1000:]
            
            # Store with 7 day TTL
            self.session_manager.put("reminders_sent", session_data, ttl_minutes=10080)
            
        except Exception as exc:
            logger.error(f"Mark reminder sent error: {exc}")
    
    async def send_manual_reminder(self, booking_id: str) -> Dict[str, Any]:
        """
        Send a manual reminder for a specific booking.
        
        Args:
            booking_id: Booking ID
            
        Returns:
            Send result
        """
        try:
            # Get booking details
            booking_result = await self.api_client.get(f"/booking/{booking_id}")
            
            if not booking_result or booking_result.get("error"):
                return {
                    "success": False,
                    "error": "Booking not found"
                }
            
            booking = booking_result.get("data", booking_result)
            
            # Send reminder
            sent = await self._process_booking_reminders(booking, datetime.now())
            
            return {
                "success": sent,
                "booking_id": booking_id
            }
            
        except Exception as exc:
            logger.error(f"Manual reminder error: {exc}")
            return {
                "success": False,
                "error": str(exc)
            }


# Singleton instance
_reminder_service = None

def get_reminder_service() -> ReminderService:
    """Get singleton reminder service instance"""
    global _reminder_service
    if _reminder_service is None:
        _reminder_service = ReminderService()
    return _reminder_service
