"""
Selection Handler
==================
Handles numbered selections from users when multiple options are presented.

Following intelligent_booking_agent patterns for selection tracking.
"""
from typing import Dict, Any, Optional
from loguru import logger

from ..models.conversation_context import ConversationContext


class SelectionHandler:
    """
    Handle numbered selections from users.
    
    When user sends a number (e.g., "1", "2", "3"), this handler:
    1. Detects it's a number selection
    2. Retrieves the last displayed list from context
    3. Extracts the corresponding item
    4. Saves to booking state
    5. Returns the selected item for further processing
    """
    
    @staticmethod
    def is_number_selection(message: str) -> bool:
        """
        Check if message is a simple number selection.
        
        Args:
            message: User's message
            
        Returns:
            True if message is a pure number (will validate range later)
        """
        message_clean = message.strip()
        
        # Check if it's a pure digit (any number, we'll validate range in handle_selection)
        if not message_clean.isdigit():
            return False
        
        # Convert to int - accept any positive number
        try:
            num = int(message_clean)
            return num > 0  # Any positive number is a potential selection
        except ValueError:
            return False
    
    @staticmethod
    async def handle_selection(
        number: int,
        context: ConversationContext
    ) -> Dict[str, Any]:
        """
        Handle user's number selection.
        
        Args:
            number: The number user selected (1-indexed)
            context: Conversation context with last displayed list
            
        Returns:
            Dict with success status and selected item data
        """
        
        logger.info(f"🔢 User selected number: {number}")
        
        # Get last displayed list from context
        last_list = context.metadata.get("last_displayed_list")
        list_type = context.metadata.get("last_list_type", "unknown")
        
        if not last_list:
            logger.warning(f"⚠️ No selection list found in context")
            return {
                "success": False,
                "message": "لم أجد قائمة للاختيار منها. ممكن تعيد طلبك؟"
            }
        
        # Validate selection is within range
        if number < 1 or number > len(last_list):
            logger.warning(f"⚠️ Invalid selection: {number} (list has {len(last_list)} items)")
            return {
                "success": False,
                "message": f"الرجاء اختيار رقم من 1 إلى {len(last_list)}"
            }
        
        # Get selected item (convert to 0-indexed)
        selected_item = last_list[number - 1]
        
        logger.info(f"✅ Selected item from {list_type} list: {selected_item.get('name', selected_item.get('id'))}")
        
        # Handle based on list type
        if list_type == "services":
            # Save service selection to booking state
            context.booking_state.collected_data["service_id"] = selected_item["id"]
            context.booking_state.collected_data["service_name"] = selected_item.get("name")
            context.booking_state.collected_data["service_price"] = selected_item.get("price")
            
            logger.info(f"📥 Saved service_id={selected_item['id']} to booking_state")
            
            # Clear the displayed list (selection consumed)
            context.metadata["last_displayed_list"] = None
            context.metadata["last_list_type"] = None
            
            return {
                "success": True,
                "selection_type": "service",
                "data": {
                    "service_id": selected_item["id"],
                    "service_name": selected_item.get("name"),
                    "service_price": selected_item.get("price"),
                    "requires_doctor": selected_item.get("requires_doctor", False),
                    "requires_specialist": selected_item.get("requires_specialist", False),
                    "requires_device": selected_item.get("requires_device", False),
                    "duration_minutes": selected_item.get("duration_minutes", 60)
                },
                "message": f"تم اختيار: {selected_item.get('name')}",
                "next_step": "check_availability"
            }
        
        elif list_type == "doctors":
            # Save doctor selection
            context.booking_state.collected_data["doctor_id"] = selected_item["id"]
            context.booking_state.collected_data["doctor_name"] = selected_item.get("name")
            
            logger.info(f"📥 Saved doctor_id={selected_item['id']} to booking_state")
            
            # Clear the displayed list
            context.metadata["last_displayed_list"] = None
            context.metadata["last_list_type"] = None
            
            return {
                "success": True,
                "selection_type": "doctor",
                "data": {
                    "doctor_id": selected_item["id"],
                    "doctor_name": selected_item.get("name")
                },
                "message": f"تم اختيار الدكتور: {selected_item.get('name')}",
                "next_step": "check_availability"
            }
        
        elif list_type == "slots":
            # Save slot selection
            context.booking_state.collected_data["slot_id"] = selected_item.get("id")
            context.booking_state.collected_data["slot_time"] = selected_item.get("time")
            context.booking_state.collected_data["slot_date"] = selected_item.get("date")
            
            logger.info(f"📥 Saved slot selection: {selected_item.get('date')} at {selected_item.get('time')}")
            
            # Clear the displayed list
            context.metadata["last_displayed_list"] = None
            context.metadata["last_list_type"] = None
            
            return {
                "success": True,
                "selection_type": "slot",
                "data": {
                    "slot_id": selected_item.get("id"),
                    "slot_time": selected_item.get("time"),
                    "slot_date": selected_item.get("date")
                },
                "message": f"تم اختيار الموعد: {selected_item.get('date')} الساعة {selected_item.get('time')}",
                "next_step": "confirm_booking"
            }
        
        else:
            logger.warning(f"⚠️ Unknown list type: {list_type}")
            return {
                "success": False,
                "message": "نوع القائمة غير معروف"
            }
