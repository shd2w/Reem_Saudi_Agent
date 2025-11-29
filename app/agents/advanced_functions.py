"""
Advanced Function Definitions for Dynamic Hybrid Architecture
==============================================================

Granular functions that Reem can call to execute ONE specific action.
Each function does ONE thing - no monolithic workflows!

Key Principles:
1. Functions are SMALL and FOCUSED
2. Functions are INTERRUPTIBLE by design
3. Functions return STRUCTURED data (no conversation text)
4. Reem wraps results naturally
"""
from typing import List, Dict, Any


def get_advanced_functions() -> List[Dict[str, Any]]:
    """
    Get function definitions for Reem's autonomous control.
    
    These are the "tools" Reem can use to execute technical operations.
    Reem decides WHEN and HOW to use them based on conversation context.
    """
    
    return [
        {
            "name": "execute_booking_step",
            "description": """
            Execute ONE step of the booking process (not entire workflow!).
            
            This does NOT complete the entire booking - just one specific step.
            You can call this multiple times, stopping between steps to talk to user.
            
            Available steps:
            - collect_service: Identify which service user wants
            - collect_datetime: Collect date and time from user (replaces old slot selection)
            - confirm_booking: Final confirmation and booking creation
            
            DEPRECATED:
            - check_availability: No longer needed (dates are flexible)
            - select_slot: Removed (use collect_datetime instead)
            
            🚨 CRITICAL for collect_service step:
            - ALWAYS extract service name from user's CURRENT message
            - Pass it in data parameter: {"service_name": "فل بدي"}
            - DO NOT rely on conversation history/context
            - User's explicit request ALWAYS takes priority
            
            Examples:
            User: "أبي أحجز بوتوكس"
            ✅ Call: execute_booking_step("collect_service", {"service_name": "بوتوكس"})
            
            User: "أبي أحجز فل بدي 6 جلسات"
            ✅ Call: execute_booking_step("collect_service", {"service_name": "فل بدي"})
            
            User: "أبي أحجز" (no service mentioned)
            ✅ Call: execute_booking_step("collect_service", {})
            (System will ask user which service)
            
            IMPORTANT: 
            - Call this ONLY when user is ready for that specific step
            - You can pause between ANY steps to answer questions
            - Don't rush through all steps at once
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "step_name": {
                        "type": "string",
                        "enum": ["collect_service", "collect_datetime", "confirm_booking"],
                        "description": "Which specific step to execute. Use 'collect_datetime' to get date+time from user."
                    },
                    "data": {
                        "type": "object",
                        "description": """Data for this step. For collect_service: {"service_name": "service_name_in_arabic"}. For other steps: relevant data like date, slot_id, etc.""",
                        "properties": {
                            "service_name": {
                                "type": "string",
                                "description": "Service name extracted from user's CURRENT message (Arabic)"
                            },
                            "service_id": {
                                "type": "integer",
                                "description": "Service ID if known"
                            },
                            "date": {
                                "type": "string",
                                "description": "Appointment date (YYYY-MM-DD). Extract from user: 'بكرة'=tomorrow, 'السبت'=next Saturday, etc."
                            },
                            "time": {
                                "type": "string",
                                "description": "Appointment time in HH:MM format (e.g., '15:00'). Extract from user: '3 العصر'=15:00, '10 الصبح'=10:00, etc."
                            },
                            "user_message": {
                                "type": "string",
                                "description": "User's original message for context (helps with date/time extraction)"
                            }
                        }
                    }
                },
                "required": ["step_name"]
            }
        },
        
        {
            "name": "pause_booking",
            "description": """
            Temporarily pause the booking process.
            
            Use this when:
            - User asks a question mid-booking
            - User changes topic
            - User seems hesitant or needs more info
            - User says "wait" or "hold on"
            
            The booking state is saved and can be resumed later.
            
            Example:
            Context: You're collecting date for booking
            User: "بس قبل كذا، وش الفرق بين البوتوكس والفيلر؟"
            You: Call pause_booking() → Answer question naturally
            
            DON'T call this if booking hasn't started yet!
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why pausing (for analytics): 'user_question', 'user_hesitant', 'topic_change'"
                    }
                }
            }
        },
        
        {
            "name": "resume_booking",
            "description": """
            Resume a previously paused booking.
            
            Use this when:
            - User finished asking questions and is ready to continue
            - User says "طيب نكمل" or "يلا احجز"
            - User answered your question about resuming
            
            DON'T call this automatically! Ask user first:
            "رجعنا نكمل الحجز؟"
            
            Only call if user confirms they want to continue.
            """,
            "parameters": {
                "type": "object",
                "properties": {}
            }
        },
        
        {
            "name": "cancel_booking",
            "description": """
            Cancel current booking process completely.
            
            Use this when:
            - User explicitly says "لا خلاص" or "ألغي"
            - User wants different service (cancel current, start new)
            - User says "بعدين" or "مو الحين"
            
            This clears all collected booking data.
            
            Example:
            Context: Booking Botox
            User: "لا خليها، أفكر في الفيلر أحسن"
            You: Call cancel_booking() → Start fresh conversation about Filler
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Why cancelling: 'user_changed_mind', 'different_service', 'not_ready'"
                    }
                }
            }
        },
        
        # DEPRECATED: get_service_details and get_pricing removed
        # Reason: Require service_id which LLM doesn't know - causes hallucination
        # Solution: Use search_services(query="service_name") instead
        # Example: search_services(query="ليزر كربوني") instead of get_service_details(service_id=?)
        
        # {
        #     "name": "get_service_details",
        #     "description": "DEPRECATED - Use search_services() instead",
        #     "parameters": {"type": "object", "properties": {}}
        # },
        # {
        #     "name": "get_pricing",
        #     "description": "DEPRECATED - Use search_services() instead",
        #     "parameters": {"type": "object", "properties": {}}
        # },
        
        # DEPRECATED: check_availability - No longer needed
        # Modern flow: Just ask "متى تبي الموعد؟" and collect date+time directly
        # Old complex flow: service → check slots → show list → user selects slot
        # New simple flow: service → ask "متى تبي؟" → user says "بكرة 3 العصر" → book directly
        
        {
            "name": "search_services",
            "description": """
            Search services - CRITICAL: Extract COMPLETE user request!
            
            🚨 EXTREMELY IMPORTANT - Extract ALL Details:
            When user says "4 جلسات ليزر رجال فل بدي":
            ❌ WRONG: query="فل بدي"
            ✅ CORRECT: query="4 جلسات ليزر رجال"
            
            ALWAYS preserve:
            - Session count: "4 جلسات", "6 جلسات", "8 جلسات"
            - Service type: "ليزر", "بوتوكس", "فيلر"
            - Gender: "رجال", "نساء" (CRITICAL!)
            - Area: "فل بدي", "وجه", "منطقة صغيرة"
            
            Use when:
            - User wants to book a service
            - User asks about specific service
            - User mentions symptoms/needs
            
            Examples:
            User: "ابي احجز 4 جلسات ليزر رجال"
            ✅ Call: search_services(query="4 جلسات ليزر رجال")
            
            User: "عندي تجاعيد في الجبين"
            ✅ Call: search_services(query="تجاعيد جبين")
            
            User: "ابي بوتوكس للوجه كامل"
            ✅ Call: search_services(query="بوتوكس وجه كامل")
            
            The system will show numbered list - user will select number.
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "COMPLETE search query from user's message - preserve ALL details (session count, gender, service type, area)"
                    }
                },
                "required": ["query"]
            }
        },
        
        {
            "name": "get_all_services",
            "description": """
            Get ALL available services and offers.
            
            Use when:
            - User asks "وش عندكم؟"
            - User asks "وش عروضكم؟"
            - User wants to see everything
            
            Returns: List of all services with categories
            
            Example:
            User: "وش الخدمات اللي عندكم؟"
            You: Call get_all_services()
            You: "عندنا مجموعة رائعة! بوتوكس، فيلر، ليزر..."
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional: Filter by category",
                        "enum": ["", "laser", "botox", "fillers", "skin", "hair", "body"]
                    }
                }
            }
        },
        
        {
            "name": "view_my_bookings",
            "description": """
            Show patient's current and upcoming bookings.
            
            🧠 SMART: Fetches from API database - always shows real, current bookings.
            
            Use when:
            - User asks "وش حجوزاتي؟"
            - User asks "عندي مواعيد؟"
            - User wants to check their bookings
            - User asks "متى موعدي؟"
            - User wants to see appointment details
            
            Returns: List of patient's bookings with dates, times, services
            
            Examples:
            User: "وش مواعيدي القادمة؟"
            ✅ Call: view_my_bookings()
            
            User: "عندي حجز بكرة؟"
            ✅ Call: view_my_bookings()
            
            User: "اعرض لي حجوزاتي"
            ✅ Call: view_my_bookings()
            """,
            "parameters": {
                "type": "object",
                "properties": {
                    "show_past": {
                        "type": "boolean",
                        "description": "Include past bookings (default: false, only upcoming)"
                    }
                }
            }
        }
    ]


def get_function_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata about functions for intelligent handling.
    
    This helps the system understand which functions need follow-up responses.
    """
    
    return {
        "execute_booking_step": {
            "needs_wrapping": True,
            "returns_data": True,
            "modifies_state": True,
            "interruptible": True
        },
        "pause_booking": {
            "needs_wrapping": False,
            "returns_data": False,
            "modifies_state": True,
            "interruptible": False
        },
        "resume_booking": {
            "needs_wrapping": True,
            "returns_data": True,
            "modifies_state": True,
            "interruptible": False
        },
        "cancel_booking": {
            "needs_wrapping": False,
            "returns_data": False,
            "modifies_state": True,
            "interruptible": False
        },
        "get_service_details": {
            "needs_wrapping": True,
            "returns_data": True,
            "modifies_state": False,
            "interruptible": False
        },
        "get_pricing": {
            "needs_wrapping": True,
            "returns_data": True,
            "modifies_state": False,
            "interruptible": False
        },
        "check_availability": {
            "needs_wrapping": True,
            "returns_data": True,
            "modifies_state": False,
            "interruptible": False
        },
        "search_services": {
            "needs_wrapping": True,
            "returns_data": True,
            "modifies_state": False,
            "interruptible": False
        },
        "get_all_services": {
            "needs_wrapping": True,
            "returns_data": True,
            "modifies_state": False,
            "interruptible": False
        }
    }
