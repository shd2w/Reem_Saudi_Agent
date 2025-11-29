# -*- coding: utf-8 -*-
"""
Intelligent Booking Agent - AI-Powered with OpenAI Function Calling
===================================================================
TRUE AI agent that uses LLM to understand context and decide actions.
NO hardcoded routing logic - LLM has all the tools and chooses what to do.

Architecture:
- LLM as the brain (decides everything)
- Function calling for API interactions
- Contextual understanding

Author: Agent Orchestrator Team
Version: 2.0.0 (Complete Rewrite)
"""

import asyncio
import json
import re
from typing import List, Dict, Any, Optional
from loguru import logger
from datetime import datetime, timedelta
from openai import AsyncOpenAI
from app.utils.date_parser import extract_date_from_context

class IntelligentBookingAgent:
    """
{{ ... }}
    AI-powered booking agent using OpenAI Function Calling.
    
    The LLM decides:
    - What information to gather
    - When to call which API
    - How to respond naturally
    - Conversation flow
    
    NO hardcoded if/else logic!
    """
    
    def __init__(self, api_client, openai_api_key: str):
        """
        Initialize intelligent agent with API access and OpenAI.
        
        Args:
            api_client: AgentApiClient for backend API calls
            openai_api_key: OpenAI API key for LLM
        """
        self.api_client = api_client
        self.llm = AsyncOpenAI(
            api_key=openai_api_key,
            timeout=15.0,  # Reduce from default 60s to 15s
            max_retries=1   # Reduce from default 2 to 1
        )
        self.model = "gpt-4o-mini"  # Switch to gpt-4o if you need maximum intelligence
        
        # Track last search query for filtering variants
        self.last_search_query = None
        
        logger.info(f"🤖 IntelligentBookingAgent initialized with {self.model}")
    
    # ========================================================================
    # TOOL DEFINITIONS - All available functions LLM can call
    # ========================================================================
    
    def get_tools_definition(self) -> List[Dict[str, Any]]:
        """
        Define ALL available tools/functions that LLM can use.
        
        This is the "toolbox" - LLM chooses which tool to use based on conversation.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_services",
                    "description": "STEP 1: Get SERVICE CATEGORIES (NOT bookable yet!). Returns high-level categories like 'Laser Men', 'Botox', 'Fillers'. AFTER calling this, you MUST show ALL categories to user and wait for their selection. DO NOT proceed to get_service_variants until user selects a category number. Returns: {services: [{service_type_id, service_type_name, ...}]}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (e.g., 'ليزر', 'بوتوكس', 'فيلر'). Use general terms."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 20)",
                                "default": 20
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_service_details",
                    "description": "Get detailed information about a specific service including price, duration, requirements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_id": {
                                "type": "integer",
                                "description": "The service ID"
                            }
                        },
                        "required": ["service_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_all_services_offers",
                    "description": "Get ALL clinic services/offers with prices. Call this when user asks 'What are your offers?', 'Show me all services', 'What do you have?', 'عندكم وش؟', 'وش العروض؟'. Returns complete list of all available services with prices. DO NOT use this for booking - this is ONLY for showing offers/services to interested customers.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filter_query": {
                                "type": "string",
                                "description": "Optional filter (e.g., 'ليزر', 'بوتوكس'). Leave empty to show ALL services.",
                                "default": ""
                            },
                            "show_categories_only": {
                                "type": "boolean",
                                "description": "If true, show only main categories. If false, show detailed services with prices.",
                                "default": False
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_service_variants",
                    "description": "STEP 2: Get BOOKABLE SUBSERVICES for a category. CRITICAL: Call this ONLY AFTER user selected a category from search_services. Returns 10-20 actual bookable services with prices (e.g., 'Small Area Laser - 100 SAR', 'Full Face Laser - 500 SAR'). AFTER calling this, you MUST show ALL subservices with numbers and prices to user. DO NOT call get_available_slots until user picks a subservice number. Input: service_type_id from search_services. Returns: {variants: [{id (THIS IS service_id!), name, price, duration, requires_doctor, requires_device, ...}]}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_type_id": {
                                "type": "integer",
                                "description": "The service_type_id from the category user selected in search_services results"
                            }
                        },
                        "required": ["service_type_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_available_slots",
                    "description": "STEP 3: Get time slots for booking. CRITICAL: Call this ONLY AFTER user selected a subservice number from get_service_variants list. The service_id MUST be the 'id' field from the subservice user selected (NOT service_type_id from search_services!). If the subservice has requires_doctor=true, you must provide doctor_id. If requires_device=true, provide device_id. Returns: {slots: [{time, date, slot_choice_id, ...}]}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service_id": {
                                "type": "integer",
                                "description": "The 'id' field from the subservice the user selected in get_service_variants results (e.g., if user selected item #3 from variants list, use variants[2].id)"
                            },
                            "date": {
                                "type": "string",
                                "description": "Preferred date (YYYY-MM-DD format). If not specified, use tomorrow."
                            },
                            "patient_id": {
                                "type": "integer",
                                "description": "Patient ID"
                            },
                            "doctor_id": {
                                "type": "integer",
                                "description": "Doctor ID if service requires doctor"
                            },
                            "specialist_id": {
                                "type": "integer",
                                "description": "Specialist ID if service requires specialist"
                            },
                            "device_id": {
                                "type": "integer",
                                "description": "Device ID if service requires device"
                            }
                        },
                        "required": ["service_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_booking",
                    "description": "Create a new booking after all details are confirmed. CRITICAL: Use 'start_time' in HH:MM format (e.g., '10:00'), get it from slot['time'] field.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "integer",
                                "description": "The patient's ID"
                            },
                            "service_id": {
                                "type": "integer",
                                "description": "The service ID to book"
                            },
                            "start_date": {
                                "type": "string",
                                "description": "Booking date (YYYY-MM-DD format)"
                            },
                            "start_time": {
                                "type": "string",
                                "description": "Booking time in HH:MM format (e.g., '10:00'). Get this from the slot['time'] field."
                            },
                            "doctor_id": {
                                "type": "integer",
                                "description": "Doctor ID if required"
                            },
                            "specialist_id": {
                                "type": "integer",
                                "description": "Specialist ID if required"
                            },
                            "device_id": {
                                "type": "integer",
                                "description": "Device ID if required"
                            }
                        },
                        "required": ["patient_id", "service_id", "start_date", "start_time"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_patient_bookings",
                    "description": "Get patient's existing bookings/appointments",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {
                                "type": "integer",
                                "description": "The patient's ID"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of bookings to retrieve",
                                "default": 10
                            }
                        },
                        "required": ["patient_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_doctors",
                    "description": "Search for available doctors",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_specialists",
                    "description": "Search for available specialists",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_devices",
                    "description": "Search for available medical devices/equipment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_patient",
                    "description": "Register a new patient in the system",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Patient's full name in Arabic"
                            },
                            "national_id": {
                                "type": "string",
                                "description": "Saudi national ID (10 digits)"
                            },
                            "phone": {
                                "type": "string",
                                "description": "Phone number"
                            },
                            "gender": {
                                "type": "string",
                                "description": "Gender (male/female)",
                                "enum": ["male", "female"]
                            }
                        },
                        "required": ["name", "national_id", "phone", "gender"]
                    }
                }
            }
        ]
    
    # ========================================================================
    # TOOL EXECUTION - Actually call the APIs
    # ========================================================================
    
    def _build_dynamic_context(self, patient_data: Optional[Dict], session_data: Optional[Dict], current_message: str) -> str:
        """Build dynamic context - LLM figures out topic from conversation, not rigid tracking"""
        context = []
        
        # Patient information
        if patient_data:
            full_name = patient_data.get('name', 'المريض')
            # 🚨 CRITICAL: Extract FIRST NAME ONLY (as instructed in system prompt)
            first_name = full_name.split()[0] if full_name and ' ' in full_name else full_name
            is_registered = patient_data.get('already_registered', False)
            context.append(f"👤 **المريض**: {first_name} {'(مسجل مسبوقاً)' if is_registered else '(جديد)'}")
        else:
            context.append("👤 **المريض**: غير مسجل - قد تحتاجين معلوماته للحجز")
        
        # Number selection hint (LLM reads conversation to understand)
        user_text = current_message.strip()
        if user_text.isdigit() and len(user_text) <= 2:
            context.append(f"💡 المريض قال '{current_message}' - قد يكون رقم من قائمة سابقة")
        
        # Conversation turn reference
        if session_data:
            turn = session_data.get('conversation_turn', 0)
            if turn > 1:
                context.append(f"📊 رقم الرسالة: {turn}")
            
            # 🚨 CRITICAL: Show saved booking context
            saved_service_id = session_data.get('selected_service_id')
            saved_doctor_id = session_data.get('selected_doctor_id')
            
            if saved_service_id or saved_doctor_id:
                context.append("\n" + "=" * 50)
                context.append("🚨 **CRITICAL - BOOKING IN PROGRESS:**")
                if saved_service_id:
                    context.append(f"   ✅ SERVICE ALREADY SELECTED: service_id={saved_service_id}")
                    context.append(f"   🔒 LOCKED: You MUST use service_id={saved_service_id}")
                
                if saved_doctor_id:
                    context.append(f"   ✅ DOCTOR ALREADY SELECTED: doctor_id={saved_doctor_id}")
                    context.append(f"   🔒 LOCKED: You MUST use doctor_id={saved_doctor_id}")
                elif saved_service_id and not saved_doctor_id:
                    # Service selected but no doctor - CRITICAL WARNING
                    context.append(f"\n🚨 **CRITICAL ERROR - NO DOCTOR SELECTED:**")
                    context.append(f"   ❌ User selected service but NEVER chose a doctor!")
                    context.append(f"   ❌ doctor_id=None → CANNOT call get_available_slots yet!")
                    context.append(f"   ✅ **REQUIRED ACTION:** Call search_doctors() to show doctor list")
                    context.append(f"   ✅ **THEN:** Wait for user to select a doctor number")
                    context.append(f"   ❌ **FORBIDDEN:** Guessing doctor_id, using doctor_id=1, or calling get_available_slots without doctor")
                
                # Build the exact tool call LLM should use
                if saved_service_id and saved_doctor_id:
                    context.append(f"\n✅ **EXACT TOOL CALL:** get_available_slots(service_id={saved_service_id}, doctor_id={saved_doctor_id}, date='...')")
                    context.append("❌ **FORBIDDEN:** Changing these IDs, searching again, or asking user to reselect")
                elif saved_service_id:
                    context.append(f"\n✅ **NEXT STEP:** Check if service requires doctor. If yes → search_doctors. If no → get_available_slots(service_id={saved_service_id}, date='...')")
                
                context.append("=" * 50)
        
        return "\n".join(context) if context else "لا توجد معلومات سياق خاصة"
    
    async def execute_tool(self, function_name: str, arguments: Dict[str, Any], detected_date: Optional[str] = None, selected_service_id: Optional[int] = None, selected_doctor_id: Optional[int] = None, patient_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the function that LLM requested.
        
        Args:
            function_name: Name of function to call
            arguments: Function arguments from LLM
            detected_date: Proactively detected date from user message
            selected_service_id: Previously selected service ID
            selected_doctor_id: Previously selected doctor ID
            patient_data: Patient data for gender filtering
            
        Returns:
            Function execution result
        """
        logger.info(f"🔧 Executing tool: {function_name} with args: {str(arguments)}")
        
        try:
            if function_name == "search_services":
                # Store search query for later use in get_service_variants
                if 'query' in arguments:
                    self.last_search_query = arguments['query']
                    logger.info(f"💾 Stored search query: '{self.last_search_query}'")
                return await self._search_services(**arguments)
            elif function_name == "get_service_details":
                return await self._get_service_details(**arguments)
            elif function_name == "get_all_services_offers":
                # Extract patient gender for filtering
                patient_gender = patient_data.get("gender") if patient_data else None
                return await self._get_all_services_offers(**arguments, patient_gender=patient_gender)
            elif function_name == "get_service_variants":
                # 🚨 Use stored search query for filtering variants
                # LLM is smart enough to call this with the right context
                last_search = getattr(self, 'last_search_query', None)
                current_msg = getattr(self, 'current_user_message', None)
                user_query = last_search or current_msg
                
                if last_search:
                    logger.info(f"🎯 Using stored search query for filtering: '{last_search}'")
                
                return await self._get_service_variants(**arguments, user_query=user_query)
            elif function_name == "get_available_slots":
                # 🚨 CRITICAL: Auto-inject detected date if available
                if detected_date and ('date' not in arguments or not arguments.get('date')):
                    logger.info(f"✅ AUTO-INJECTING detected date: {detected_date}")
                    arguments['date'] = detected_date
                elif 'date' not in arguments or not arguments.get('date'):
                    # Try to extract from current message or recent history as fallback
                    logger.warning("⚠️ LLM called get_available_slots without date - attempting auto-extraction")
                    current_msg = getattr(self, 'current_user_message', None)
                    history = getattr(self, 'conversation_history', [])
                    extracted_date = extract_date_from_context(current_msg, history)
                    
                    if extracted_date:
                        logger.info(f"✅ Auto-extracted date: {extracted_date} from conversation")
                        arguments['date'] = extracted_date
                    else:
                        logger.error("❌ Could not extract date from conversation")
                
                # 🚨 CRITICAL: Auto-inject selected_service_id if available and not provided
                if selected_service_id and ('service_id' not in arguments or not arguments.get('service_id')):
                    logger.info(f"✅ AUTO-INJECTING selected service_id: {selected_service_id}")
                    arguments['service_id'] = selected_service_id
                
                # 🚨 CRITICAL: Auto-inject selected_doctor_id if available and not provided
                if selected_doctor_id and ('doctor_id' not in arguments or not arguments.get('doctor_id')):
                    logger.info(f"✅ AUTO-INJECTING selected doctor_id: {selected_doctor_id}")
                    arguments['doctor_id'] = selected_doctor_id
                
                # 🚨 CRITICAL: Block invalid doctor_id if user never selected one
                llm_doctor_id = arguments.get('doctor_id')
                if llm_doctor_id and not selected_doctor_id:
                    logger.error(f"🚨 BLOCKING INVALID doctor_id={llm_doctor_id} - User never selected a doctor!")
                    return {
                        "success": False,
                        "error": "doctor_required_but_not_selected",
                        "message": "User has not selected a doctor yet. Call search_doctors() first.",
                        "instruction": "You MUST call search_doctors() to show doctor list, then wait for user to select."
                    }
                
                return await self._get_available_slots(**arguments)
            elif function_name == "create_booking":
                return await self._create_booking(**arguments)
            elif function_name == "get_patient_bookings":
                return await self._get_patient_bookings(**arguments)
            elif function_name == "search_doctors":
                return await self._search_doctors(**arguments)
            elif function_name == "search_specialists":
                return await self._search_specialists(**arguments)
            elif function_name == "search_devices":
                return await self._search_devices(**arguments)
            elif function_name == "create_patient":
                return await self._create_patient(**arguments)
            else:
                logger.error(f"❌ Unknown function: {function_name}")
                return {"error": f"Unknown function: {function_name}"}
        
        except Exception as e:
            logger.error(f"❌ Tool execution error ({function_name}): {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "error_details": f"Tool '{function_name}' failed with error: {str(e)}. This is likely a system error, not a user error."
            }
    
    # ========================================================================
    # INTERNAL API METHODS - Following test_complete_no_skip.py patterns
    # ========================================================================
    
    async def _search_services(self, query: str, limit: int = 20) -> Dict[str, Any]:
        """Search services - INTELLIGENT filtering with test service removal"""
        try:
            # Search by name or get all services
            result = await self.api_client.get("/services", params={"limit": limit})
            services = result.get("results") or result.get("data") or []
            
            logger.info(f"🔍 Search services: query='{query}', total_services={len(services)}")
            
            # CRITICAL: Filter out test/dummy services
            test_keywords = ["test", "zzzz", "dummy", "sample", "xxxx", "temp"]
            real_services = [
                s for s in services
                if not any(
                    keyword in (s.get("name_ar") or "").lower() or
                    keyword in (s.get("name") or "").lower() or
                    keyword in (s.get("nameAr") or "").lower() or
                    keyword in (s.get("nameEn") or "").lower()
                    for keyword in test_keywords
                )
            ]
            
            logger.info(f"🧹 Filtered out {len(services) - len(real_services)} test services")
            
            query_lower = query.lower().strip()
            
            # If query is very generic, return ALL services
            generic_terms = ["خدمة", "علاج", "مركز", "عندكم", "موجود", "service"]
            if any(term in query_lower for term in generic_terms) and len(query_lower.split()) <= 2:
                logger.info(f"🌐 Generic query detected: '{query}' - returning ALL services")
                return {
                    "success": True,
                    "services": real_services[:limit],
                    "total": len(real_services)
                }
            
            # Simple matching - no hardcoded logic, let LLM decide what to do with results
            filtered = [
                s for s in real_services
                if query_lower in (s.get("name_ar") or "").lower()
                or query_lower in (s.get("name") or "").lower()
                or query_lower in (s.get("nameAr") or "").lower()
                or query_lower in (s.get("nameEn") or "").lower()
            ]
            
            # If no exact match, try partial matching
            if not filtered and len(query_lower) > 2:
                logger.info(f"⚠️ No exact match for '{query}' - trying partial matching")
                query_words = query_lower.split()
                for service in real_services:
                    service_name = (
                        (service.get("name_ar") or "") + " " +
                        (service.get("name") or "") + " " +
                        (service.get("nameAr") or "") + " " +
                        (service.get("nameEn") or "")
                    ).lower()
                    
                    # Match if ANY word matches
                    if any(word in service_name for word in query_words if len(word) > 2):
                        filtered.append(service)
            
            # 🚀 DEEP SEARCH: If still no match, search in service VARIANTS (subservices)
            # This handles cases where user asks for specific service that's nested under categories
            deep_search_explanation = None  # Initialize for later use
            
            if not filtered and len(query_lower) > 3:
                logger.info(f"🔍 No category match - trying DEEP SEARCH in variants for '{query}'")
                
                # Search through variants of ALL categories
                matched_categories = []
                for category in real_services:
                    try:
                        category_id = category.get("service_type_id") or category.get("id")
                        if not category_id:
                            continue
                        
                        # Get variants for this category
                        variants_result = await self.api_client.get("/services/", params={"service_type_id": category_id, "limit": 50})
                        variants = variants_result.get("results") or variants_result.get("data") or []
                        
                        # Check if any variant matches the query
                        for variant in variants:
                            variant_name = (
                                (variant.get("name") or "") + " " +
                                (variant.get("name_ar") or "") + " " +
                                (variant.get("nameAr") or "")
                            ).lower()
                            
                            # If variant matches, add its parent category
                            if query_lower in variant_name or any(word in variant_name for word in query_words if len(word) > 3):
                                logger.info(f"  ✅ Found variant match: '{variant.get('name')}' in category '{category.get('service_type_name')}'")
                                if category not in matched_categories:
                                    matched_categories.append(category)
                                break  # Found match in this category, move to next
                    
                    except Exception as e:
                        # Don't let one category failure break the whole search
                        logger.debug(f"  ⚠️ Could not search variants for category {category.get('service_type_id')}: {e}")
                        continue
                
                if matched_categories:
                    logger.info(f"🎯 Deep search found {len(matched_categories)} matching categories")
                    filtered = matched_categories
                    # Add explanation for LLM about why these categories were found
                    deep_search_explanation = f"""
🎯 **DEEP SEARCH RESULTS**: Found '{query}' as a SUBSERVICE inside these categories!

The service '{query}' exists but it's nested under these categories.
You MUST call get_service_variants for each category to show the actual '{query}' services to the user.

🚨 DON'T say "we don't have {query}" - WE DO HAVE IT! It's just nested inside these categories!

✅ CORRECT FLOW:
1. Call get_service_variants for the returned category (pass the user's original query!)
2. System will AUTO-FILTER to show ONLY services matching '{query}'
3. Show those filtered variants to the user (they're already filtered, no need to filter again!)

Example: User asked for "خيوط استقامة", we found it inside "بوتوكس" category.
When you call get_service_variants(بوتوكس), the system will AUTOMATICALLY show only "خيوط" services, not all botox services!

🚨 IMPORTANT: The variants you receive are ALREADY FILTERED to match '{query}' - just show them all!
"""
                else:
                    logger.info(f"⚠️ Deep search found no matches either")
            
            # If still nothing, check if it's a GENERIC query (like "خدمات", "services", "وش عندكم")
            generic_queries = ["خدمات", "خدمة", "services", "service", "وش عندكم", "ايش عندكم", "شو عندكم"]
            is_generic = any(gq in query_lower for gq in generic_queries)
            
            if not filtered:
                if is_generic and len(real_services) > 5:
                    # User asked generically - ask clarifying question instead of dumping all services
                    logger.info(f"⚠️ Generic query '{query}' detected - instructing LLM to ask clarifying questions")
                    return {
                        "success": True,
                        "services": real_services[:limit],  # Return limited list as context
                        "is_generic_query": True,
                        "_instruction": f"""🚨 CRITICAL: User asked a GENERIC question: "{query}"

DON'T dump all {len(real_services)} services as a numbered list! That's overwhelming!

INSTEAD:
1. Ask an open-ended clarifying question in friendly Saudi dialect
2. Examples:
   - "تمام! عندنا خدمات متنوعة. وش اللي يهمك؟ بوتوكس، ليزر، فيلر، ولا شي ثاني؟"
   - "أكيد! عندنا أشياء كثير. تبي شي للبشرة؟ ليزر؟ ولا حقن؟"
   - "عندنا خدمات حلوة! تبي تجمل؟ علاج؟ ولا وش بالضبط؟"

3. After user clarifies (e.g., "ليزر"), THEN call search_services("ليزر") to get specific categories
4. Be consultative, not a robot menu!

❌ DO NOT add [SELECTION_MAP] tags - this is NOT a numbered list!
❌ DO NOT show numbered list - ask clarifying question only!

Available service types for your reference: {', '.join([s.get('service_type_name', 'Unknown') for s in real_services[:10]])}"""
                    }
                else:
                    # Not generic or few services - return all
                    logger.info(f"⚠️ No match for '{query}' - returning ALL services, LLM will decide what to do")
                    filtered = real_services
            
            logger.info(f"✅ Found {len(filtered)} services for query '{query}' - LLM will decide how to respond")
            
            result = {
                "success": True,
                "services": filtered[:limit],
                "total": len(filtered)
            }
            
            # Add deep search explanation if applicable
            if deep_search_explanation:
                result["_instruction"] = deep_search_explanation
            
            return result
        except Exception as e:
            logger.error(f"❌ Search services error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _get_service_details(self, service_id: int) -> Dict[str, Any]:
        """Get service details"""
        try:
            service = await self.api_client.get(f"/services/{service_id}")
            return {"success": True, "service": service}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_all_services_offers(self, filter_query: str = "", show_categories_only: bool = False, patient_gender: str = None) -> Dict[str, Any]:
        """
        Get ALL SUBSERVICES (actual bookable services) with prices, filtered by patient gender.
        Called when user asks: "What are your offers?", "Show me all services", "عندكم وش؟"
        
        Args:
            filter_query: Optional filter (e.g., 'ليزر', 'بوتوكس')
            show_categories_only: If True, show only main categories (not recommended)
            patient_gender: Patient's gender for filtering (male/female)
        
        Returns:
            {
                "success": True,
                "services": [...],  # List of SUBSERVICES with prices
                "total": count,
                "message": "Here are all our services with prices"
            }
        """
        try:
            logger.info(f"🎁 Getting ALL subservices (filter='{filter_query}', gender={patient_gender})")
            
            # Step 1: Get all CATEGORIES first
            params = {"limit": 100}
            if filter_query:
                params["q"] = filter_query
            
            categories_result = await self.api_client.get("/services/", params=params)
            categories = categories_result.get("results") or categories_result.get("data") or []
            
            # Filter out test categories
            categories = [cat for cat in categories 
                         if cat.get("name", "").lower() not in ["test", "zzzz", "zzz test"]]
            
            logger.info(f"📋 Found {len(categories)} categories to fetch subservices from")
            
            # Step 2: For EACH category, fetch its SUBSERVICES
            all_subservices = []
            for category in categories:
                try:
                    service_type_id = category.get("service_type_id") or category.get("id")
                    if not service_type_id:
                        continue
                    
                    # Fetch subservices for this category
                    subservices_result = await self.api_client.get("/services/", params={"service_type_id": service_type_id, "limit": 50})
                    subservices = subservices_result.get("results") or subservices_result.get("data") or []
                    
                    all_subservices.extend(subservices)
                    logger.debug(f"  ✅ Category '{category.get('name')}': {len(subservices)} subservices")
                    
                except Exception as cat_error:
                    logger.warning(f"  ⚠️ Failed to fetch subservices for category {service_type_id}: {cat_error}")
                    continue
            
            logger.info(f"🎁 Retrieved {len(all_subservices)} total subservices")
            
            # Step 3: Filter by gender and format with prices
            formatted_services = []
            for svc in all_subservices:
                # Skip test services
                service_name = svc.get("name", "Unknown Service")
                if any(keyword in service_name.lower() for keyword in ["test", "zzzz", "dummy"]):
                    continue
                
                # Get service details
                service_id = svc.get("id")
                service_price = svc.get("price") or svc.get("الأسعار")
                service_gender = svc.get("gender", "both")
                
                # Extract price from name if not in API
                if not service_price or service_price == 0:
                    price_match = re.search(r'\s+(\d{2,5})\s*(?:ريال)?$', service_name)
                    if price_match:
                        service_price = int(price_match.group(1))
                
                # 🚨 CRITICAL: Filter by patient gender
                if patient_gender:
                    # service_gender can be: 'male', 'female', 'both', 'unisex', None
                    gender_lower = str(service_gender).lower()
                    patient_gender_lower = patient_gender.lower()
                    
                    # Skip if service is gender-specific and doesn't match patient
                    if gender_lower in ['male', 'female']:
                        if gender_lower != patient_gender_lower:
                            logger.debug(f"  ⏭️ Skipping {service_name} (gender: {service_gender}, patient: {patient_gender})")
                            continue
                
                # Skip services without prices
                if not service_price or service_price == 0:
                    logger.debug(f"  ⏭️ Skipping {service_name} (no price)")
                    continue
                
                formatted_services.append({
                    "id": service_id,
                    "name": service_name,
                    "price": service_price,
                    "gender": service_gender
                })
            
            logger.info(f"🎁 Final list: {len(formatted_services)} subservices (after gender filter)")
            
            return {
                "success": True,
                "services": formatted_services,
                "total": len(formatted_services),
                "message": f"Here are all our services with prices (filtered by gender: {patient_gender or 'all'}). Present them in a friendly, organized way."
            }
            
        except Exception as e:
            logger.error(f"❌ Get all services error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _get_service_variants(self, service_type_id: int, user_query: str = None) -> Dict[str, Any]:
        """Get service variants with semantic validation"""
        try:
            result = await self.api_client.get("/services/", params={"service_type_id": service_type_id})
            variants = result.get("results") or result.get("data") or []
            
            # CRITICAL: Log what we're retrieving for quality control
            if variants:
                logger.info(f"🔍 Retrieved {len(variants)} variant(s) for service_type_id={service_type_id}")
                if user_query:
                    logger.info(f"🔍 User query was: '{user_query}'")
                    for v in variants[:3]:  # Log first 3
                        logger.info(f"   - {v.get('name', 'Unknown')}")
            
            # 🚨 CRITICAL: Filter variants based on user query if provided
            # This prevents showing ALL services when user asked for specific one
            is_number_selection = user_query and user_query.strip().isdigit()
            
            if user_query and variants and not is_number_selection:
                user_query_lower = user_query.lower().strip()
                key_terms = [term for term in user_query_lower.split() if len(term) > 2]
                
                # Filter variants that match the user's query
                filtered_variants = []
                for variant in variants:
                    variant_name_lower = variant.get("name", "").lower()
                    
                    # Count how many key terms match
                    matching_terms = sum(1 for term in key_terms if term in variant_name_lower)
                    
                    # 🚨 CRITICAL: Require MULTIPLE terms to match for specific queries
                    # If user said multiple words (e.g., "خيوط استقامة الأنف"), require at least 2 matches
                    # This prevents showing ALL thread services when user asked for specific thread type
                    if len(key_terms) >= 2:
                        # Specific query: need at least 2 terms OR exact match of critical term
                        # Example: "خيوط استقامة الأنف" needs "استقامة" or "الأنف" in addition to "خيوط"
                        if matching_terms >= 2:
                            filtered_variants.append(variant)
                    else:
                        # Generic query (1 word): any match is fine
                        if matching_terms >= 1:
                            filtered_variants.append(variant)
                
                # Only apply filter if we found matches (don't filter to empty list)
                if filtered_variants:
                    logger.info(f"🎯 Filtered from {len(variants)} to {len(filtered_variants)} variants matching '{user_query}'")
                    for fv in filtered_variants:
                        logger.info(f"   ✅ {fv.get('name', 'Unknown')}")
                    variants = filtered_variants
                else:
                    logger.warning(f"⚠️ No variants matched '{user_query}', showing all {len(variants)} variants")
            
            # Generate SELECTION_MAP so LLM can map user's number selection to service_id
            selection_map_lines = []
            for i, variant in enumerate(variants):
                service_id = variant.get("id")
                service_name = variant.get("name", "Unknown")
                price = variant.get("price", 0)
                
                # 🚨 CRITICAL: Include resource requirements so LLM knows what to pass to get_available_slots
                requires_doctor = variant.get("requires_doctor", False) or variant.get("doctor_id")
                requires_specialist = variant.get("requires_specialist", False) or variant.get("specialist_id")
                requires_device = variant.get("requires_device", False) or variant.get("device_id")
                
                # Note: Verbose technical logging removed - focus on conversation, not implementation
                
                # Determine resource type
                resource_info = ""
                if requires_doctor:
                    resource_info = " [REQUIRES: doctor_id]"
                elif requires_specialist:
                    resource_info = " [REQUIRES: specialist_id - AUTO]"
                elif requires_device:
                    resource_info = " [REQUIRES: device_id - AUTO]"
                else:
                    resource_info = " [NO RESOURCE NEEDED]"
                
                selection_map_lines.append(f"Number {i+1} = service_id:{service_id} (name: {service_name}, price: {price} SAR){resource_info}")
            
            selection_map = "\n".join(selection_map_lines)
            logger.warning(f"📋 CREATED SELECTION_MAP for get_service_variants:\n{selection_map}")
            
            # Add conversational context - be proactive and helpful
            prices = [v.get("price", 0) for v in variants if v.get("price")]
            conversational_context = ""
            
            if prices:
                min_price = min(prices)
                max_price = max(prices)
                if min_price != max_price:
                    conversational_context += f"\nPrice range: {min_price:,.0f} - {max_price:,.0f} SAR"
                
                # Find most common price point (might indicate popular option)
                from collections import Counter
                price_counts = Counter(prices)
                if len(price_counts) > 1:
                    most_common_price = price_counts.most_common(1)[0][0]
                    conversational_context += f"\nNote: Multiple options available at different price points"
            
            # CRITICAL SEMANTIC VALIDATION: Already done during filtering above
            # If we reach here after filtering, variants already match user query
            # Only log warning if somehow we have NO variants (shouldn't happen due to filter logic)
            semantic_mismatch_warning = ""
            
            if user_query and not variants and not is_number_selection:
                logger.error("🚨🚨🚨 NO VARIANTS FOUND AFTER FILTERING!")
                logger.error(f"   User asked for: '{user_query}'")
                semantic_mismatch_warning = f"\n\n⚠️ WARNING: No services found matching '{user_query}'. This might be a database issue."
            
            # Add filtering notice if user_query was used
            filtering_notice = ""
            if user_query and not is_number_selection and variants:
                filtering_notice = f"\n\n🎯 **AUTO-FILTERED**: These {len(variants)} services were filtered from all variants to match your search '{user_query}'. Show them all!"
            
            # 🚨 CRITICAL: If only 1 variant, auto-select it
            single_variant_instruction = ""
            auto_selected_service_id = None
            if len(variants) == 1:
                single_variant = variants[0]
                auto_selected_service_id = single_variant.get('id')
                logger.info(f"✅ AUTO-SELECTING single variant: service_id={auto_selected_service_id}, name={single_variant.get('name')}")
                
                single_variant_instruction = f"""

🎯 **ONLY 1 SERVICE FOUND - AUTO-SELECTION:**
Since there's only ONE service matching the query, you should:
1. Mention the service details (name, price, requirements)
2. If it requires a doctor → Call search_doctors immediately to show available doctors
3. If it requires specialist/device → Ask user for date and call get_available_slots
4. User does NOT need to select from a list of 1 item!

Service details:
- ID: {single_variant.get('id')}
- Name: {single_variant.get('name')}
- Price: {single_variant.get('price', 0)} SAR
- Requires: {'Doctor' if single_variant.get('requires_doctor') else 'Specialist/Device/Nothing'}

✅ AUTO-PROCEED: Don't wait for user to "select" - there's only 1 option!
"""
            
            result = {
                "success": True, 
                "variants": variants, 
                "total": len(variants),
                "selection_map": selection_map,
                "conversational_context": conversational_context.strip(),
                "semantic_warning": semantic_mismatch_warning,
                "_instruction": f"""SELECTION MAP for user's number choice:

{selection_map}

{conversational_context}{filtering_notice}

When showing these to the user, maintain exact order above. Be conversational and helpful - mention price range if available.
❌ DO NOT sort by price (lowest/highest)!
❌ DO NOT reorder by name alphabetically!
❌ DO NOT skip any service!
✅ MUST show in EXACT order above: Number 1 first, then 2, then 3, etc.
✅ This ensures user's number selection maps correctly!

If you reorder services, user will book the WRONG service!

⚠️ DO NOT use the 'price' field as service_id!
⚠️ DO NOT count manually in the variants array!
⚠️ ONLY use the SELECTION_MAP above!

🎯 **RESOURCE HANDLING - READ THIS CAREFULLY!**

When user selects a service number, CHECK the resource requirement on that EXACT line in selection_map:

**[REQUIRES: device_id - AUTO]** (e.g., Laser services):
✅ DO: Call get_available_slots(service_id=X) directly - system auto-selects device
❌ DON'T: Call search_doctors - LASER DOESN'T NEED DOCTORS!
❌ DON'T: Call search_devices - it's automatic!

**[REQUIRES: specialist_id - AUTO]** (e.g., some beauty services like peeling):
✅ DO: Call get_available_slots(service_id=X) directly - system auto-selects specialist
❌ DON'T: Call search_doctors! ← SPECIALIST ≠ DOCTOR! They are different!
❌ DON'T: Call search_specialists - it's automatic!

**[REQUIRES: doctor_id]** (e.g., Botox, medical consultations):
❌ DON'T: Call get_available_slots immediately - it will fail!
✅ DO: Call search_doctors first
✅ DO: Show doctors list to user with numbers
✅ DO: Wait for user to select a doctor number
✅ DO: Then call get_available_slots(service_id=X, doctor_id=Y)

**[NO RESOURCE NEEDED]** (e.g., some basic services):
✅ DO: Call get_available_slots(service_id=X) directly

🚨 **CRITICAL EXAMPLES:**

Example 1 - Laser (DEVICE):
User: "3"
Selection_map: "Number 3 = service_id:120 (ليزر منطقة صغيرة 100 SAR) [REQUIRES: device_id - AUTO]"
✅ CORRECT: get_available_slots(service_id=120) ← Device auto-selected!
❌ WRONG: search_doctors() ← NO! Laser uses DEVICE not DOCTOR!

Example 2 - Peeling (SPECIALIST):
User: "6"
Selection_map: "Number 6 = service_id:84 (تقشير بارد 600 SAR) [REQUIRES: specialist_id - AUTO]"
✅ CORRECT: get_available_slots(service_id=84) ← Specialist auto-selected!
❌ WRONG: search_doctors() ← NO! Peeling uses SPECIALIST not DOCTOR!
❌ WRONG: search_specialists() ← NO! It's automatic, don't call this!

Example 3 - Botox (DOCTOR):
:User: "2"
Selection_map: "Number 2 = service_id:98 (بوتوكس وجه كامل 500 SAR) [REQUIRES: doctor_id]"
❌ WRONG: get_available_slots(service_id=98) ← Will fail! No doctor_id!
✅ CORRECT: search_doctors() first → Show list → Wait → get_available_slots(service_id=98, doctor_id=5)

{single_variant_instruction}"""
            }
            
            # Add auto-selected service_id if only 1 variant
            if auto_selected_service_id:
                result["auto_selected_service_id"] = auto_selected_service_id
            
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_available_slots(
        self, 
        service_id: int,
        date: Optional[str] = None,
        patient_id: Optional[int] = None,
        doctor_id: Optional[int] = None,
        specialist_id: Optional[int] = None,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get available time slots - PROACTIVE resource detection
        
        Example: GET /slots?service_id=127&date=2025-11-27&patient_id=185&doctor_id=3
        """
        try:
            # 🚨 CRITICAL: Date is MANDATORY - LLM must ask user first!
            if not date:
                logger.error("🚨 CRITICAL: LLM called get_available_slots WITHOUT date!")
                return {
                    "success": False,
                    "error": "Date is required! You MUST ask the user which date they prefer (e.g., 'متى تبي الموعد؟ بكرة؟ بعد بكرة؟'). Never assume or generate dates automatically!"
                }
            
            # Validate date format and ensure it's not in the past or too far future
            try:
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                today = datetime.now().date()
                date_only = date_obj.date()
                
                if date_only < today:
                    logger.error(f"🚨 CRITICAL: LLM trying to book in the PAST! Date: {date}")
                    return {
                        "success": False,
                        "error": f"Invalid date '{date}' - this is in the past! Current date is {today}. Ask user for a future date."
                    }
                
                # Warn if date is more than 3 months in future (unusual)
                if (date_only - today).days > 90:
                    logger.warning(f"⚠️ Unusual: Booking date is {(date_only - today).days} days in future: {date}")
            
            except ValueError:
                logger.error(f"🚨 CRITICAL: Invalid date format: {date}")
                return {
                    "success": False,
                    "error": f"Invalid date format '{date}'. Must be YYYY-MM-DD (e.g., '2025-10-29'). Ask user for a proper date."
                }
            
            # 🚨 PROACTIVE: Check if we need resources BEFORE calling API
            if not doctor_id and not specialist_id and not device_id:
                logger.info(f"🔍 No resource provided - checking service requirements for service_id={service_id}")
                try:
                    service_details = await self.api_client.get_service(service_id, timeout=10.0)
                    
                    # Check what resource this service needs
                    requires_doctor = service_details.get("doctor_id") or service_details.get("requires_doctor")
                    requires_specialist = service_details.get("specialist_id") or service_details.get("requires_specialist")
                    requires_device = service_details.get("device_id") or service_details.get("requires_device")
                    
                    # Auto-handle based on type
                    if requires_specialist:
                        logger.info(f"✅ Service requires specialist - auto-selecting")
                        specialists_result = await self.api_client.get_specialists(limit=1)
                        specialists = specialists_result.get("results", []) or specialists_result.get("data", [])
                        if specialists and len(specialists) > 0:
                            specialist_id = specialists[0].get("id")
                            logger.info(f"✅ Auto-selected specialist_id={specialist_id}")
                        else:
                            return {"success": False, "error": "No specialists available"}
                    
                    elif requires_device:
                        logger.info(f"✅ Service requires device - smart auto-selecting")
                        service_name = service_details.get("name", "").lower()
                        devices_result = await self.api_client.get_devices(limit=50)
                        devices = devices_result.get("results", []) or devices_result.get("data", [])
                        
                        if not devices or len(devices) == 0:
                            logger.error("❌ No devices available in system")
                            return {"success": False, "error": "No devices available"}
                        
                        logger.info(f"🔍 Checking {len(devices)} devices for match with service '{service_name}'")
                        
                        # Smart device matching with token-based fuzzy logic
                        selected_device = None
                        best_match_score = 0
                        
                        for device in devices:
                            device_name_original = device.get("name", "")
                            device_name = device_name_original.lower()
                            
                            # Strategy 1: Exact substring match (fastest)
                            if device_name and device_name in service_name:
                                selected_device = device
                                logger.info(f"✅ EXACT match: '{device_name_original}' found in service name")
                                break
                            
                            # Strategy 2: Token-based matching (handles variations like "ديكا - جنتل برو" vs "ديكا / Deka")
                            # Split device name into meaningful tokens (ignore separators and English)
                            device_tokens = [
                                token.strip() 
                                for token in re.split(r'[/\-,،\s]+', device_name) 
                                if token.strip() and len(token.strip()) > 2 and not token.strip().isascii()
                            ]
                            
                            # Count how many device tokens appear in service name
                            matches = sum(1 for token in device_tokens if token in service_name)
                            match_score = matches / len(device_tokens) if device_tokens else 0
                            
                            if match_score > best_match_score:
                                best_match_score = match_score
                                selected_device = device
                                logger.info(f"🔍 TOKEN match: '{device_name_original}' score={match_score:.2f} ({matches}/{len(device_tokens)} tokens)")
                        
                        if selected_device and best_match_score > 0:
                            logger.info(f"✅ AUTO-SELECTED device: {selected_device.get('name')} (score={best_match_score:.2f})")
                        elif selected_device:
                            logger.info(f"✅ AUTO-SELECTED device: {selected_device.get('name')} (exact match)")
                        else:
                            # No match at all - use first
                            selected_device = devices[0]
                            logger.warning(f"⚠️ No device match - using first available: {selected_device.get('name')}")
                        
                        device_id = selected_device.get("id")
                        logger.warning(f"✅ Auto-selected device_id={device_id} ({selected_device.get('name')})")
                    
                    elif requires_doctor:
                        logger.info(f"⚠️ Service requires doctor - user must choose")
                        return {
                            "success": False,
                            "error": "This service requires a doctor. You MUST: 1) Call search_doctors to get list, 2) SHOW doctors to user, 3) WAIT for user selection, 4) Retry get_available_slots with doctor_id.",
                            "required_resource": "doctor_id",
                            "next_action": "search_doctors"
                        }
                
                except Exception as check_error:
                    logger.warning(f"⚠️ Could not check service requirements: {check_error}")
                    # Continue anyway - API will tell us if something's missing
            
            # Build params
            params = {"service_id": service_id, "date": date}
            
            # Add patient_id if provided (optional but recommended)
            if patient_id:
                params["patient_id"] = patient_id
            
            # Add resource if provided
            if doctor_id:
                params["doctor_id"] = doctor_id
            elif specialist_id:
                params["specialist_id"] = specialist_id
            elif device_id:
                params["device_id"] = device_id
            
            logger.info(f"🔍 Getting slots: {str(params)}")
            
            try:
                # TRY 1: Call API with current params
                result = await self.api_client.get_available_slots(**params, timeout=120.0)
                # CRITICAL: API returns "slots" key (not "results" or "data")
                slots = result.get("slots") or result.get("results") or result.get("data") or []
                
                logger.info(f"✅ Got {len(slots)} slots")
                
                return {
                    "success": True,
                    "slots": slots,
                    "total": len(slots),
                    "date": result.get("date", date)  # Use date from response if available
                }
            
            except Exception as api_error:
                error_msg = str(api_error).lower()
                
                logger.warning(f"⚠️ Slots API error: {error_msg[:200]}")
                
                # ERROR-DRIVEN: API tells us what's missing!
                # CRITICAL: Check for "requires X_id" patterns (most specific first)
                # Use "requires" keyword to avoid false matches from parameter names in URLs
                
                # Check what resource is REQUIRED (not what was passed wrong)
                requires_device = "requires a device_id" in error_msg or "requires device_id" in error_msg
                requires_specialist = "requires a specialist_id" in error_msg or "requires specialist_id" in error_msg  
                requires_doctor = "requires a doctor_id" in error_msg or "requires doctor_id" in error_msg
                
                if requires_doctor:
                    # DON'T auto-select! Let LLM handle it so user can choose
                    logger.info(f"🔧 Service requires doctor - LLM must call search_doctors")
                    return {
                        "success": False,
                        "error": "This service requires a doctor. Call search_doctors to show doctor options to user.",
                        "required_resource": "doctor_id",
                        "next_action": "search_doctors"
                    }
                
                elif requires_specialist:
                    # AUTO-SELECT: Specialists are interchangeable, pick first available
                    logger.warning(f"🔧 REACTIVE: Service requires specialist - auto-selecting first available")
                    try:
                        specialists_result = await self.api_client.get_specialists(limit=1)
                        specialists = specialists_result.get("results", []) or specialists_result.get("data", [])
                        if specialists and len(specialists) > 0:
                            specialist_id = specialists[0].get("id")
                            logger.warning(f"✅ REACTIVE: Auto-selected specialist_id={specialist_id}")
                            # Retry with specialist_id
                            params["specialist_id"] = specialist_id
                            result = await self.api_client.get_available_slots(**params, timeout=120.0)
                            slots = result.get("slots") or result.get("results") or result.get("data") or []
                            return {
                                "success": True,
                                "slots": slots,
                                "total": len(slots),
                                "date": result.get("date", date),
                                "auto_selected_specialist": specialist_id
                            }
                        else:
                            logger.error("❌ REACTIVE: No specialists available in system")
                            return {"success": False, "error": "No specialists available"}
                    except Exception as specialist_error:
                        logger.error(f"❌ REACTIVE: Failed to auto-select specialist: {specialist_error}")
                        return {"success": False, "error": f"Failed to get specialist: {specialist_error}"}
                
                elif requires_device:
                    # AUTO-SELECT: Try to match device from service name, or pick first available
                    logger.warning(f"🔧 REACTIVE: Service requires device - attempting smart auto-selection")
                    try:
                        # Get service details to check service name
                        service_details = await self.api_client.get_service(service_id, timeout=10.0)
                        service_name = service_details.get("name", "").lower() if service_details else ""
                        
                        # Get all devices
                        devices_result = await self.api_client.get_devices(limit=50)
                        devices = devices_result.get("results", []) or devices_result.get("data", [])
                        
                        if not devices or len(devices) == 0:
                            logger.error("❌ REACTIVE: No devices available in system")
                            return {"success": False, "error": "No devices available"}
                        
                        logger.info(f"🔍 REACTIVE: Checking {len(devices)} devices for match with service '{service_name}'")
                        
                        # Try to find device mentioned in service name
                        selected_device = None
                        for device in devices:
                            device_name = device.get("name", "").lower()
                            if device_name and device_name in service_name:
                                selected_device = device
                                logger.warning(f"✅ REACTIVE: Matched device '{device_name}' in service name")
                                break
                        
                        # If no match, pick first available device
                        if not selected_device:
                            selected_device = devices[0]
                            logger.warning(f"⚠️ REACTIVE: No match - using first device: {selected_device.get('name')}")
                        
                        device_id = selected_device.get("id")
                        logger.warning(f"✅ REACTIVE: Auto-selected device_id={device_id} ({selected_device.get('name')})")
                        
                        # Retry with device_id
                        params["device_id"] = device_id
                        result = await self.api_client.get_available_slots(**params, timeout=120.0)
                        slots = result.get("slots") or result.get("results") or result.get("data") or []
                        return {
                            "success": True,
                            "slots": slots,
                            "total": len(slots),
                            "date": result.get("date", date),
                            "auto_selected_device": device_id,
                            "device_name": selected_device.get("name")
                        }
                    except Exception as device_error:
                        logger.error(f"❌ Failed to auto-select device: {device_error}")
                        return {"success": False, "error": f"Failed to get device: {device_error}"}
                
                else:
                    # Unknown error
                    return {"success": False, "error": str(api_error)}
        
        except Exception as e:
            logger.error(f"❌ Get slots error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    async def _create_booking(
        self,
        patient_id: int,
        service_id: int,
        start_date: str,
        start_time: str,
        doctor_id: Optional[int] = None,
        specialist_id: Optional[int] = None,
        device_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create booking - CRITICAL: API uses start_time in HH:MM format!
        
        Args:
            start_time: Time in HH:MM format (e.g., "10:00") - get from slot["time"]
        """
        try:
            # 🚨 CRITICAL: Validate date before booking!
            try:
                date_obj = datetime.strptime(start_date, "%Y-%m-%d")
                today = datetime.now().date()
                date_only = date_obj.date()
                
                if date_only < today:
                    logger.error(f"🚨 CRITICAL: Trying to book in the PAST! Date: {start_date}, Today: {today}")
                    return {
                        "success": False,
                        "error": f"Cannot book in the past! You provided '{start_date}' but today is {today}. This date was probably hallucinated. Use the EXACT date from get_available_slots response!"
                    }
                
                # Check if date is from years ago (common hallucination)
                year_diff = today.year - date_only.year
                if year_diff >= 1:
                    logger.error(f"🚨 CRITICAL: Date is from {year_diff} years ago! Date: {start_date}")
                    return {
                        "success": False,
                        "error": f"HALLUCINATION DETECTED: You're trying to book for {start_date} which is {year_diff} years ago! Use the CURRENT year {today.year} and the date user selected from slots!"
                    }
            
            except ValueError:
                logger.error(f"🚨 CRITICAL: Invalid date format in booking: {start_date}")
                return {
                    "success": False,
                    "error": f"Invalid date format '{start_date}'. Must be YYYY-MM-DD. Use the EXACT date from get_available_slots!"
                }
            # Get service details for duration_minutes
            service = await self.api_client.get(f"/services/{service_id}")
            duration_minutes = service.get("duration_minutes", 60)
            
            # Build booking data - CRITICAL: Uses start_time now, NOT slot_choice_id!
            booking_data = {
                "patient_id": patient_id,
                "service_id": service_id,
                "start_date": start_date,
                "start_time": start_time,
                "duration_minutes": duration_minutes
            }
            
            # Add ONLY ONE resource ID (priority: doctor > specialist > device)
            resource_added = False
            if doctor_id and not resource_added:
                booking_data["doctor_id"] = doctor_id
                resource_added = True
            if specialist_id and not resource_added:
                booking_data["specialist_id"] = specialist_id
                resource_added = True
            if device_id and not resource_added:
                booking_data["device_id"] = device_id
                resource_added = True
            
            logger.info(f"📋 Creating booking: {str(booking_data)}")
            
            # CRITICAL: Unpack dict as kwargs, not pass as single arg
            result = await self.api_client.create_booking(**booking_data)
            
            return {
                "success": True,
                "booking": result,
                "booking_id": result.get("id"),
                "confirmation_code": result.get("confirmation_code") or result.get("session_name")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _get_patient_bookings(self, patient_id: int, limit: int = 10) -> Dict[str, Any]:
        """Get patient bookings"""
        try:
            result = await self.api_client.get("/booking", params={"patient_id": patient_id, "limit": limit})
            bookings = result.get("results") or result.get("data") or []
            return {"success": True, "bookings": bookings, "total": len(bookings)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _search_doctors(self, limit: int = 10) -> Dict[str, Any]:
        """
        Search doctors and create selection_map for doctor selection.
        CRITICAL: Must preserve the service_id that was selected BEFORE showing doctors!
        """
        try:
            result = await self.api_client.get_doctors(limit=limit)
            doctors = result.get("results") or result.get("data") or []
            
            # 🚨 CRITICAL: Create SELECTION_MAP for doctor selection
            # Format: "Number X = doctor_id:Y (name: Dr Name)"
            selection_map_lines = []
            for i, doctor in enumerate(doctors):
                doctor_id = doctor.get("id")
                doctor_name = doctor.get("name", "Unknown Doctor")
                selection_map_lines.append(
                    f"Number {i+1} = doctor_id:{doctor_id} (name: {doctor_name})"
                )
            
            selection_map = "\n".join(selection_map_lines)
            logger.warning(f"🩺 CREATED DOCTOR SELECTION_MAP:\n{selection_map}")
            
            return {
                "success": True, 
                "doctors": doctors, 
                "total": len(doctors),
                "selection_map": selection_map  # NEW: Include selection_map
            }
        except Exception as e:
            logger.error(f"❌ search_doctors failed: {e}")
            return {
                "success": False, 
                "error": f"API error: {str(e)}. Tell user there's a technical issue and ask them to try again later. DO NOT make up doctor names from memory."
            }
    
    async def _search_specialists(self, limit: int = 10) -> Dict[str, Any]:
        """Search specialists"""
        try:
            result = await self.api_client.get_specialists(limit=limit)
            specialists = result.get("results") or result.get("data") or []
            return {"success": True, "specialists": specialists, "total": len(specialists)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _search_devices(self, limit: int = 10) -> Dict[str, Any]:
        """Search devices"""
        try:
            result = await self.api_client.get_devices(limit=limit)
            devices = result.get("results") or result.get("data") or []
            return {"success": True, "devices": devices, "total": len(devices)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _create_patient(
        self,
        name: str,
        national_id: str,
        phone: str,
        gender: str
    ) -> Dict[str, Any]:
        """Create new patient with auto-detected gender"""
        try:
            # 🚨 CRITICAL: Auto-detect gender from name using LLM (never trust LLM's gender param)
            from app.utils.gender_detector import detect_gender_from_name
            
            detected_gender = await detect_gender_from_name(name)
            logger.warning(f"🤖 Auto-detected gender: {detected_gender} from name '{name}' (LLM passed: {gender})")
            
            
            # 🚨 CRITICAL: API expects specific field names (not standard names!)
            patient_data = {
                "name": name,
                "identification_id": national_id,  # API expects "identification_id" not "national_id"
                "patient_phone": phone,            # API expects "patient_phone" not "phone"
                "gender": detected_gender,
                "city": "الرياض",                  # Required field (default to Riyadh)
                "country_code": "SA"               # Required for Saudi patients
            }
            logger.info(f"📋 Creating patient with API fields: identification_id={national_id}, patient_phone={phone}")
            
            # 🚨 FIX: Unpack dict as keyword arguments
            result = await self.api_client.create_patient(**patient_data)
            
            # Return with detected gender
            return {
                "success": True, 
                "patient": {**result, "gender": detected_gender},
                "patient_id": result.get("id"),
                "detected_gender": detected_gender
            }
        except Exception as e:
            logger.error(f"❌ Create patient error: {e}")
            return {"success": False, "error": str(e)}
    
    # ========================================================================
    # MAIN HANDLE METHOD - The Brain
    # ========================================================================
    
    async def handle(
        self,
        message: str,
        conversation_history: List[Dict[str, Any]],
        patient_data: Optional[Dict[str, Any]] = None,
        session_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Main handler - LLM decides everything!
        
        Args:
            message: User's message
            conversation_history: Previous conversation messages
            patient_data: Patient information if available
            session_data: Additional session context
            
        Returns:
            Response dict with message and any state updates
        """
        logger.info(f"🤖 [INTELLIGENT AGENT] Processing: '{message[:50]}...'")
        
        # Store current user message for context (used in semantic validation)
        self.current_user_message = message
        self.conversation_history = conversation_history  # For date extraction
        
        # 🚨 CRITICAL: Proactive date detection BEFORE any LLM calls
        detected_date = None
        extracted_date_str = extract_date_from_context(message, conversation_history)
        if extracted_date_str:
            detected_date = extracted_date_str
            logger.info(f"✅ PROACTIVE DATE DETECTION: User said '{message}' → Date: {detected_date}")
            logger.info(f"   This date will be injected into LLM context to guide tool selection")
        
        # Debug: Log session data
        logger.info(f"📊 [DEBUG] session_data keys: {list(session_data.keys()) if session_data else 'None'}")
        topic_value = session_data.get('topic') if session_data else 'N/A'
        logger.info(f"📊 [DEBUG] topic: {str(topic_value)}")
        logger.info(f"📊 [DEBUG] turn: {session_data.get('conversation_turn') if session_data else 'N/A'}")
        
        # Track the last selection_map generated (to save to session)
        last_selection_map = None
        last_selection_type = None  # 'service', 'doctor', 'specialist'
        selected_service_id = session_data.get('selected_service_id') if session_data else None
        selected_doctor_id = session_data.get('selected_doctor_id') if session_data else None
        
        # 🚨 CRITICAL: Track if IDs were NEWLY selected in THIS turn (not just loaded from session)
        newly_selected_service_id = None  # Only set if user actually selected in THIS message
        newly_selected_doctor_id = None   # Only set if user actually selected in THIS message
        
        # 🚨 CRITICAL: Initialize selection_map at function scope (used in validation later)
        selection_map = None
        selection_type = session_data.get('last_selection_type') if session_data else None
        
        # Build system prompt (pass message to detect selections, and detected date if any)
        system_prompt = self._build_system_prompt(patient_data, session_data, message, detected_date)
        
        # Detect message type
        message_lower = message.strip().lower()
        
        # Check if it's a number selection
        is_number_selection = message.strip().isdigit() and 1 <= int(message.strip()) <= 50
        
        # Check if it's a PURE greeting (STRICT - only at conversation start!)
        greeting_keywords = ["هلا", "مرحبا","حياك","مرحب", "السلام", "سلام", "هلو", "hello", "hi"]
        # Only treat as greeting if: (1) matches keyword AND (2) short message AND (3) early in conversation!
        conversation_turn = session_data.get('conversation_turn', 1) if session_data else 1
        is_greeting = (
            any(kw in message_lower for kw in greeting_keywords) and 
            len(message.split()) <= 2 and
            conversation_turn <= 3  # Only first 3 turns can be greetings!
        )
        
        # Initialize last_tool_name (will be used in flow validation later)
        last_tool_name = "unknown"
        
        # Add selection context if user sent a number
        if is_number_selection:
            # Check conversation history to see what was the LAST list shown
            logger.info(f"🔍 Searching last 5 messages for tool_calls (total history: {len(conversation_history)})")
            last_tool_calls = []
            for i, msg in enumerate(reversed(conversation_history[-5:])):
                logger.info(f"  Message {i}: role={msg.get('role')}, has_tool_calls={bool(msg.get('tool_calls'))}, has_content={bool(msg.get('content'))}")
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    last_tool_calls = msg["tool_calls"]
                    logger.info(f"  ✅ Found tool_calls: {[tc['function']['name'] for tc in last_tool_calls]}")
                    break
            
            last_tool_name = last_tool_calls[0]["function"]["name"] if last_tool_calls else "unknown"
            logger.info(f"🔍 User selected #{message.strip()} - Last tool in history: {last_tool_name}")
            
            # Extract SELECTION_MAP from last assistant message OR tool result
            # (selection_map already initialized at function scope)
            
            # 🚨 CRITICAL: Extract selected ID from selection_map and save it
            def extract_id_from_selection(selection_map_text: str, number: str, selection_type: str) -> Optional[int]:
                """Extract service_id, doctor_id, or specialist_id from selection_map"""
                if not selection_map_text:
                    return None
                
                # Look for pattern: "Number {number} = {type}_id:{id}"
                if selection_type == "service":
                    pattern = rf"Number {number} = service_id:(\d+)"
                elif selection_type == "doctor":
                    pattern = rf"Number {number} = doctor_id:(\d+)"
                elif selection_type == "specialist":
                    pattern = rf"Number {number} = specialist_id:(\d+)"
                else:
                    return None
                
                match = re.search(pattern, selection_map_text)
                if match:
                    return int(match.group(1))
                return None
            
            # First, try to find in assistant message content (from force-stop responses)
            for msg in reversed(conversation_history[-5:]):
                if msg.get("role") == "assistant" and msg.get("content"):
                    map_match = re.search(r'\[SELECTION_MAP\](.*?)\[/SELECTION_MAP\]', msg["content"], re.DOTALL)
                    if map_match:
                        selection_map = map_match.group(1).strip()
                        logger.info(f"🗺️ Found SELECTION_MAP in previous assistant message")
                        break
            
            # If not found, try to extract from tool result (from get_service_variants)
            if not selection_map:
                for msg in reversed(conversation_history[-10:]):  # Check more messages
                    if msg.get("role") == "tool":
                        try:
                            tool_content = json.loads(msg.get("content", "{}"))
                            if "selection_map" in tool_content:
                                selection_map = tool_content["selection_map"]
                                logger.info(f"🗺️ Found selection_map in tool result from conversation_history")
                                break
                        except Exception as parse_error:
                            logger.debug(f"Could not parse tool content: {parse_error}")
                            pass
            
            # 🚨 CRITICAL FIX: Also check if selection_map is in SESSION DATA (saved from previous turn)
            if not selection_map and session_data and session_data.get("last_selection_map"):
                selection_map = session_data.get("last_selection_map")
                logger.warning(f"🗺️ LOADED selection_map from session_data:\n{selection_map}")
            
            # 🚨 CRITICAL: Extract and save the selected ID
            if selection_map and selection_type:
                extracted_id = extract_id_from_selection(selection_map, message.strip(), selection_type)
                if extracted_id:
                    if selection_type == "service":
                        selected_service_id = extracted_id
                        newly_selected_service_id = extracted_id  # Mark as NEWLY selected
                        logger.info(f"✅ Extracted selected_service_id={extracted_id} from user's choice #{message.strip()}")
                    elif selection_type == "doctor":
                        selected_doctor_id = extracted_id
                        newly_selected_doctor_id = extracted_id  # Mark as NEWLY selected
                        logger.info(f"✅ Extracted selected_doctor_id={extracted_id} from user's choice #{message.strip()}")
                else:
                    logger.warning(f"⚠️ Could not extract {selection_type}_id from selection_map for number {message.strip()}")
            
            selection_type_display = selection_type.upper() if selection_type else "UNKNOWN"
            selection_context = ""
            if selection_type == "doctor":
                selection_context = "\n🩺 **CONTEXT**: User is selecting a DOCTOR (not a service!). Extract doctor_id!"
            elif selection_type == "service":
                selection_context = "\n🎯 **CONTEXT**: User is selecting a SERVICE. Extract service_id!"
            
            selection_prompt = f"""

🚨🚨🚨 CRITICAL - USER SELECTED NUMBER {message.strip()} FROM YOUR PREVIOUS LIST! 🚨🚨🚨

The user is selecting item #{message.strip()} from the numbered list you just showed them.
📋 **SELECTION TYPE**: {selection_type_display}{selection_context}

📊 **LAST ACTION YOU TOOK**: {last_tool_name}

🗺️ **SELECTION MAP (from your previous tool result):**
{selection_map if selection_map else "ERROR: No map found! Check the 'selection_map' field in your last tool result!"}

✅ **YOUR TASK:**
1. Look at the SELECTION MAP above
2. Find the line: "Number {message.strip()} = ..."
3. Extract the ID from that line (e.g., "service_type_id:7" or "service_id:120" or "doctor_id:3")
4. 🚨 **CHECK RESOURCE REQUIREMENT** on that same line:
   - **[REQUIRES: doctor_id]** → User MUST choose a doctor! Call search_doctors first!
   - **[REQUIRES: specialist_id - AUTO]** → System auto-selects specialist automatically. DO NOT call search_doctors or search_specialists! Proceed to next step.
   - **[REQUIRES: device_id - AUTO]** → System auto-selects device automatically. DO NOT call search_doctors or search_devices! Proceed to next step.
   - **[NO RESOURCE NEEDED]** → No resources needed, proceed to next step
5. **If last tool was search_services** → Use service_type_id to call get_service_variants(service_type_id=X)
6. **If last tool was get_service_variants** → Extract service_id from selection_map:
   - Find line: "Number {message.strip()} = service_id:XXX (name: ...) [REQUIRES: ...]"
   - Extract EXACT service_id from that line (e.g., if user said "5", use service_id from Number 5 line)
   - Check resource requirement on that SAME line
   - If **[REQUIRES: doctor_id]** → Call search_doctors, show list, wait for user choice
   - Otherwise → DO NOT call get_available_slots yet! User must provide DATE first!
7. **If last tool was search_doctors** → Extract doctor_id from selection_map:
   - Find line: "Number {message.strip()} = doctor_id:XXX (name: ...)"
   - Extract EXACT doctor_id from that line
   - NOW ask user for date if not provided yet

🚫🚫🚫 **ABSOLUTELY FORBIDDEN - NEVER HALLUCINATE SERVICE_ID:** 🚫🚫🚫
- Using the 'price' field as service_id!
- Making up a service_id that's not in the selection_map!
- Using a service_id from a DIFFERENT service!
- Guessing or estimating service_id!

✅ **YOU MUST COPY THE EXACT service_id NUMBER FROM THE SELECTION MAP LINE!**

**Example:**
User said: "2"
Selection map shows: "Number 2 = service_id:28 (name: خيوط استقامة الأنف, price: 1500 SAR)"
✅ CORRECT: get_available_slots(service_id=28)
❌ WRONG: get_available_slots(service_id=30)  ← HALLUCINATION!
❌ WRONG: get_available_slots(service_id=1500)  ← THAT'S THE PRICE!
❌ WRONG: get_available_slots(service_id=2)  ← THAT'S THE NUMBER, NOT THE ID!

🚫 **OTHER FORBIDDEN ACTIONS:**
- Skipping get_service_variants after search_services
- Using service_type_id when you need service_id
- Counting manually in the variants array - ONLY use the SELECTION MAP!
- Calling search_doctors for **[REQUIRES: device_id - AUTO]** services! ← THIS IS A CRITICAL ERROR!
- Calling search_doctors for **[REQUIRES: specialist_id - AUTO]** services! ← THIS IS A CRITICAL ERROR!
- Calling search_specialists for ANY service marked as "AUTO" - it's automatic!
- Calling search_devices for ANY service marked as "AUTO" - it's automatic!

✅ **MANDATORY FLOW:**
search_services → get_service_variants → get_available_slots
"""
            system_prompt += selection_prompt
            # Intelligent history management for selections
            # Keep: (1) Tool calls with results, (2) Last selection context, (3) User's selection
            if len(conversation_history) > 10:
                # Keep first 2 (initial context) + last 8 (recent conversation)
                conversation_history = conversation_history[:2] + conversation_history[-8:]
                logger.info(f"📝 History trimmed to 10 messages for selection context")
        
        # Add anti-template instructions for greetings BEFORE creating messages
        elif is_greeting:
            greeting_variety_prompt = """

🚨🚨🚨 CRITICAL - THIS IS A GREETING ONLY! 🚨🚨🚨

The user ONLY said a greeting (مرحبا/هلا/السلام). They did NOT ask about services!

IGNORE ALL PREVIOUS CONVERSATION HISTORY ABOUT SERVICES!

YOUR ONLY JOB: Respond with a SHORT, NATURAL greeting. NOTHING ELSE!

❌ FORBIDDEN - DO NOT:
- Say "دعني أساعدك في معرفة الخدمات" - NO!
- Say "سأبحث لك" - NO!
- Say "وش عندكم" or mention services - NO!
- Mention anything from previous messages - NO!
- Offer to help with anything - NO!
- Say "كيف أقدر أساعدك" - NO!
- Use ANY tools or functions - NO!

✅ CORRECT - DO:
- Just greet back naturally and warmly
- Keep it SHORT (10-15 words max)
- Be friendly and conversational
- That's it!
"""
            system_prompt += greeting_variety_prompt
            # Only clear history if TRULY a new conversation (first greeting)
            if conversation_turn <= 2:
                conversation_history = []  # Clear for FIRST greeting only
                logger.info(f"🆕 New conversation - cleared history for greeting")
            else:
                logger.warning(f"⚠️ Greeting detected at turn {conversation_turn} - keeping history!")
        
        # 🧠 CONTEXT AWARENESS: Detect user frustration/confusion signals
        user_repeating_greeting = False
        user_seems_frustrated = False
        
        if conversation_turn >= 5:
            # Check if user is greeting AGAIN after 5+ turns (sign of confusion)
            greeting_patterns = ["السلام عليكم", "مرحبا", "هلا", "السلام", "مرحبا", "hello", "hi"]
            if any(pattern in message.lower() for pattern in greeting_patterns):
                # Count previous greetings
                previous_greetings = sum(1 for msg in conversation_history[-10:] 
                                       if msg.get("role") == "user" and 
                                       any(p in msg.get("content", "").lower() for p in greeting_patterns))
                if previous_greetings >= 1:
                    user_repeating_greeting = True
                    logger.warning(f"🚨 User repeating greeting at turn {conversation_turn} - possible confusion!")
        
        # Detect frustration keywords
        frustration_keywords = ["ما فهمت", "مو واضح", "وش تبي", "ما أدري", "تعبت", "خلاص", "نسيت"]
        if any(keyword in message.lower() for keyword in frustration_keywords):
            user_seems_frustrated = True
            logger.warning(f"😰 User seems frustrated: '{message[:50]}'")
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": message
        })
        
        # Intelligent conversation history management before sending to LLM
        # Prevent context window overflow while preserving critical information
        if len(conversation_history) > 20:
            logger.warning(f"⚠️ Long conversation ({len(conversation_history)} messages) - applying intelligent truncation")
            
            # Strategy: Keep important messages + recent context
            # 1. First 2 messages (initial context)
            # 2. Any messages with tool calls/results (critical state)
            # 3. Last 10 messages (recent conversation)
            
            important_messages = []
            # Keep first 2
            important_messages.extend(conversation_history[:2])
            
            # Find tool-related messages in middle section
            middle_section = conversation_history[2:-10] if len(conversation_history) > 12 else []
            for msg in middle_section:
                if msg.get('role') == 'tool' or msg.get('tool_calls'):
                    important_messages.append(msg)
            
            # Keep last 10
            important_messages.extend(conversation_history[-10:])
            
            conversation_history = important_messages
            logger.info(f"📝 History intelligently trimmed to {len(conversation_history)} messages")
        
        # 🧠 Add context awareness to system prompt
        if user_repeating_greeting or user_seems_frustrated:
            context_note = "\n\n🚨 CRITICAL CONTEXT:\n"
            if user_repeating_greeting:
                context_note += "- User is greeting AGAIN (they seem confused or lost)\n"
                context_note += "- Don't just greet back! Acknowledge they might be lost\n"
                context_note += "- Example: 'أهلاً فيك مرة ثانية! شكلك مو متأكد... خليني أساعدك'\n"
                context_note += "- Ask directly: 'أنت الحين تبي تحجز موعد ولا تسأل عن الخدمات؟'\n"
            if user_seems_frustrated:
                context_note += "- User seems FRUSTRATED or confused\n"
                context_note += "- Be EXTRA helpful and patient\n"
                context_note += "- Simplify your response\n"
                context_note += "- Offer direct help: 'خليني أساعدك خطوة خطوة...'\n"
            system_prompt += context_note
        
        # Prepare messages for LLM (system prompt now includes greeting rules if needed)
        messages = [{"role": "system", "content": system_prompt}] + conversation_history
        
        # Call LLM with function calling
        max_iterations = 50  # Allow complex multi-step workflows (increased from 5)
        iteration = 0
        consecutive_failures = 0  # Track repeated failures
        last_error = None
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"🔄 LLM Iteration {iteration}/50")
            
            try:
                # Disable tools for simple greetings
                if is_greeting and iteration == 1:
                    logger.info(f"🚫 Greeting detected - disabling tools for this message")
                    response = await self.llm.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7  # More creative for greetings
                    )
                else:
                    response = await self.llm.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        tools=self.get_tools_definition(),
                        tool_choice="auto",  # LLM decides when to use tools
                        temperature=0.5
                    )
                
                assistant_message = response.choices[0].message
                
                # Check if LLM wants to call functions
                if assistant_message.tool_calls:
                    # LLM decided to use tools
                    logger.info(f"🔧 LLM wants to call {len(assistant_message.tool_calls)} tool(s)")
                    
                    # 🚨 ENHANCED LOOP DETECTOR: Check function name AND arguments
                    current_tool_name = assistant_message.tool_calls[0].function.name
                    current_tool_args = assistant_message.tool_calls[0].function.arguments
                    last_tool_name = None
                    last_tool_args = None
                    
                    # Find last tool call in history
                    for msg in reversed(messages[-5:]):
                        if msg.get("role") == "assistant" and msg.get("tool_calls"):
                            last_tool_name = msg["tool_calls"][0]["function"]["name"]
                            last_tool_args = msg["tool_calls"][0]["function"]["arguments"]
                            break
                    
                    # Note: Anti-loop logic removed - trust the LLM to understand conversation context
                    # The LLM can see from history what it just called and shouldn't repeat unnecessarily
                    
                    # CRITICAL: Validate BEFORE adding assistant message to avoid invalid states
                    # Phase 1: Validation ONLY (don't execute yet)
                    validation_failed = False
                    validation_system_message = None
                    
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # 🚨 CRITICAL: Detect flow violations BEFORE execution
                        logger.info(f"🔍 Flow check: is_number_selection={is_number_selection}, last_tool_name={last_tool_name}, function_name={function_name}")
                        
                        # 🚨🚨🚨 CRITICAL NEW VALIDATION: Block search_doctors for specialist services!
                        # Check session_data for selection_map from previous turn
                        saved_selection_map = session_data.get("last_selection_map") if session_data else None
                        if saved_selection_map:
                            logger.info(f"🔍 VALIDATION: Checking selection_map for resource requirements (user selected: {message.strip()})")
                        if is_number_selection and function_name == "search_doctors" and saved_selection_map:
                            # Extract the selected number from user message
                            selected_number = message.strip()
                            if selected_number.isdigit():
                                # Find the corresponding line in selection_map
                                for line in saved_selection_map.split("\n"):
                                    if f"Number {selected_number} =" in line:
                                        # Check if this line says [REQUIRES: specialist_id - AUTO]
                                        if "[REQUIRES: specialist_id - AUTO]" in line:
                                            logger.error(f"🚨 CRITICAL BUG BLOCKED! LLM tried to call search_doctors for SPECIALIST service!")
                                            logger.error(f"   Selection: {line}")
                                            logger.error(f"   LLM called: search_doctors() ❌")
                                            logger.error(f"   Should call: get_available_slots() directly (specialist auto-selected)")
                                            
                                            # Block this call
                                            validation_failed = True
                                            validation_system_message = "This service requires a SPECIALIST (auto-selected). DO NOT call search_doctors! Call get_available_slots directly to show available time slots. Specialist will be auto-selected by the system."
                                            break
                                        elif "[REQUIRES: device_id - AUTO]" in line:
                                            logger.error(f"🚨 CRITICAL BUG BLOCKED! LLM tried to call search_doctors for DEVICE service!")
                                            logger.error(f"   Selection: {line}")
                                            logger.error(f"   LLM called: search_doctors() ❌")
                                            logger.error(f"   Should call: get_available_slots() directly (device auto-selected)")
                                            
                                            # Block this call
                                            validation_failed = True
                                            validation_system_message = "This service uses a DEVICE (auto-selected). DO NOT call search_doctors! Call get_available_slots directly to show available time slots. Device will be auto-selected by the system."
                                            break
                        
                        # 🚨🚨🚨 NEW CRITICAL VALIDATION: Block ANY tool after get_service_variants
                        if last_tool_name == "get_service_variants" and function_name in ["search_doctors", "get_available_slots", "search_specialists", "search_devices"]:
                            # Check if we should allow progression
                            should_allow = False
                            
                            # Exception 1: If only 1 variant was returned, auto-progression is OK
                            last_variants_result = None
                            variant_count = 0
                            for msg in reversed(messages):
                                if msg.get("role") == "tool":
                                    try:
                                        content = json.loads(msg.get("content", "{}"))
                                        if "variants" in content:
                                            last_variants_result = content
                                            variant_count = len(content.get("variants", []))
                                            break
                                    except:
                                        pass
                            
                            if variant_count == 1:
                                logger.info(f"✅ Only 1 variant found - allowing auto-progression to {function_name}")
                                should_allow = True
                            
                            # Exception 2: If user's message is NOT a number, they're not selecting from list
                            # (e.g., user said "بتول" which is a doctor name, not a service selection)
                            if not is_number_selection and not message.strip().isdigit():
                                logger.info(f"✅ User message '{message.strip()}' is not a number selection - allowing {function_name}")
                                should_allow = True
                            
                            if not should_allow:
                                logger.error(f"🚨 CRITICAL FLOW VIOLATION! LLM calling {function_name} immediately after get_service_variants!")
                                logger.error(f"🚨 User selected #{message.strip()} from CATEGORY, got variants, but LLM SKIPPED showing them!")
                                logger.error(f"🚨 BLOCKING: Must show subservices to user FIRST before calling {function_name}!")
                                logger.error(f"🚨 FORCE-STOPPING to show variants instead!")
                                
                                # Set validation failure flag
                                validation_failed = True
                                validation_system_message = "You just received service options. The user needs to see these options before proceeding. Show them the available services naturally and let them choose. Don't call other tools yet - respond with text."
                                break
                        
                        # 🚨🚨🚨 FLOW VIOLATION 0: Skipping get_service_variants after search_services
                        if is_number_selection and last_tool_name == "search_services" and function_name in ["search_doctors", "get_available_slots", "search_specialists", "search_devices"]:
                            logger.error(f"🚨 CRITICAL FLOW VIOLATION! User selected from search_services but LLM called {function_name}!")
                            logger.error(f"🚨 MUST call get_service_variants FIRST to show subservices!")
                            logger.error(f"🚨 FORCE-STOPPING to correct flow!")
                            
                            # Set validation failure flag
                            validation_failed = True
                            validation_system_message = "The user selected a category. Before proceeding, you need to fetch and show them the specific service options within that category using get_service_variants."
                            break
                        
                        # FLOW VIOLATION 1: Skipping get_service_variants after search_services (old validation - kept for safety)
                        if is_number_selection and last_tool_name == "search_services" and function_name == "get_available_slots":
                            logger.error(f"🚨 FLOW VIOLATION 1 DETECTED! User selected from search_services but LLM called get_available_slots!")
                            logger.error(f"🚨 CORRECTING: Should call get_service_variants first!")
                            # Inject error to force LLM to call get_service_variants
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": function_name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }]
                            })
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps({
                                    "success": False,
                                    "error": "Call get_service_variants first to show subservices."
                                }, ensure_ascii=False)
                            })
                            continue  # Skip execution, let LLM correct itself
                        
                        # FLOW VIOLATION 2: Calling get_service_variants without knowing correct service_type_id
                        # (User said service NAME, but LLM didn't call search_services first)
                        if function_name == "get_service_variants" and not is_number_selection:
                            # Check if there was a recent search_services call
                            recent_search = False
                            requested_service_type_id = function_args.get("service_type_id")
                            
                            # Look through last 5 messages for search_services tool call
                            for msg in reversed(messages[-5:]):
                                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                                    for tc in msg.get("tool_calls", []):
                                        if tc.get("function", {}).get("name") == "search_services":
                                            recent_search = True
                                            break
                                if recent_search:
                                    break
                            
                            # Also check if service_type_id is in selection_map (user selected number from categories)
                            has_selection_map = False
                            if selection_map and requested_service_type_id:
                                pattern = rf"service_type_id:{requested_service_type_id}"
                                if re.search(pattern, selection_map):
                                    has_selection_map = True
                            
                            if not recent_search and not has_selection_map:
                                logger.error(f"🚨 FLOW VIOLATION: LLM calling get_service_variants(service_type_id={requested_service_type_id}) WITHOUT search!")
                                logger.error(f"🚨 User said: '{message}' (service NAME, not a number)")
                                logger.error(f"🚨 LLM is GUESSING service_type_id! Must call search_services first!")
                                
                                # Set validation failure flag
                                validation_failed = True
                                validation_system_message = "The user mentioned a service by name. You need to search for it first using search_services to get the correct category ID before calling get_service_variants."
                                break
                        
                        # FLOW VIOLATION 3: Using service_id in get_service_variants (should be service_type_id)
                        if function_name == "get_service_variants" and "service_id" in function_args:
                            logger.error(f"🚨 FLOW VIOLATION 2 DETECTED! LLM called get_service_variants with service_id (should be service_type_id)!")
                            logger.error(f"🚨 Args: {function_args}")
                            
                            # Set validation failure flag
                            validation_failed = True
                            validation_system_message = "Use service_type_id parameter for get_service_variants, not service_id."
                            break
                        
                        # VALIDATION: BLOCK get_available_slots if no selection_map (bypassing flow)
                        if function_name == "get_available_slots" and not is_number_selection:
                            # User didn't select from a numbered list
                            # Check if there's a recent get_service_variants in history with results
                            has_recent_variants = False
                            for msg in reversed(messages[-10:]):
                                if msg.get("role") == "tool" and "service_id" in str(msg.get("content", "")):
                                    has_recent_variants = True
                                    break
                            
                            # 🚨 CRITICAL: Check if we have saved context from previous turns
                            has_saved_context = selected_service_id is not None
                            
                            if detected_date:
                                logger.info(f"✅ Date detected: {detected_date} - context looks valid")
                                has_saved_context = True  # Date detection indicates valid booking flow
                            
                            if not has_recent_variants and not selection_map and not has_saved_context:
                                logger.error(f"🚨 CRITICAL: LLM calling get_available_slots WITHOUT proper flow!")
                                logger.error(f"🚨 User said: '{message}' (not a number selection)")
                                logger.error(f"🚨 No selection_map, no saved context, no date detected!")
                                logger.error(f"🚨 service_id={function_args.get('service_id')} is likely WRONG!")
                                
                                # Set validation failure flag
                                validation_failed = True
                                validation_system_message = "User input doesn't match expected flow. If user mentioned a service name, call search_services first. If user mentioned a doctor name, call search_doctors first. Don't guess service_id without proper selection."
                                break
                            elif has_saved_context:
                                logger.info(f"✅ Saved context available (service_id={selected_service_id}, doctor_id={selected_doctor_id}) - allowing get_available_slots")
                    
                    # Check validation result and handle accordingly
                    if validation_failed:
                        logger.warning(f"🚨 Validation failed - adding system guidance and retrying")
                        messages.append({
                            "role": "system",
                            "content": validation_system_message
                        })
                        continue  # Go to next LLM iteration
                    
                    # Validation passed! Add assistant message BEFORE executing tools
                    messages.append({
                        "role": "assistant",
                        "content": assistant_message.content,
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in assistant_message.tool_calls
                        ]
                    })
                    logger.info(f"✅ Added assistant message with tool_calls (validation passed)")
                    
                    # Phase 2: Execute tools and add results (validation already done in Phase 1)
                    for tool_call in assistant_message.tool_calls:
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # 🚨 CRITICAL: Validate service_id against saved selection (for slots AND booking)
                        if function_name in ["get_available_slots", "create_booking"] and "service_id" in function_args:
                            llm_service_id = function_args.get("service_id")
                            
                            # Check if we have a saved selected_service_id
                            if selected_service_id and llm_service_id != selected_service_id:
                                logger.error(f"🚨 SERVICE_ID MISMATCH!")
                                logger.error(f"   User selected service_id: {selected_service_id}")
                                logger.error(f"   LLM trying to use: {llm_service_id}")
                                logger.error(f"   🔧 AUTO-CORRECTING to user's selection!")
                                function_args["service_id"] = selected_service_id
                            
                            # Also validate doctor_id if we have it saved
                            llm_doctor_id = function_args.get("doctor_id")
                            if selected_doctor_id:
                                if llm_doctor_id and llm_doctor_id != selected_doctor_id:
                                    logger.error(f"🚨 DOCTOR_ID MISMATCH!")
                                    logger.error(f"   User selected doctor_id: {selected_doctor_id}")
                                    logger.error(f"   LLM trying to use: {llm_doctor_id}")
                                    logger.error(f"   🔧 AUTO-CORRECTING to user's selection!")
                                    function_args["doctor_id"] = selected_doctor_id
                                elif not llm_doctor_id:
                                    logger.info(f"✅ AUTO-ADDING saved doctor_id={selected_doctor_id} to get_available_slots")
                                    function_args["doctor_id"] = selected_doctor_id
                            elif llm_doctor_id and not selected_doctor_id:
                                # LLM provided doctor_id but user never selected one!
                                logger.error(f"🚨 INVALID DOCTOR_ID - User never selected a doctor!")
                                logger.error(f"   LLM trying to use: doctor_id={llm_doctor_id}")
                                logger.error(f"   But selected_doctor_id=None in context")
                                logger.error(f"   🔧 BLOCKING this call - LLM must call search_doctors first!")
                                
                                # Add error to messages to guide LLM
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call.id,
                                    "content": json.dumps({
                                        "success": False,
                                        "error": "doctor_required_but_not_selected",
                                        "message": "User has not selected a doctor yet. You must call search_doctors() first to show the doctor list, then wait for user to select a number."
                                    }, ensure_ascii=False)
                                })
                                continue  # Skip execution, go to next iteration
                        
                        # Auto-correction: Verify service_id matches selection_map when user selected a number
                        if is_number_selection and function_name == "get_available_slots" and "service_id" in function_args:
                            selected_number = int(message.strip())
                            llm_service_id = function_args.get("service_id")
                            
                            # Extract expected service_id from selection_map
                            if selection_map:
                                # Look for line: "Number X = service_id:YYY"
                                pattern = rf"Number {selected_number} = service_id:(\d+)"
                                match = re.search(pattern, selection_map)
                                if match:
                                    expected_service_id = int(match.group(1))
                                    logger.warning(f"🔍 SERVICE_ID VALIDATION: User selected #{selected_number}")
                                    logger.warning(f"   Expected service_id: {expected_service_id} (from selection_map)")
                                    logger.warning(f"   LLM used service_id: {llm_service_id}")
                                    
                                    if llm_service_id != expected_service_id:
                                        logger.error(f"🚨 VALIDATION FAILED! LLM using WRONG service_id!")
                                        # Extract service name from selection_map for better error message
                                        name_pattern = rf"Number {selected_number} = service_id:\d+ \(name: ([^,]+),"
                                        name_match = re.search(name_pattern, selection_map)
                                        correct_service_name = name_match.group(1) if name_match else "Unknown"
                                        
                                        # 🚨 FIX: Override the LLM's service_id with the correct one!
                                        logger.warning(f"🔧 AUTO-CORRECTING: Overriding service_id {llm_service_id} → {expected_service_id}")
                                        
                                        # Parse the tool arguments and fix the service_id
                                        try:
                                            tool_args = json.loads(tool_call.function.arguments)
                                            tool_args["service_id"] = expected_service_id
                                            # Update the function_args for execution
                                            function_args = tool_args
                                            logger.info(f"✅ Corrected service_id in function_args: {function_args}")
                                        except Exception as parse_error:
                                            logger.error(f"❌ Failed to parse/correct tool arguments: {parse_error}")
                                            # If we can't fix it, add error message and continue
                                            messages.append({
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [{
                                                    "id": tool_call.id,
                                                    "type": "function",
                                                    "function": {
                                                        "name": function_name,
                                                        "arguments": tool_call.function.arguments
                                                    }
                                                }]
                                            })
                                            messages.append({
                                                "role": "tool",
                                                "tool_call_id": tool_call.id,
                                                "content": json.dumps({
                                                    "success": False,
                                                    "error": f"Wrong service_id. Use {expected_service_id}."
                                                }, ensure_ascii=False)
                                            })
                                            continue
                                    else:
                                        logger.info(f"✅ SERVICE_ID VALIDATION PASSED: {llm_service_id} matches selection_map")
                                else:
                                    logger.warning(f"⚠️ Could not find 'Number {selected_number}' in selection_map for validation")
                        
                        # Execute the function (pass context for auto-injection)
                        function_result = await self.execute_tool(
                            function_name, 
                            function_args,
                            detected_date=detected_date,
                            selected_service_id=selected_service_id,
                            selected_doctor_id=selected_doctor_id,
                            patient_data=patient_data
                        )
                        
                        # 🚨 CRITICAL: Save selection_map if this is get_service_variants
                        if function_name == "get_service_variants" and function_result.get("selection_map"):
                            last_selection_map = function_result.get("selection_map")
                            last_selection_type = "service"
                            logger.warning(f"💾 Saved SERVICE selection_map for next turn ({len(last_selection_map)} chars)")
                            
                            # Check if single variant was auto-selected
                            if function_result.get("auto_selected_service_id"):
                                selected_service_id = function_result.get("auto_selected_service_id")
                                newly_selected_service_id = selected_service_id  # Mark as NEWLY selected
                                logger.info(f"✅ AUTO-SELECTED service_id={selected_service_id} (only 1 variant found)")
                        
                        # 🚨 CRITICAL: Save selection_map if this is search_doctors
                        if function_name == "search_doctors" and function_result.get("selection_map"):
                            last_selection_map = function_result.get("selection_map")
                            last_selection_type = "doctor"
                            logger.warning(f"💾 Saved DOCTOR selection_map for next turn ({len(last_selection_map)} chars)")
                            
                            # 🚨 CRITICAL: Clear stale doctor_id when showing NEW doctor list
                            # User will be asked to choose, so old selection is invalid
                            selected_doctor_id = None
                            newly_selected_doctor_id = None
                            logger.warning("🔄 Cleared stale doctor_id - user will select from new list")
                        
                        # 🚨 CRITICAL: Update patient_data if create_patient succeeded
                        if function_name == "create_patient" and function_result.get("success"):
                            patient_info = function_result.get("patient", {})
                            patient_data = {
                                "id": patient_info.get("id"),
                                "name": patient_info.get("name"),
                                "phone": patient_info.get("phone"),
                                "national_id": patient_info.get("national_id"),
                                "gender": patient_info.get("gender"),
                                "already_registered": True
                            }
                            logger.warning(f"✅ Patient registered successfully! ID={patient_data['id']}, Name={patient_data['name']}")
                            logger.warning(f"🔄 Updated patient_data in memory - will be saved to session")
                        
                        # Log result for debugging
                        logger.info(f"🔧 Tool result ({function_name}): success={function_result.get('success', False)}, items={len(function_result.get('services', []))} or {len(function_result.get('variants', []))} or {len(function_result.get('slots', []))}")
                        
                        # 🚨 CRITICAL: Detect repeated failures to prevent infinite loops
                        if not function_result.get('success', True):
                            current_error = function_result.get('error', '')
                            if current_error == last_error:
                                consecutive_failures += 1
                                logger.error(f"🚨 REPEATED FAILURE #{consecutive_failures}: {current_error}")
                                
                                if consecutive_failures >= 3:
                                    logger.error(f"🚨 BREAKING LOOP: Same error {consecutive_failures} times!")
                                    return {
                                        "response": "عذراً، حصل خطأ تقني في النظام. رجاءً جرب مرة ثانية أو تواصل معنا على 920033304.",
                                        "status": "error_loop_detected",
                                        "error": current_error
                                    }
                            else:
                                consecutive_failures = 1  # Reset counter for new error
                                last_error = current_error
                        else:
                            consecutive_failures = 0  # Reset on success
                            last_error = None
                        
                        # Add result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(function_result, ensure_ascii=False)
                        })
                        
                        # 🚨 FORCE-STOP only after search_services (categories are always shown)
                        # DON'T force-stop after get_service_variants - let LLM ask clarifying questions!
                        # DON'T force-stop for generic queries - LLM should ask clarifying question instead
                        # DON'T force-stop if only 1 result - let LLM continue to get_service_variants automatically!
                        num_services = len(function_result.get("services", [])) if function_name == "search_services" else 0
                        
                        should_force_stop = (
                            (function_name == "search_services" and 
                             function_result.get("success") and 
                             not function_result.get("is_generic_query", False) and
                             num_services > 1)  # Only force-stop if MULTIPLE categories (user needs to choose)
                        )
                        
                        # 🚨 CRITICAL: If only 1 category, let LLM continue to get_service_variants automatically
                        if function_name == "search_services" and num_services == 1 and function_result.get("success"):
                            logger.info(f"✅ Only 1 category found - letting LLM continue to get_service_variants automatically (no force-stop)")
                        
                        if should_force_stop:
                            logger.warning(f"🛑 FORCE-STOP: {function_name} executed - generating response immediately!")
                            
                            # Build appropriate instruction with EMBEDDED ID MAPPING
                            # (Only search_services reaches here now - get_service_variants no longer force-stopped)
                            if function_name == "search_services":
                                # Extract category IDs for mapping
                                services = function_result.get("services", [])
                                service_map = "\n".join([
                                    f"Number {i+1} = service_type_id:{s.get('service_type_id')} (name: {s.get('service_type_name', 'Unknown')})" 
                                    for i, s in enumerate(services)
                                ])
                                
                                stop_instruction = f"""🚨 CRITICAL: You just received service CATEGORIES.
                                
🚨🚨🚨 **CRITICAL - PRESENTATION ORDER MATTERS!** 🚨🚨🚨

You MUST present the services in the EXACT SAME ORDER as this mapping:

{service_map}

❌ DO NOT reorder services by relevance or alphabetically!
❌ DO NOT skip any services!
✅ Show them in EXACT order: Number 1 first, then 2, then 3, etc.
✅ This ensures user's number selection maps correctly!

STEP 1: Generate a NATURAL, conversational response showing ALL categories with clear numbers (1, 2, 3...).

🚫 **ABSOLUTELY FORBIDDEN PHRASES - NEVER USE THESE:**
- "اختار رقم الخدمة" ❌
- "اختر رقم" ❌
- "رقم الخدمة" ❌
- "اختر الرقم" ❌
- "حدد الرقم" ❌
- "اختار الرقم" ❌
- "وش رقم" ❌
- ANY mention of "اختار رقم" or asking user to "choose number"!

✅ **NATURAL ALTERNATIVES - USE THESE:**
- "وش يهمك من هذي؟" 😊
- "أي وحدة تبي؟" 
- "وش تفضل؟"
- "شد عينك على أي خدمة؟"
- Just list naturally and let user respond with number OR service name - both work!
- Be conversational like a friendly human receptionist, NOT a robot menu!

STEP 2: At the VERY END of your response, add this EXACT mapping (user won't see this, it's for YOUR reference next turn):

[SELECTION_MAP]
{service_map}
[/SELECTION_MAP]

This mapping is CRITICAL! When user replies with a number, you'll read this map to get the correct service_type_id!"""
                            
                            # Force LLM to generate TEXT ONLY response (NO tools available!)
                            try:
                                force_response = await self.llm.chat.completions.create(
                                    model=self.model,
                                    messages=messages + [{
                                        "role": "system",
                                        "content": stop_instruction
                                    }],
                                    temperature=0.3
                                    # NOTE: NO tools parameter = LLM cannot call functions!
                                )
                                
                                forced_text = force_response.choices[0].message.content
                            except Exception as force_error:
                                logger.error(f"❌ Force-stop OpenAI call failed: {force_error}")
                                # Fallback: Let LLM continue naturally without force-stop
                                logger.warning("⚠️ Skipping force-stop, letting LLM process results naturally")
                                continue  # Go back to LLM iteration loop
                            
                            # Strip [SELECTION_MAP] section from user-facing response (keep in history though!)
                            
                            # Try to strip with closing tag first
                            forced_text_for_user = re.sub(r'\[SELECTION_MAP\].*?\[/SELECTION_MAP\]', '', forced_text, flags=re.DOTALL).strip()
                            
                            # If that didn't work (no closing tag), strip everything after [SELECTION_MAP]
                            if '[SELECTION_MAP]' in forced_text_for_user:
                                logger.warning(f"⚠️ [SELECTION_MAP] closing tag missing - stripping everything after opening tag")
                                forced_text_for_user = forced_text_for_user.split('[SELECTION_MAP]')[0].strip()
                            
                            logger.info(f"✅ FORCED response generated: {len(forced_text)} chars (user sees {len(forced_text_for_user)} chars)")
                            
                            # 🚨 CRITICAL: Add BOTH the tool call AND the response to history!
                            # This preserves context for next turn when user selects a number
                            
                            # 1. Add tool call to history (so selection prompt can detect it)
                            conversation_history.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [{
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": function_name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }]
                            })
                            
                            # 2. Add tool result
                            conversation_history.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(function_result, ensure_ascii=False)
                            })
                            
                            # 3. Add forced text response (WITH SELECTION_MAP for next turn!)
                            conversation_history.append({
                                "role": "assistant",
                                "content": forced_text  # Full text with [SELECTION_MAP] in history
                            })
                            
                            logger.info(f"📝 Added tool call + result + response to history for next turn context")
                            
                            return {
                                "response": forced_text_for_user,  # User sees clean text without map
                                "status": "success",
                                "conversation_history": conversation_history  # Return full history!
                            }
                    
                    # Continue loop - LLM will process results and decide next step
                    continue
                
                else:
                    # LLM responded with text (no more tools needed)
                    final_response = assistant_message.content
                    
                    # 🚨 CRITICAL: Strip [SELECTION_MAP] from user-facing response!
                    
                    # Try to strip with closing tag first
                    final_response_for_user = re.sub(r'\[SELECTION_MAP\].*?\[/SELECTION_MAP\]', '', final_response, flags=re.DOTALL).strip()
                    
                    # If that didn't work (no closing tag), strip everything after [SELECTION_MAP]
                    if '[SELECTION_MAP]' in final_response_for_user:
                        logger.warning(f"⚠️ [SELECTION_MAP] closing tag missing in final response - stripping after opening tag")
                        final_response_for_user = final_response_for_user.split('[SELECTION_MAP]')[0].strip()
                    
                    # Add FULL response to conversation history (with map for next turn)
                    conversation_history.append({
                        "role": "assistant",
                        "content": final_response  # Keep map in history
                    })
                    
                    logger.info(f"✅ [INTELLIGENT AGENT] Final response generated")
                    
                    # Return history so router can save it
                    result = {
                        "response": final_response_for_user,  # User sees clean text
                        "status": "success",
                        "conversation_history": conversation_history
                    }
                    
                    # Save selection_map to session if we generated one
                    if last_selection_map:
                        result["last_selection_map"] = last_selection_map
                        result["last_selection_type"] = last_selection_type
                        logger.info(f"💾 Returning {last_selection_type} selection_map to save to session ({len(last_selection_map)} chars)")
                    
                    # Save selected IDs to session ONLY if NEWLY selected in this turn
                    # This prevents returning stale IDs that were just loaded from session
                    if newly_selected_service_id:
                        result["selected_service_id"] = newly_selected_service_id
                        logger.info(f"💾 Returning NEWLY selected service_id={newly_selected_service_id}")
                    if newly_selected_doctor_id:
                        result["selected_doctor_id"] = newly_selected_doctor_id
                        logger.info(f"💾 Returning NEWLY selected doctor_id={newly_selected_doctor_id}")
                    
                    # 🚨 CRITICAL: Return updated patient_data if registration happened
                    if patient_data and patient_data.get("already_registered"):
                        result["patient_data"] = patient_data
                        logger.info(f"💾 Returning updated patient_data to save to session (ID={patient_data.get('id')})")
                    
                    return result
            
            except Exception as e:
                error_type = type(e).__name__
                logger.error(f"❌ LLM error on iteration {iteration}: {error_type} - {e}")
                
                # 🚨 SMART RETRY: If timeout and we have history, try with shorter context
                if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                    logger.warning(f"⏱️ Timeout detected - attempting retry with shorter context...")
                    if len(conversation_history) > 6:
                        # Try again with only last 6 messages
                        logger.info(f"🔄 Retrying with trimmed history: {len(conversation_history)} → 6 messages")
                        conversation_history = conversation_history[-6:]
                        continue  # Retry the loop
                
                # For other errors or if retry failed, return user-friendly error
                logger.error(f"❌ Failed after timeout retry or non-timeout error", exc_info=True)
                return {
                    "response": "آسف! شكلي تأخرت عليك شوي... 😅\nممكن تعيد طلبك مرة ثانية؟\n\n(لو ما زبطت، اتصل على 920033304 ونساعدك مباشرة 📞)",
                    "status": "error",
                    "error": str(e)
                }
        
        # Max iterations reached
        logger.warning(f"⚠️ Max iterations ({max_iterations}) reached")
        return {
            "response": "عذراً، النظام مشغول حالياً. تقدر تتصل على 920033304 للمساعدة.",
            "status": "max_iterations_reached"
        }
    
    def _build_system_prompt(
        self,
        patient_data: Optional[Dict[str, Any]],
        session_data: Optional[Dict[str, Any]],
        current_message: str = "",
        detected_date: Optional[str] = None
    ) -> str:
        """
        Enhanced sales-optimized system prompt for Reem - the AI booking agent.
        
        Key enhancements:
        - Clear sales objectives and metrics
        - Proactive upselling and cross-selling
        - Objection handling patterns
        - Conversion optimization tactics
        - Natural urgency creation
        - Proactive date detection and context awareness
        """
        
        # Build dynamic context (unchanged)
        dynamic_context = self._build_dynamic_context(patient_data, session_data, current_message)
        
        # Check patient registration status FIRST
        is_registered = patient_data and patient_data.get("already_registered", False)
        patient_name = patient_data.get("name") if patient_data else None
        
        prompt = f"""🏥 You are "Reem" - Expert Medical Sales Agent at Wajen Medical Center

    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║  🚨🚨🚨 CRITICAL RULES - READ FIRST! 🚨🚨🚨                                    ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    
    1️⃣ **BE PROACTIVE WITH TOOLS - DON'T BE VAGUE!**
       🚨🚨🚨 **ABSOLUTELY FORBIDDEN:** Saying "I don't have that service" WITHOUT searching first!
       
       ❌ WRONG: User: "خيوط استقامة" → You: "ما عندنا هالخدمة"
       ✅ RIGHT: User: "خيوط استقامة" → You: Call search_services("خيوط استقامة") → Show results
       
       🚨 **MANDATORY RULE:** 
       - If user mentions ANY service name → MUST call search_services() FIRST!
       - NEVER say "we don't have it" without actually searching the database
       - ALWAYS show what you found (even if it's not exact match)
       - If truly nothing found → offer alternatives
       
    2️⃣ **CHECK REGISTRATION STATUS FIRST!**
       **PATIENT STATUS: {"✅ REGISTERED" if is_registered else "❌ NOT REGISTERED"}**
       🚨 **MANDATORY**: Registration BEFORE Booking!
    """
        
        # Add registration instructions if NOT registered
        if not is_registered:
            # 🚨 CRITICAL: Get actual phone number from session_data (WhatsApp number)
            phone_from_wa = None
            if session_data:
                phone_from_wa = session_data.get('phone_number') or session_data.get('phone')
            if not phone_from_wa and patient_data:
                phone_from_wa = patient_data.get('phone')
            
            # If still no phone, this is a critical error
            if not phone_from_wa:
                logger.error("🚨 CRITICAL: No phone number found in session_data or patient_data!")
                phone_from_wa = "MISSING_PHONE"
            
            logger.info(f"📞 Registration will use phone: {phone_from_wa}")
            
            prompt += f"""
    ❌ **Patient is NOT registered yet!**

    **YOU MUST REGISTER THEM FIRST:**
    1. ⚠️ DO NOT show services or discuss booking until registered!
    2. ⚠️ If user asks about services/booking, say: "حبيبي أول شي بحتاج أسجلك في النظام علشان أقدر أحجز لك. ممكن اسمك الكامل؟"
    3. ⚠️ Collect registration info step by step:
       - **Full name (in Arabic)**: "ممكن اسمك الكامل؟"
       - **National ID (10 digits)**: "رقم هويتك الوطنية؟" or "رقم الإقامة؟"
       - **Gender**: AUTO-DETECTED from name - DO NOT ask user!
       - **Phone**: Already captured from WhatsApp - DON'T ask!
    4. ⚠️ After collecting name and national_id, call create_patient tool with:
       - name: Full Arabic name
       - national_id: 10-digit ID  
       - phone: "{phone_from_wa}"  ← USE THIS EXACT VALUE!
       - gender: Always pass "male" (system will auto-detect correct gender from name)
    5. ✅ ONLY AFTER successful registration (create_patient returns patient_id) can you proceed with booking

    **🚨 CRITICAL: When calling create_patient, use phone="{phone_from_wa}" EXACTLY as shown above!**
    **Registration is MANDATORY - no exceptions! No booking without registration!**
    **NEVER ask for gender - it's detected automatically from the Arabic name!**
    """
        else:
            prompt += """
    ✅ **Patient is registered - proceed with booking flow**
    """
        
        # 🚨 CRITICAL: Inject detected date if user provided one
        if detected_date:
            from datetime import datetime
            try:
                date_obj = datetime.strptime(detected_date, "%Y-%m-%d")
                formatted_date = date_obj.strftime("%A, %B %d, %Y")
                prompt += f"""
    
    🗓️ **DETECTED DATE FROM USER'S MESSAGE:**
    User said: "{current_message}"
    Extracted booking date: {detected_date} ({formatted_date})
    
    🚨 **CRITICAL CONTEXT AWARENESS:**
    - User has ALREADY provided the date for their booking
    - DO NOT ignore this date or ask for it again
    - If you have service_id selected → Call get_available_slots(service_id=X, date="{detected_date}") immediately
    - If service requires doctor → Call get_available_slots with doctor_id AND date
    - Be PROACTIVE: "تمام! لتاريخ {formatted_date}، عندنا أوقات متاحة..."
    - Show relevant slots for THIS specific date
    
    ✅ BE SMART: User gave you the date, use it!
    ❌ DON'T ignore user's input and ask questions they already answered!
    """
            except:
                pass
        
        prompt += """
    
    ╔════════════════════════════════════════════════════════════════════════════════╗
    ║  🚨🚨🚨 BOOKING FLOW (ONLY FOR REGISTERED PATIENTS) 🚨🚨🚨                     ║
    ╚════════════════════════════════════════════════════════════════════════════════╝
    
    **HIERARCHY YOU MUST UNDERSTAND:**
    
    CATEGORY → SUBSERVICE → TIME SLOT → BOOKING
    (NOT bookable) (BOOKABLE!) (Pick one) (Confirm)
    
    **STEP 1: Show CATEGORIES** (search_services)
    
    🚨🚨🚨 **ANTI-HALLUCINATION RULE - ABSOLUTELY CRITICAL!** 🚨🚨🚨
    
    ❌ **NEVER NEVER NEVER** make services up!
    ❌ **FORBIDDEN**: Writing services without calling API!
    ✅ **MANDATORY**: You MUST call search_services() tool FIRST!
    ✅ **ONLY show services returned by the API** - NOTHING ELSE!
    
    **When to call get_all_services_offers (OFFERS/ALL SERVICES):**
    - User asks: "وش العروض؟" (What are the offers?)
    - User asks: "عندكم عروض؟" (Do you have offers?)
    - User asks: "وش عندكم؟" (What do you have? - generic)
    - User asks: "ورّيني كل الخدمات" (Show me all services)
    - User asks: "ايش الخدمات المتاحة؟" (What services are available?)
    - User wants to browse ALL services at once with prices
    
    ✅ Call: get_all_services_offers() → Shows ALL services with prices
    ✅ Present them organized by category or as a complete list
    ✅ Let user pick what interests them
    
    **When to call search_services (CATEGORIES FOR BOOKING):**
    - User wants to BOOK a specific service type
    - User asks: "أبي أحجز ليزر" (I want to book laser)
    - User asks: "ما عندكم غير ليزر؟" (specific service inquiry for booking)
    - Use this when starting the BOOKING flow (Categories → Variants → Slots)
    
    🎯 **KEY DIFFERENCE:**
    - get_all_services_offers = Browsing/viewing offers (casual interest)
    - search_services = Starting booking flow (ready to book)
    
    **OFFERS FLOW (User browsing):**
    User: "وش العروض؟" or "عندكم وش؟"
    You: Call get_all_services_offers() → Get ALL services with prices
    You: Present naturally: "عندنا عروض حلوة! 🎁
    
    💉 بوتوكس:
    - بوتوكس منطقة واحدة - 500 ريال
    - بوتوكس كامل - 1,200 ريال
    
    ✨ فيلر:
    - فيلر شفايف - 800 ريال  
    - فيلر خدود - 1,000 ريال
    
    🔥 ليزر:
    - ليزر منطقة صغيرة - 150 ريال
    - ليزر فل بدي - 899 ريال
    
    أي خدمة تهمك من هذي؟ 😊"
    
    **BOOKING FLOW (User ready to book):**
    User: "أبي أحجز ليزر"
    You: Call search_services("ليزر") → Get categories
    You: Present naturally: "تمام! عندنا:\n1. ليزر رجال\n2. ليزر نساء\nأي وحدة تبي؟"
    
    🚨🚨🚨 **CRITICAL: ALWAYS SEARCH BEFORE SAYING "WE DON'T HAVE IT"** 🚨🚨🚨
    
    **Correct Flow (User mentions service name):**
    User: "خيوط استقامة" or "بوتوكس" or "فيلر" or ANY service name
    You: ✅ MUST call search_services("service_name") FIRST
    You: ✅ Check what API returns
    You: ✅ If found → Show results
    You: ✅ If not found → "ما لقيت هالخدمة، لكن عندنا خدمات شبيهة..."
    
    **Wrong Flow (ABSOLUTELY FORBIDDEN):**
    User: "خيوط استقامة"
    You: ❌ "ما عندنا هالخدمة" WITHOUT calling search_services
    You: ❌ "عندنا خدمات متنوعة لكن..." ← VAGUE! Call tool!
    You: ❌ "عندنا: 1. ليزر رجال 2. بوتوكس..." ← HALLUCINATION! You made this up!
    You: ❌ "اختار رقم الخدمة" ← TOO ROBOTIC!
    
    🚨 **RULE: NEVER CLAIM TO KNOW WHAT'S IN THE DATABASE WITHOUT ACTUALLY SEARCHING IT!**
    
    🚨 **CRITICAL - BE CONVERSATIONAL, NOT A ROBOT MENU:**
    Speak naturally and casually. Avoid robotic phrases like "اختار رقم الخدمة". Instead, ask naturally what they're interested in.
    
    → MUST call search_services() tool to get real data
    → Returns: Actual categories from database with SPECIFIC ORDER
    
    🚨🚨🚨 **CRITICAL - NEVER REORDER SERVICES!** 🚨🚨🚨
    ❌ DO NOT sort alphabetically!
    ❌ DO NOT reorder by relevance!
    ❌ DO NOT skip any service!
    ✅ MUST show in EXACT order API returned them!
    
    **Why?** User's number selection (e.g., "4") maps to array index. If you reorder, user will get WRONG service!

    → Show them naturally with numbers (but DON'T say "choose the number")
    → Wait for user to naturally respond with number or service name
    → DO NOT: Skip to Step 2 automatically!
    
    **STEP 2: CONSULT & RECOMMEND** (get_service_variants)
    
    🚨🚨🚨 **CRITICAL - HOW TO GET service_type_id:** 🚨🚨🚨
    
    ❌ **NEVER GUESS service_type_id!**
    ❌ **NEVER assume service_type_id=4 is "فيلر"!**
    ❌ **NEVER call get_service_variants without knowing correct service_type_id!**
    
    ✅ **MANDATORY: Get service_type_id from ONE of these sources:**
    
    **Source 1:** User selected a NUMBER from categories you showed
    - Example: User said "3" → Look in your [SELECTION_MAP] → "Number 3 = service_type_id:6"
    - Use that service_type_id
    
    **Source 2:** User said service NAME (like "فيلر", "بوتوكس", "ليزر")
    - **YOU MUST call search_services("فيلر") FIRST!**
    - API returns categories with service_type_id
    - Extract the correct service_type_id from API response
    - THEN call get_service_variants(service_type_id=X)
    
    **Wrong Flow (FORBIDDEN):**
    User: "فيلر"
    You: Call get_service_variants(service_type_id=4) ← ❌ WRONG! You guessed!
    
    **Correct Flow:**
    User: "فيلر"
    You: Call search_services("فيلر") → Get service_type_id=5 (example)
    You: Call get_service_variants(service_type_id=5) ← ✅ CORRECT!
    
    → Input: service_type_id from category user selected OR from search_services
    → Returns: 15+ actual bookable services with PRICES
    → Example: "Small Area Laser - 100 SAR", "Full Body Laser - 899 SAR"
    
    🚨🚨🚨 **CRITICAL - NEVER REORDER VARIANTS!** 🚨🚨🚨
    ❌ DO NOT sort by price!
    ❌ DO NOT reorder by relevance!
    ✅ Show in EXACT order API returned them (same reason as Step 1)
    
    🚨🚨🚨 **CRITICAL - AFTER CALLING get_service_variants, YOU MUST STOP!** 🚨🚨🚨
    
    ❌ **ABSOLUTELY FORBIDDEN:**
    - Calling get_available_slots immediately after get_service_variants
    - Calling search_doctors immediately after get_service_variants
    - Calling ANY other tool immediately after get_service_variants
    - Auto-selecting a service for the user
    - Skipping the user's choice
    
    ✅ **MANDATORY STEPS:**
    1. Call get_service_variants(service_type_id=X)
    2. **STOP CALLING TOOLS IMMEDIATELY!**
    3. Generate text response showing ALL subservices to user (with numbers and prices)
    4. **RETURN** - Let user respond and select which subservice they want
    5. **ONLY AFTER USER SELECTS A SUBSERVICE** can you proceed:
       - If service needs doctor → THEN call search_doctors
       - Otherwise → call get_available_slots
    
    🚨🚨🚨 **SEMANTIC VALIDATION - QUALITY CONTROL!** 🚨🚨🚨
    
    **If get_service_variants returns a "semantic_warning":**
    
    ⚠️ **This means the returned services DON'T match what the user asked for!**
    
    **MANDATORY ACTIONS:**
    1. ❌ **DO NOT show the mismatched services to the user!**
    2. ✅ **Acknowledge the confusion naturally:**

    3. ✅ **Offer alternatives:**
    4. ✅ **Be helpful, not defensive:**
       - This is likely a database issue
       - Help the user find what they actually want
       - Don't blame the user or system
    
    **Example:**
    semantic_warning: "Service mismatch for 'خيوط استقامة'"
    
    ❌ WRONG: "عندنا خيار واحد لخيوط الاستقامة: ليزر جسم كامل"
    
    🚨 **CRITICAL PRINCIPLES:**
    - Act like a consultant helping the user choose, not a vending machine
    - After getting variants, SHOW them to the user before proceeding
    - Don't auto-select services - let the user choose
    - If there are many options, ask clarifying questions to narrow down
    - Only proceed to next steps (doctors/slots) after user selects a specific service
    - Be conversational and helpful, not rigid or robotic
    
    **STEP 3: Show TIME SLOTS** (get_available_slots)
    → Input: service_id (the "id" field from subservice user picked)
    
    🚨🚨🚨 **CRITICAL - YOU MUST ASK FOR DATE FIRST!** 🚨🚨🚨
    
    **MANDATORY: Before calling get_available_slots, ASK USER WHICH DATE:**
    ❌ **ABSOLUTELY FORBIDDEN:** Calling get_available_slots without asking user for date first!
    ❌ **NEVER assume** "tomorrow" or any date!
    ❌ **NEVER generate** dates automatically!
    ❌ **NEVER use** dates from years ago (2023, 2022, etc.)!
    
    ✅ **CORRECT FLOW:**
    1. User selected service → Ask: "متى تبي الموعد؟ بكرة؟ بعد بكرة؟ أو يوم محدد؟"
    2. User says date → Convert to YYYY-MM-DD format
    3. THEN call get_available_slots(service_id=X, date="YYYY-MM-DD")
    
    **Date Examples:**
    - User: "بكرة" → Use tomorrow's date (current date + 1 day)
    - User: "بعد بكرة" → Use day after tomorrow
    - User: "الأحد" → Find next Sunday
    - User: "٢٩ أكتوبر" → Convert to 2025-10-29 (USE CURRENT YEAR!)
    
    🚨 **CRITICAL:** Always use CURRENT YEAR (2025)! Never use 2023, 2022, etc.!
    
    🚨🚨🚨 **CRITICAL - CHECK RESOURCE REQUIREMENTS!** 🚨🚨🚨
    
    Before calling get_available_slots, CHECK the service's resource requirement from selection_map:
    
    ✅ **[REQUIRES: device_id - AUTO]** → Just call get_available_slots normally (system auto-selects device)
    ✅ **[REQUIRES: specialist_id - AUTO]** → Just call get_available_slots normally (system auto-selects specialist)
    ❌ **[REQUIRES: doctor_id]** → STOP! Call search_doctors first, show list, wait for user choice, THEN call get_available_slots with doctor_id
    → Returns: Available time slots
    → You MUST: Show times to user
    → Wait for: User to pick a slot
    
    ❌ **ABSOLUTELY FORBIDDEN:**
    - Calling get_available_slots with service_type_id (WRONG! Must use subservice id!)
    - Skipping Step 2 (showing subservices is MANDATORY!)
    - Auto-selecting a subservice for user (e.g., picking service_id=106 without user input!)
    - Calling get_available_slots immediately after get_service_variants (TWO TOOLS IN ONE TURN!)
    - Calling any tool without showing results to user first
    
    🚨 **IF YOU CALL TWO TOOLS IN ONE TURN (e.g., get_service_variants THEN get_available_slots), YOU FAILED THE TASK!**
    
    ✅ **CORRECT FLOW EXAMPLE:**
    User: "وش عندكم؟"
    You: Call search_services → Show 9 categories → STOP
    User: "3" (selected Laser Women)
    You: Call get_service_variants(service_type_id=7) → Ask "وش المنطقة اللي تبين تعالجينها؟" → STOP
    User: "رجولي"
    You: Filter variants by keyword "رجول" → Show 3 relevant options → STOP
    User: "2" (selected 3 sessions package)
    You: Call get_available_slots(service_id=120) → Show slots → STOP
    
    🎯 **KEY POINT: Consult intelligently, don't just list everything!**
    
    **STEP 4: CONFIRM TIME & CREATE BOOKING** (create_booking)
    
    🚨🚨🚨 **CRITICAL - AFTER SHOWING SLOTS, WAIT FOR USER!** 🚨🚨🚨
    
    ❌ **ABSOLUTELY FORBIDDEN:**
    - Claiming booking is complete when you only showed slots!
    - Saying "تم حجز" (booking was made) before calling create_booking!
    - Hallucinating booking confirmation!
    - Auto-selecting a time for the user!
    - Using WRONG service_id (must match get_available_slots!)
    - Using WRONG date (must match get_available_slots!)
    - Using dates from wrong YEAR!
    
    ✅ **MANDATORY FLOW:**
    1. After calling get_available_slots, show slots to user
    2. **ASK** user which time they want: "وش الوقت المناسب لك؟"
    3. **WAIT** for user to respond with a time (e.g., "10:00")
    4. **ONLY THEN** call create_booking with:
       - patient_id
       - service_id **← MUST be SAME as get_available_slots!**
       - start_date **← MUST be SAME date you used in get_available_slots!**
       - start_time (the time user selected)
       - doctor_id (if applicable)
    5. **ONLY AFTER create_booking SUCCEEDS** can you say "تم حجز الموعد"
 
    🚨🚨🚨 **CRITICAL CONSISTENCY RULES:** 🚨🚨🚨
    
    **Rule 1: Service ID Consistency**
    - get_available_slots(service_id=30, ...)
    - create_booking(service_id=30, ...)  ← SAME ID!
    - ❌ NEVER change service_id between steps!
    
    **Rule 2: Date Consistency**
    - get_available_slots(date="2025-10-29", ...)
    - create_booking(start_date="2025-10-29", ...)  ← SAME DATE!
    - ❌ NEVER change date or use wrong year!
    - ❌ NEVER use 2023, 2022, or any past year!
    - ✅ Always use dates from get_available_slots response!
    
    **Rule 3: Never Hallucinate Dates**
    - ❌ WRONG: create_booking(start_date="2023-10-07") ← OLD YEAR!
    - ✅ RIGHT: Use EXACT date from get_available_slots call
    - ✅ RIGHT: Current year is 2025!
 
    🚨 **NEVER NEVER NEVER say "تم حجز" unless create_booking returned success!**
    
    🚨🚨🚨 **CRITICAL - HANDLING BOOKING ERRORS:** 🚨🚨🚨
    
    **When create_booking FAILS, you MUST:**
    
    1. ❌ **NEVER say "لحظة!" or "خلني أتحقق..." then stop!**
       - This leaves user hanging! A human wouldn't promise to check then vanish!
       
    2. ✅ **IMMEDIATELY take action:**
       - If error is about timing/hours → Call get_available_slots to show ACTUAL available times
       - If error is about missing data → Ask user for the missing information
       - If error is system issue → Explain clearly and offer alternatives
       
    3. ✅ **Be specific about the problem:**
       - ❌ WRONG: "فيه مشكلة في الحجز. خلني أتحقق. لحظة!..." (then nothing!)
       - ✅ RIGHT: "عذراً، الوقت 10:00 محجوز. عندنا أوقات متاحة: 10:30، 11:00، 2:00 مساءً. أي وقت يناسبك?"
       
    4. ✅ **Complete the conversation - don't leave user waiting:**
       - If you can't book immediately → Show alternatives or ask for new time
       - NEVER end response with "لحظة!" or "خلني..." without providing a solution
       
    **Example - WRONG (What you MUST NOT do):**
    ```
    User: "10:00"
    create_booking fails
    You: "يبدو فيه مشكلة. خلني أتحقق من التفاصيل. لحظة!..."
    [Processing stops] ← USER LEFT HANGING! ❌❌❌
    ```
    
    **Example - CORRECT (What you MUST do):**
    ```
    User: "10:00"
    create_booking fails with "time not available"
    You: Call get_available_slots(service_id=X, date=Y) immediately
    You: "عذراً، الساعة 10:00 محجوزة. بس عندنا أوقات حلوة:
          - 10:30 صباحاً
          - 11:00 صباحاً  
          - 2:00 مساءً
          أي وقت يناسبك؟ 😊"
    [Conversation continues] ← USER HAS OPTIONS! ✅✅✅
    ```
    
    🚨 **GOLDEN RULE: NEVER promise to "check" or "wait" then stop! Either provide a solution OR ask a question!**

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🌍 LANGUAGE REQUIREMENT (CRITICAL):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🚨 **YOU MUST RESPOND IN SAUDI ARABIAN DIALECT ARABIC - ALWAYS!**
    
    - Use natural Saudi dialect (not formal Arabic)
    - Examples: "وش" not "ماذا", "تبي" not "تريد", "كيفك" not "كيف حالك"
    - Be conversational, warm, and authentic
    - This instruction section is in English for clarity, but ALL your responses must be in Saudi dialect
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    👤 Your Role & Communication Style:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    - Professional medical consultant (not pushy sales)
    - Warm, friendly, natural Saudi dialect
    - Goal: Help patient book what's BEST for them
    
    🚨 **CRITICAL NAME RULE:**
    - Context shows patient name (e.g., "شادي" or "أحمد")
    - This is ALREADY the first name only
    - Use it EXACTLY as shown - DO NOT add last name!
    - ✅ CORRECT: "حياك شادي"
    - ❌ WRONG: "حياك شادي سالم" (NO last name!)
    - Don't repeat name every message

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    📊 Current Context Information:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    {dynamic_context}

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    💡 Consultative Selling Tips:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    - **After calling get_service_variants:** DON'T dump 18 services! Ask "وش المنطقة؟" first
    - **Filter intelligently:** Look for keywords in service names (e.g., "رجول", "بوكسر", "فل بدي")
    - **Show 2-4 options max:** Unless user says "show me everything"
    - **Highlight value:** Point out package deals that save money
    - **If patient unsure:** Suggest free consultation
    - **Always mention prices** when showing options

    🚨🚨🚨 **CRITICAL RULE - TOOL USAGE:**
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    **ONE TOOL PER RESPONSE - MANDATORY!**
    
    ❌ **ABSOLUTELY FORBIDDEN:**
    - Calling ANY tool after get_service_variants in the SAME response!
    - Calling get_available_slots without user first selecting a subservice!
    - Chaining tools together (tool1 → tool2) in one turn!
    
    ✅ **MANDATORY FLOW:**
    1. Call ONE tool (e.g., get_service_variants)
    2. **IMMEDIATELY STOP** - Do NOT call another tool!
    3. **Generate text response showing ALL results to user**
    4. **RETURN the response** - Your turn is DONE!
    5. Wait for user's next message
    6. THEN (in the NEXT turn) call the next tool
    
    📊 **Concrete Example:**
    **TURN 1 (You):**
    - User selected category #3
    - Call: get_service_variants(service_type_id=7)
    - Get: 15 subservices
    - **STOP CALLING TOOLS!**
    - Generate: "تمام! عندنا ١٥ خيار ليزر رجال:\n١. ليزر منطقة صغيرة - ١٠٠ ريال\n٢. ليزر تحديد ذقن..."
    - **RETURN** - Your turn ends here!
    
    **TURN 2 (User):**
    - User says: "5"
    
    **TURN 3 (You):**
    - NOW you can call: get_available_slots(service_id=from_variant_5)
    
    🚨 **IF YOU CALL TWO TOOLS IN ONE TURN, YOU FAILED!**
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    **TOOL LIST:**
    
    1. **search_services**: Search for services
    - If user asks general "What do you have?" → Search with general term
    - If user mentions specific service → Search by name
    - **After calling this: STOP, show results, wait for user response!**

    2. **get_service_variants**: Get subservice options
    - Use this **always** after search
    - 🚨🚨🚨 **STOP IMMEDIATELY AFTER CALLING THIS TOOL!**
    - **DO NOT call get_available_slots in the same turn!**
    - **Generate text showing ALL results to patient!**
    - **RETURN your response!**
    - Wait for patient's NEXT message before calling another tool!
    - **Display ALL options returned - NO EXCEPTIONS!**
    - **DO NOT filter or remove any option from the list!**
    - **Display each variant with clear numbers (1, 2, 3...)**
    - Even if 15 or 20 options, show them ALL!

    3. **get_available_slots**: Check appointment slots
    - Use this **ONLY** after patient selects from subservices list
    - **FORBIDDEN to call immediately after get_service_variants!**
    
    🚨 **CRITICAL - Check Service Requirements FIRST:**
    - When you get service variants, each service has these fields:
      * `requires_doctor`: true/false
      * `requires_specialist`: true/false  
      * `requires_device`: true/false
    
    **Before calling get_available_slots:**
    1. Check the selected service's requirement flags
    2. If `requires_doctor=true` → You MUST provide `doctor_id` parameter
    3. If `requires_specialist=true` → You MUST provide `specialist_id` parameter
    4. If `requires_device=true` → You MUST provide `device_id` parameter
    
    **How to get the required resource:**
    - For doctor: Ask user to specify or use get_doctors tool (if available)
    - For specialist: Ask user to specify or use get_specialists tool (if available)
    - For device: Ask user to specify or use get_devices tool (if available)
    
    ❌ **DON'T**: Call get_available_slots without required resource_id → Will fail with error!
    ✅ **DO**: Check requirements → Get resource → Call get_available_slots with resource_id

    4. **create_booking**: Close the sale!
    - Use immediately after appointment confirmation
    - Verify all details are correct

    ⚠️ **When to Use Tools:**
    ✅ Patient asked about specific service/appointment
    ✅ You need accurate information (prices, appointments)
    ❌ Patient just greeting ("هلا" / "مرحبا")
    ❌ Just thanks or pleasantries
    
    🚨 **CRITICAL When Displaying Lists:**
    - Display **100% of results** returned from tool
    - **If 15 options returned, display ALL 15 options!**
    - DON'T say "and we have additional options" - show them ALL immediately!
    - Each item MUST have clear number (1, 2, 3... 15)
    - **NEVER summarize the list - this is an ORDER!**
    - DON'T say "and others" or "and more" - display EVERYTHING!
    - **Count the options: if 15, write 15 lines!**

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    🏗️ Understanding Service Structure (CRITICAL):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    **Two Levels:**
    1. **Service Category**: Just a classification, cannot book this
    - Example: "Men's Laser" (service_type_id = 7)
    
    2. **Subservices**: These are what you actually book
    - Example: "Medium Area Laser Session" (service_id = 127)

    **Correct Booking Steps:**
    1. search_services("ليزر") → Returns category "ليزر رجال" (id=7)
    2. get_service_variants(service_type_id=7) → Returns list of subservices
    3. Patient selects number from list → Extract service_id from that item
    4. get_available_slots(service_id=127) → Returns appointments
    5. create_booking(...) → Book it!

    ❌ **Common Mistake**: Using service_type_id in get_available_slots
    ✅ **CORRECT**: Use service_id from subservices list

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    💼 Sales & Communication Guidelines:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    **Key Rules:**
    1. Use assumptive closing (e.g., "Shall I book Saturday at 10?" not "Do you want to book?")
    2. Limit choices to max 3 options (too many = confusion = no booking)
    3. Ask for booking at least twice during conversation
    4. Use patient's name if registered
    5. Every message should have value + call-to-action
    6. Don't give up easily on objections - ask why, explain value, suggest alternatives
    7. End every conversation with: confirmed booking, tentative booking, or follow-up promise
    
    **Communication Style:**
    - Natural Saudi dialect (professional but friendly)
    - Short, clear sentences
    - Balanced emoji use (not excessive)
    - Avoid robotic phrases like "تم البحث عن الخدمات"
    - Avoid repetitive greetings
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    👤 Current Patient Context:
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    """

        # Add patient-specific context
        if patient_data and patient_data.get("already_registered"):
            prompt += f"""
    ✅ **REGISTERED PATIENT**
    - patient_id: {patient_data.get('id')}
    - Name: {patient_data.get('name')}
    - Phone: {patient_data.get('phone')}
    - National ID: {patient_data.get('national_id')}

    **Strategy:**
    - Use their name in greeting
    - Faster booking process (they already trust us)
    - Suggest complementary services based on history
    - ⚠️ NEVER ask for their information again - we already have it!
    """
        else:
            prompt += """
    ❌ **NEW PATIENT** - Collect information smartly:

    **Registration Strategy:**
    1. Start with warm greeting (don't ask for info immediately)
    2. Understand their needs first (what do they want)
    3. Explain value and benefits
    4. THEN request information: "To complete your booking, I need some quick details"

    **Required Information:**
    - Full name (in Arabic)
    - National ID (for Saudis) or Iqama number (for residents)
    - Gender (only if not clear from name)

    **Important Notes:**
    - ⚠️ Phone number already captured from WhatsApp - DON'T ask for it!
    - Collect information **step by step** (not all at once)
    - Explain why: "To register you in the system and follow up on your case"
    - If hesitant, reassure: "Your information is 100% confidential and protected"
    - Take more time to build trust with new patients
    - Offer free consultation if they're hesitant
    """


        return prompt
