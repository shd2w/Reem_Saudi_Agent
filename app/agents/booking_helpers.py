"""
Booking Agent Helper Methods
Time slot selection and confirmation logic
"""

from loguru import logger
from typing import Dict, List, Optional
from datetime import datetime, timedelta


async def show_available_time_slots(
    api_client,
    booking_state: Dict,
    sender_name: str
) -> Dict:
    """
    Fetch and display available time slots for selected service and doctor.
    
    Returns formatted response with numbered time slot options.
    """
    try:
        service_id = booking_state.get("service_id")
        resource_type = booking_state.get("resource_type", "doctor")  # Default to doctor for backward compatibility
        
        # Get the appropriate resource ID based on type
        doctor_id = booking_state.get("doctor_id")
        specialist_id = booking_state.get("specialist_id")
        device_id = booking_state.get("device_id")
        
        logger.info(f"🔍 Fetching slots for service={service_id}, resource_type={resource_type}")
        logger.info(f"📋 DEBUG booking_state IDs: doctor_id={doctor_id}, specialist_id={specialist_id}, device_id={device_id}")
        
        # Build params based on resource type
        params = {
            "service_id": service_id,
            "date": datetime.now().strftime("%Y-%m-%d")
        }
        
        if resource_type == "doctor" and doctor_id:
            params["doctor_id"] = doctor_id
            logger.info(f"📋 Using doctor_id={doctor_id}")
        elif resource_type == "specialist" and specialist_id:
            params["specialist_id"] = specialist_id
            logger.info(f"📋 Using specialist_id={specialist_id}")
        elif resource_type == "device" and device_id:
            params["device_id"] = device_id
            logger.info(f"📋 Using device_id={device_id}")
        else:
            logger.warning(f"⚠️ No resource ID found for type {resource_type}")
            logger.warning(f"⚠️ DEBUG: doctor_id={doctor_id}, specialist_id={specialist_id}, device_id={device_id}")
        
        slots_result = await api_client.get("/slots", params=params)
        slots = slots_result.get("data") or slots_result.get("slots") or []
        
        logger.info(f"✅ Fetched {len(slots)} available time slots")
        
        if not slots:
            return {
                "response": f"يا عيني يا {sender_name} 😅\nما في مواعيد فاضية الحين\nتبغى تجرب خدمة ثانية؟ أو دكتور ثاني؟",
                "intent": "booking",
                "status": "no_slots_available"
            }
        
        # Format slots with numbers
        slots_text = "\n".join([
            f"*{i+1}.* {slot.get('date')} - {slot.get('time')}"
            for i, slot in enumerate(slots[:10])
        ])
        
        # Save slots to booking state
        booking_state["available_slots"] = slots[:10]
        booking_state["step"] = "time_selection"
        
        response = f"""تمام يا {sender_name}! 📅
عندنا هالمواعيد الفاضية:

{slots_text}

وش رقم الموعد اللي يناسبك؟ (مثلاً: 1 أو 2)"""
        
        return {
            "response": response,
            "intent": "booking",
            "status": "showing_time_slots"
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching time slots: {e}")
        return {
            "response": f"يا عيني يا {sender_name} 😅\nشكله فيه مشكلة بسيطة\nعطِني لحظة وجرب مرة ثانية 🙏",
            "intent": "booking",
            "status": "error"
        }


async def request_booking_confirmation(
    booking_state: Dict,
    sender_name: str,
    api_client
) -> Dict:
    """
    Show booking summary and request user confirmation.
    
    Displays: Service, Doctor, Date, Time, Price, Location
    Asks: نعم للتأكيد أو لا للإلغاء
    """
    try:
        service_name = booking_state.get("service_name", "الخدمة")
        doctor_name = booking_state.get("doctor_name", "الدكتور")
        date = booking_state.get("preferred_date", "التاريخ")
        time = booking_state.get("preferred_time", "الوقت")
        
        # Fetch price from service details
        service_id = booking_state.get("service_id")
        price = "حسب الاستشارة"
        
        if service_id:
            try:
                service_details = await api_client.get(f"/services/{service_id}")
                price = service_details.get("price", "حسب الاستشارة")
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch service price: {e}")
        
        # Mark as awaiting confirmation
        booking_state["awaiting_confirmation"] = True
        booking_state["step"] = "awaiting_confirmation"
        
        # Prefer Arabic names for Saudi context
        service_display = booking_state.get("service_name_ar") or service_name
        doctor_display = booking_state.get("doctor_name") or doctor_name  # Already Arabic from extraction
        
        confirmation_text = f"""📋 *تأكيد الحجز*

يا {sender_name}، شيّك على تفاصيل حجزك:

🏥 *الخدمة:* {service_display}
👨‍⚕️ *الدكتور:* د. {doctor_display}
📅 *التاريخ:* {date}
🕐 *الوقت:* {time}
💰 *السعر:* {price} ريال
📍 *الموقع:* مركز وجن الطبي

✅ *للتأكيد:* اكتب "نعم" أو "تأكيد"
❌ *للإلغاء:* اكتب "لا" أو "إلغاء"

تبغى تأكد الحجز؟"""
        
        logger.info(f"📋 Confirmation request sent - awaiting user response")
        
        return {
            "response": confirmation_text,
            "intent": "booking",
            "status": "awaiting_confirmation"
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating confirmation: {e}")
        return {
            "response": f"يا عيني يا {sender_name} 😅\nشكله فيه مشكلة بسيطة\nعطِني لحظة وجرب مرة ثانية 🙏",
            "intent": "booking",
            "status": "error"
        }


async def complete_booking_with_details(
    api_client,
    booking_state: Dict,
    phone_number: str,
    sender_name: str
) -> Dict:
    """
    Complete the booking and return confirmation with booking ID.
    
    Creates booking via API and returns:
    - Booking ID
    - Confirmation number
    - All booking details
    - Cancellation instructions
    """
    try:
        # CRITICAL VALIDATION: Ensure all required info is present BEFORE creating booking
        required_fields = {
            "service_id": booking_state.get("service_id"),
            "service_name": booking_state.get("service_name"),
            "doctor_id": booking_state.get("doctor_id"),
            "preferred_date": booking_state.get("preferred_date"),
            "preferred_time": booking_state.get("preferred_time")
        }
        
        missing_fields = [k for k, v in required_fields.items() if not v]
        
        if missing_fields:
            logger.error(f"🚫 BOOKING VALIDATION FAILED: Missing {missing_fields}")
            logger.error(f"🚫 State: service✗={bool(booking_state.get('service_id'))}, doctor✗={bool(booking_state.get('doctor_id'))}, date✗={bool(booking_state.get('preferred_date'))}, time✗={bool(booking_state.get('preferred_time'))}")
            
            # DO NOT create fake booking - return error
            return {
                "response": f"لحظة شوي يا {sender_name} 😅\nينقصني شوية معلومات: {', '.join(missing_fields)}\nتقدر تعطيني هالمعلومات؟",
                "intent": "booking",
                "status": "validation_failed",
                "missing_fields": missing_fields
            }
        
        logger.info(f"✅ BOOKING VALIDATION PASSED: All required fields present")
        logger.info("🎯 Creating booking via API...")
        
        # Prepare booking data
        booking_data = {
            "patient_phone": phone_number,
            "service_id": booking_state.get("service_id"),
            "doctor_id": booking_state.get("doctor_id"),
            "appointment_date": booking_state.get("preferred_date"),
            "appointment_time": booking_state.get("preferred_time"),
            "notes": f"Booked via WhatsApp by {sender_name}"
        }
        
        # Create booking
        result = await api_client.post("/appointments", data=booking_data)
        
        booking_id = result.get("id") or result.get("booking_id") or "N/A"
        confirmation_code = result.get("confirmation_code") or f"WJ{booking_id}"
        
        logger.info(f"✅ Booking created successfully - ID: {booking_id}")
        
        # Clear booking state
        booking_state.clear()
        booking_state["started"] = False
        booking_state["last_booking_id"] = booking_id
        
        service_name = result.get("service_name") or booking_data.get("service_name", "الخدمة")
        doctor_name = result.get("doctor_name") or booking_data.get("doctor_name", "الدكتور")
        date = booking_data.get("appointment_date")
        time = booking_data.get("appointment_time")
        
        success_message = f"""✅ *تمّ الحجز!*

يا {sender_name}، حجزك جاهز! 🎉

📋 *رقم الحجز:* #{booking_id}
🔢 *رمز التأكيد:* {confirmation_code}

📌 *تفاصيل الموعد:*
🏥 الخدمة: {service_name}
👨‍⚕️ الدكتور: {doctor_name}
📅 التاريخ: {date}
🕐 الوقت: {time}
📍 الموقع: مركز وجن الطبي

📱 *ملاحظات مهمة:*
• احتفظ برقم الحجز عشان المراجعة
• تعال قبل الموعد بـ 10 دقائق
• للإلغاء: أرسل "إلغاء #{booking_id}"

نورتنا! نشوفك على خير 🤍"""
        
        return {
            "response": success_message,
            "intent": "booking",
            "status": "completed",
            "booking_id": booking_id,
            "confirmation_code": confirmation_code
        }
        
    except Exception as e:
        logger.error(f"❌ Booking creation failed: {e}")
        
        return {
            "response": f"""يا عيني يا {sender_name} 😔\nشكله فيه مشكلة بسيطة بالحجز

جرب مرة ثانية أو تواصل معانا:
📞 هاتف: 920000000

معذرة على الإزعاج 🙏""",
            "intent": "booking",
            "status": "error",
            "error": str(e)
        }
