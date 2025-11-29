"""
LLM Response Generator - Replaces all hardcoded templates with natural LLM responses
"""
from loguru import logger
from typing import Optional, List, Dict


class LLMResponseGenerator:
    """Generate natural, context-aware responses using LLM instead of hardcoded templates"""
    
    def __init__(self, llm_reasoner):
        self.llm = llm_reasoner
    
    async def generate_response(
        self,
        intent: str,
        context: str,
        user_name: str = "عزيزي",
        conversation_history: Optional[List[Dict]] = None,
        extra_data: Optional[Dict] = None
    ) -> str:
        """
        Generate natural response using LLM.
        
        Args:
            intent: What we're asking for (e.g., "ask_for_id", "confirm_booking")
            context: Current situation/context
            user_name: User's name
            conversation_history: Previous messages (optional)
            extra_data: Any additional context (optional)
        
        Returns:
            Natural Arabic response in Saudi dialect
        """
        
        # Build context from extra data
        extra_context = ""
        if extra_data:
            extra_context = "\n".join([f"- {k}: {v}" for k, v in extra_data.items()])
        
        # Build prompt for LLM
        extra_info = f"**معلومات إضافية:**\n{extra_context}" if extra_context else ""
        
        prompt = f"""أنت مساعد ذكي لعيادة طبية. مهمتك توليد رد طبيعي وودود باللهجة السعودية.

**السياق:** {context}

**الهدف:** {intent}

**اسم المستخدم:** {user_name}

{extra_info}

**التعليمات:**
- استخدم لهجة سعودية طبيعية وودودة
- كن مختصراً (جملة إلى 3 جمل كحد أقصى)
- استخدم الإيموجي بشكل خفيف ومناسب
- كن دافئ ومهني بنفس الوقت
- لا تكرر المعلومات
- لا تضيف معلومات غير مطلوبة
- ⛔ ممنوع استخدام عبارات مثل: "وش الخدمة اللي تبغاها" أو "وش ودك اليوم" أو أي عبارة مكررة
- ⛔ كل رد يجب أن يكون فريد ومبني على السياق الفعلي
- تحدث بشكل طبيعي كأنك إنسان حقيقي

**أرسل الرد فقط بدون مقدمات:**"""

        # Get LLM response
        try:
            # Use chat_completion with simple user message
            messages = [
                {"role": "system", "content": "أنت مساعد ذكي لعيادة طبية تتحدث بلهجة سعودية طبيعية"},
                {"role": "user", "content": prompt}
            ]
            
            result = await self.llm.chat_completion(
                messages=messages,
                temperature=0.8
            )
            
            response = result.get("content", "")
            
            # Clean response (remove quotes if LLM added them)
            response = response.strip().strip('"').strip("'")
            
            logger.info(f"🤖 [LLM_RESPONSE] Intent: {intent}, Generated: {response[:80]}...")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ [LLM_RESPONSE] Error generating response: {e}")
            # Fallback to basic template
            return self._fallback_template(intent, user_name)
    
    def _fallback_template(self, intent: str, user_name: str) -> str:
        """Fallback templates if LLM fails - varied responses to avoid being robotic"""
        import random
        
        fallbacks = {
            "ask_for_name": [
                f"مرحباً! ممكن تعطيني اسمك الكامل؟",
                f"أهلاً! شو اسمك؟",
                f"حياك الله! ممكن اسمك؟"
            ],
            "ask_for_national_id": [
                f"ممتاز {user_name}! الحين محتاج رقم الهوية (10 أرقام)",
                f"تمام {user_name}! ممكن رقم هويتك؟",
                f"زين {user_name}! ابغى رقم الهوية الوطنية"
            ],
            "confirm_registration": [
                f"تمام {user_name}! تم التسجيل ✅",
                f"ممتاز {user_name}! سجلناك بنجاح",
                f"حلو {user_name}! خلصنا التسجيل"
            ],
            "ask_for_service": [
                f"طيب {user_name}، ايش الخدمة اللي تحتاجها؟",
                f"{user_name}، قولي ايش تبي بالضبط؟",
                f"تمام {user_name}، ايش نوع الخدمة؟"
            ],
        }
        
        options = fallbacks.get(intent, [f"تمام {user_name}، كيف أساعدك؟"])
        return random.choice(options)
    
    # =================================================================
    # Specialized Response Generators
    # =================================================================
    
    async def ask_registration_confirmation(
        self,
        user_name: str = "حبيبنا",
        service: Optional[str] = None
    ) -> str:
        """Ask user to confirm registration (patient not found in system)"""
        context = f"لم نجد حساب مسجل للمستخدم {user_name}. اسأل بشكل ودود إذا يريد تسجيل حساب جديد"
        if service:
            context += f". المستخدم كان يسأل عن خدمة: {service}"
        
        return await self.generate_response(
            intent="ask_registration_confirmation",
            context=context,
            user_name=user_name,
            extra_data={"service": service} if service else None
        )
    
    async def ask_for_name(
        self,
        user_name: str = "حبيبنا",
        phone_display: Optional[str] = None
    ) -> str:
        """Ask user for their full name"""
        context = "نحتاج اسم المستخدم الكامل لبدء التسجيل"
        if phone_display:
            context += f". رقم الجوال المسجل: {phone_display}"
        
        return await self.generate_response(
            intent="ask_for_name",
            context=context,
            user_name=user_name,
            extra_data={"phone": phone_display} if phone_display else None
        )
    
    async def ask_for_national_id(self, user_name: str) -> str:
        """Ask user for national ID after getting name"""
        return await self.generate_response(
            intent="ask_for_national_id",
            context=f"المستخدم {user_name} قدم اسمه. الحين نحتاج رقم الهوية الوطنية (10 أرقام)",
            user_name=user_name
        )
    
    async def confirm_registration(self, user_name: str, service: Optional[str] = None) -> str:
        """Confirm successful registration"""
        context = f"التسجيل اكتمل بنجاح للمستخدم {user_name}"
        if service:
            context += f". الخدمة المطلوبة: {service}"
        
        return await self.generate_response(
            intent="confirm_registration",
            context=context,
            user_name=user_name,
            extra_data={"service": service} if service else None
        )
    
    async def handle_invalid_id(
        self,
        user_name: str,
        provided_value: str,
        reason: str
    ) -> str:
        """Handle invalid national ID"""
        return await self.generate_response(
            intent="handle_invalid_id",
            context=f"المستخدم أرسل رقم هوية غير صحيح. السبب: {reason}. اطلب منه المحاولة مرة أخرى بالصيغة الصحيحة (10 أرقام)",
            user_name=user_name,
            extra_data={
                "provided_value": provided_value,
                "reason": reason
            }
        )
    
    async def handle_invalid_name(self, user_name: str, reason: str) -> str:
        """Handle invalid name"""
        return await self.generate_response(
            intent="handle_invalid_name",
            context=f"الاسم المدخل غير صحيح. السبب: {reason}. اطلب من المستخدم إدخال الاسم الكامل بالعربي",
            user_name=user_name,
            extra_data={"reason": reason}
        )
    
    async def ask_for_service(
        self,
        user_name: str,
        services: Optional[List[str]] = None
    ) -> str:
        """Ask user which service they want"""
        context = "المستخدم يريد حجز موعد لكن لم يختر خدمة بعد"
        
        if services:
            context += f". الخدمات المتاحة: {', '.join(services)}"
        else:
            context += ". الخدمات المتاحة: ليزر، فيلر، بوتكس، تنظيف البشرة"
        
        return await self.generate_response(
            intent="ask_for_service",
            context=context,
            user_name=user_name
        )
    
    async def handle_cancellation(self, user_name: str) -> str:
        """Handle user cancellation"""
        return await self.generate_response(
            intent="handle_cancellation",
            context="المستخدم يريد إلغاء العملية الحالية. أكد الإلغاء بشكل ودود وأخبره أنك جاهز للمساعدة لاحقاً",
            user_name=user_name
        )
    
    async def present_time_slots(
        self,
        user_name: str,
        slots: List[str]
    ) -> str:
        """Present available time slots to user"""
        slots_text = "\n".join(slots)
        context = f"عرض المواعيد المتاحة للمستخدم {user_name}. المواعيد:\n{slots_text}\nاطلب منه اختيار رقم الموعد المناسب"
        
        return await self.generate_response(
            intent="present_time_slots",
            context=context,
            user_name=user_name,
            extra_data={"slots": slots}
        )
    
    async def handle_no_slots_available(self, user_name: str) -> str:
        """Handle no available slots"""
        return await self.generate_response(
            intent="handle_no_slots",
            context=f"لا توجد مواعيد متاحة حالياً. اقترح على {user_name} تجربة خدمة أخرى أو طبيب آخر بشكل ودود",
            user_name=user_name
        )
    
    async def handle_registration_error_recovery(self, user_name: str) -> str:
        """Handle registration error and offer recovery options"""
        return await self.generate_response(
            intent="handle_registration_error",
            context=f"حدث خطأ في التسجيل. اعتذر لـ{user_name} واعرض خيارات: حجز جديد، مشاهدة المواعيد، أو الاتصال بالعيادة",
            user_name=user_name
        )
    
    async def handle_catastrophic_error(self, user_name: str) -> str:
        """Handle catastrophic system failure (3+ consecutive errors)"""
        return await self.generate_response(
            intent="handle_catastrophic_error",
            context=f"حدثت أخطاء متعددة في النظام. اعتذر بشدة لـ{user_name} واطلب منه الاتصال مباشرة برقم 920033304",
            user_name=user_name
        )
    
    async def request_booking_confirmation(
        self,
        user_name: str,
        booking_details: Dict[str, Any]
    ) -> str:
        """Request user confirmation for booking"""
        details_text = f"الخدمة: {booking_details['service']}, الطبيب: {booking_details['doctor']}, التاريخ: {booking_details['date']}, الوقت: {booking_details['time']}, السعر: {booking_details['price']} ريال"
        context = f"عرض تفاصيل الحجز على {user_name} واطلب منه تأكيد الحجز. التفاصيل: {details_text}"
        
        return await self.generate_response(
            intent="request_booking_confirmation",
            context=context,
            user_name=user_name,
            extra_data=booking_details
        )
    
    async def confirm_booking_success(
        self,
        user_name: str,
        booking_info: Dict[str, Any]
    ) -> str:
        """Confirm successful booking creation"""
        info_text = f"رقم الحجز: {booking_info['booking_id']}, رمز التأكيد: {booking_info['confirmation_code']}"
        context = f"تهنئة {user_name} بنجاح الحجز واعطائه التفاصيل. {info_text}. الخدمة: {booking_info['service']}, الطبيب: {booking_info['doctor']}, التاريخ: {booking_info['date']}, الوقت: {booking_info['time']}"
        
        return await self.generate_response(
            intent="confirm_booking_success",
            context=context,
            user_name=user_name,
            extra_data=booking_info
        )
    
    async def present_services_list(
        self,
        user_name: str,
        services: List[str],
        service_type: Optional[str] = None
    ) -> str:
        """Present list of available services"""
        services_text = "\n".join(services)
        type_msg = f"من نوع {service_type}" if service_type else ""
        context = f"عرض قائمة الخدمات المتاحة {type_msg} لـ{user_name}. الخدمات:\n{services_text}\nاطلب منه اختيار الخدمة المناسبة"
        
        return await self.generate_response(
            intent="present_services_list",
            context=context,
            user_name=user_name,
            extra_data={"services": services, "type": service_type}
        )
    
    async def handle_no_services_available(
        self,
        user_name: str,
        service_type: Optional[str] = None
    ) -> str:
        """Handle no services available"""
        type_msg = f"من نوع {service_type}" if service_type else ""
        return await self.generate_response(
            intent="handle_no_services",
            context=f"لا توجد خدمات متاحة {type_msg}. اعتذر لـ{user_name} واقترح الاتصال أو المحاولة لاحقاً",
            user_name=user_name
        )
    
    async def present_service_types_list(
        self,
        user_name: str,
        service_types: List[str]
    ) -> str:
        """Present list of service types"""
        types_text = "\n".join(service_types)
        context = f"عرض أنواع الخدمات المتاحة لـ{user_name}. الأنواع:\n{types_text}\nاطلب منه اختيار النوع المناسب"
        
        return await self.generate_response(
            intent="present_service_types",
            context=context,
            user_name=user_name,
            extra_data={"types": service_types}
        )
    
    async def handle_no_service_types_available(self, user_name: str) -> str:
        """Handle no service types available"""
        return await self.generate_response(
            intent="handle_no_service_types",
            context=f"لا توجد أنواع خدمات متاحة حالياً. اعتذر لـ{user_name} واقترح الاتصال أو المحاولة لاحقاً",
            user_name=user_name
        )
    
    async def handle_user_stuck_in_loop(self, user_name: str) -> str:
        """Handle user stuck in conversation loop"""
        return await self.generate_response(
            intent="handle_loop",
            context=f"{user_name} عالق في حلقة تكرار. قدم مساعدة واضحة: خيارات الحجز، عرض المواعيد، أو رقم الاتصال 920033304",
            user_name=user_name
        )
    
    async def handle_error(
        self,
        user_name: str,
        error_type: str,
        can_retry: bool = True
    ) -> str:
        """Handle system errors"""
        context = f"حدث خطأ في النظام: {error_type}"
        if can_retry:
            context += ". اطلب من المستخدم المحاولة مرة أخرى"
        else:
            context += ". اعتذر واطلب منه الاتصال مباشرة"
        
        return await self.generate_response(
            intent="handle_error",
            context=context,
            user_name=user_name,
            extra_data={
                "error_type": error_type,
                "can_retry": can_retry
            }
        )
    
    async def show_services(
        self,
        user_name: str,
        services: List[Dict],
        context_info: Optional[str] = None
    ) -> str:
        """Show available services"""
        service_list = [f"{s.get('name', 'خدمة')}" for s in services]
        
        context = f"عرض الخدمات المتاحة للمستخدم"
        if context_info:
            context += f". السياق: {context_info}"
        
        return await self.generate_response(
            intent="show_services",
            context=context,
            user_name=user_name,
            extra_data={"services": ", ".join(service_list)}
        )
    
    async def confirm_booking(
        self,
        user_name: str,
        service: str,
        date: str,
        time: str
    ) -> str:
        """Confirm booking details"""
        return await self.generate_response(
            intent="confirm_booking",
            context=f"تأكيد تفاصيل الحجز للمستخدم {user_name}",
            user_name=user_name,
            extra_data={
                "service": service,
                "date": date,
                "time": time
            }
        )


# Singleton instance
_response_generator = None


def get_response_generator():
    """Get or create response generator instance"""
    global _response_generator
    
    if _response_generator is None:
        from .llm_reasoner import get_llm_reasoner
        llm = get_llm_reasoner()
        _response_generator = LLMResponseGenerator(llm)
    
    return _response_generator
