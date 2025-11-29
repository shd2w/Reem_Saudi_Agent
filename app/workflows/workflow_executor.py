"""
Workflow Executor
=================
Executes LangGraph workflows and returns structured results.

NO conversation generation - only technical operations!
"""
from typing import Dict, Any, Optional
from loguru import logger

from ..models.workflow_result import WorkflowResult, WorkflowStatus
from ..api.agent_api import AgentApiClient
from ..agents.booking_agent_factory import BookingAgentFactory


class WorkflowExecutor:
    """
    Executes LangGraph workflows.
    
    These are PURE state machines that:
    - Take structured input
    - Perform technical operations (API calls, validations, etc.)
    - Return structured output
    - NO conversation text generation!
    
    Reem (Intelligent Agent) wraps the results in natural language.
    
    IMPORTANT: For complex flows (booking), this delegates to LangGraph!
    """
    
    def __init__(self):
        self.api_client = AgentApiClient()
        self.booking_agent = None  # Lazy load LangGraph
        logger.info("✅ WorkflowExecutor initialized")
    
    async def execute(
        self, 
        workflow_name: str, 
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute a workflow and return structured result.
        
        Args:
            workflow_name: Name of workflow to execute
            params: Parameters for the workflow
            context: Additional context (session data, etc.)
        
        Returns:
            WorkflowResult with structured data
        
        Example:
            result = await executor.execute(
                workflow_name="registration",
                params={"name": "شادي سالم", "national_id": "1038402840"},
                context={"phone": "201064414874"}
            )
            
            # result.data contains:
            # {
            #     "patient_id": 187,
            #     "name": "شادي سالم",
            #     "success": True
            # }
        """
        
        logger.info(f"🔄 Executing workflow: {workflow_name} with params: {params}")
        
        try:
            if workflow_name == "registration":
                return await self._execute_registration(params, context)
            
            elif workflow_name == "service_selection":
                return await self._execute_service_selection(params, context)
            
            elif workflow_name == "booking":
                return await self._execute_booking(params, context)
            
            elif workflow_name == "resource_selection":
                return await self._execute_resource_selection(params, context)
            
            else:
                logger.error(f"❌ Unknown workflow: {workflow_name}")
                return WorkflowResult(
                    success=False,
                    status=WorkflowStatus.FAILED,
                    errors=[f"Unknown workflow: {workflow_name}"]
                )
        
        except Exception as e:
            logger.error(f"❌ Workflow {workflow_name} failed: {e}", exc_info=True)
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.FAILED,
                errors=[str(e)],
                metadata={"workflow": workflow_name}
            )
    
    async def _execute_registration(
        self, 
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> WorkflowResult:
        """
        Execute patient registration workflow.
        
        Required params:
            - name: Full patient name
            - national_id: National ID or residence number
        
        Optional params:
            - gender: Will be auto-detected if not provided
            - email: Patient email
        
        Context must provide:
            - phone: Patient phone number (from WhatsApp)
        
        Returns:
            WorkflowResult with patient_id if successful
        """
        
        logger.info(f"👤 Starting registration workflow")
        
        # Validate required params
        if not params.get("name"):
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.PENDING,
                next_step="awaiting_name",
                errors=["Name is required"]
            )
        
        if not params.get("national_id"):
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.PENDING,
                data={"name": params["name"]},
                next_step="awaiting_national_id",
                errors=["National ID is required"]
            )
        
        # Get phone from context
        phone = context.get("phone") if context else None
        if not phone:
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.FAILED,
                errors=["Phone number not found in context"]
            )
        
        # Auto-detect gender from name
        from ..utils.gender_detector import detect_gender_from_name
        detected_gender = await detect_gender_from_name(params["name"])
        logger.info(f"🤖 Detected gender: {detected_gender} for name '{params['name']}'")
        
        # Auto-detect country/city from phone
        phone_clean = phone.lstrip("+")
        if phone_clean.startswith("20"):
            city = "القاهرة"
            country_code = "EG"
        elif phone_clean.startswith("971"):
            city = "دبي"
            country_code = "AE"
        else:
            city = "الرياض"
            country_code = "SA"
        
        # Prepare patient data for API (correct field names!)
        patient_data = {
            "name": params["name"],
            "identification_id": params["national_id"],
            "patient_phone": phone,
            "gender": detected_gender,
            "city": city,
            "country_code": country_code
        }
        
        if params.get("email"):
            patient_data["email"] = params["email"]
        
        logger.info(f"📋 Creating patient: {patient_data['name']} | ID: {patient_data['identification_id']} | Phone: {phone}")
        
        # Call API to create patient
        try:
            result = await self.api_client.create_patient(**patient_data)
            
            patient_id = result.get("id")
            if not patient_id:
                raise ValueError("API did not return patient ID")
            
            logger.info(f"✅ Patient registered successfully: ID {patient_id}")
            
            return WorkflowResult(
                success=True,
                status=WorkflowStatus.SUCCESS,
                data={
                    "patient_id": patient_id,
                    "name": params["name"],
                    "national_id": params["national_id"],
                    "phone": phone,
                    "gender": detected_gender,
                    "city": city,
                    "country_code": country_code
                },
                next_step=None,
                errors=[],
                metadata={"workflow": "registration"}
            )
        
        except Exception as e:
            logger.error(f"❌ Registration failed: {e}")
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.FAILED,
                errors=[f"Registration API error: {str(e)}"],
                metadata={"workflow": "registration"}
            )
    
    async def _execute_service_selection(
        self, 
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> WorkflowResult:
        """
        Execute service selection workflow.
        
        Required params:
            - query: Service search query OR service_id
        
        Returns:
            WorkflowResult with list of matching services
        """
        
        logger.info(f"🔍 Starting service selection workflow")
        
        # If service_id provided, get that specific service
        if params.get("service_id"):
            service_id = params["service_id"]
            try:
                service = await self.api_client.get_service(service_id)
                
                return WorkflowResult(
                    success=True,
                    status=WorkflowStatus.SUCCESS,
                    data={
                        "selected_service": service,
                        "service_id": service_id
                    },
                    next_step=None,
                    metadata={"workflow": "service_selection"}
                )
            
            except Exception as e:
                logger.error(f"❌ Failed to get service {service_id}: {e}")
                return WorkflowResult(
                    success=False,
                    status=WorkflowStatus.FAILED,
                    errors=[f"Service not found: {service_id}"]
                )
        
        # Otherwise search for services
        query = params.get("query")
        if not query:
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.PENDING,
                next_step="awaiting_service_query",
                errors=["Service query is required"]
            )
        
        try:
            # Search services
            services = await self.api_client.search_services(query)
            
            if not services:
                logger.warning(f"⚠️ No services found for query: {query}")
                return WorkflowResult(
                    success=False,
                    status=WorkflowStatus.PARTIAL,
                    data={"query": query, "services": []},
                    next_step="awaiting_service_selection",
                    errors=[f"No services found for: {query}"]
                )
            
            logger.info(f"✅ Found {len(services)} services for query: {query}")
            
            return WorkflowResult(
                success=True,
                status=WorkflowStatus.SUCCESS,
                data={
                    "query": query,
                    "services": services,
                    "count": len(services)
                },
                next_step="awaiting_service_selection",
                metadata={"workflow": "service_selection"}
            )
        
        except Exception as e:
            logger.error(f"❌ Service search failed: {e}")
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.FAILED,
                errors=[f"Service search error: {str(e)}"]
            )
    
    async def _execute_booking(
        self, 
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> WorkflowResult:
        """
        Execute booking creation workflow using LangGraph.
        
        This delegates to the REAL LangGraph booking agent which handles:
        - Patient verification
        - Service selection
        - Slot selection
        - Resource allocation (doctor/specialist/device)
        - Booking creation
        
        Params can be partial - LangGraph will guide through missing steps.
        
        Returns:
            WorkflowResult with booking_id if successful, or next_step if incomplete
        """
        
        logger.info(f"📅 Starting LangGraph booking workflow")
        
        # Prepare context for LangGraph
        langgraph_context = context or {}
        
        # Lazy load LangGraph booking agent
        if not self.booking_agent:
            logger.info("🔄 Initializing LangGraph booking agent...")
            session_id = langgraph_context.get("session_id", "workflow_temp")
            self.booking_agent = BookingAgentFactory.create(
                session_id=session_id,
                api_client=self.api_client
            )
        langgraph_context.update({
            "conversation_history": langgraph_context.get("conversation_history", []),
            "patient_data": langgraph_context.get("patient_data"),
            "phone": langgraph_context.get("phone")
        })
        
        # Build message for LangGraph (simulate user intent)
        if params.get("service_name"):
            message = f"أبي أحجز {params['service_name']}"
        elif params.get("service_id"):
            message = f"أبي أحجز الخدمة رقم {params['service_id']}"
        else:
            message = "أبي أحجز موعد"
        
        try:
            # Call LangGraph booking agent
            logger.info(f"🤖 Calling LangGraph with message: '{message}'")
            result = await self.booking_agent.handle(
                message=message,
                session_key=langgraph_context.get("session_id", "temp"),
                context=langgraph_context
            )
            
            # Parse LangGraph result
            response_text = result.get("response", "")
            booking_state = result.get("booking_state", {})
            
            # Check if booking completed
            if booking_state.get("booking_id"):
                logger.info(f"✅ LangGraph booking completed: ID {booking_state['booking_id']}")
                return WorkflowResult(
                    success=True,
                    status=WorkflowStatus.SUCCESS,
                    data={
                        "booking_id": booking_state["booking_id"],
                        "confirmation_code": booking_state.get("confirmation_code"),
                        "appointment_date": booking_state.get("selected_date"),
                        "service_name": booking_state.get("selected_service_name"),
                        "langgraph_response": response_text
                    },
                    next_step=None,
                    metadata={"workflow": "booking", "via": "langgraph"}
                )
            
            # Booking in progress - return current state
            else:
                current_step = booking_state.get("step", "unknown")
                logger.info(f"🔄 LangGraph booking in progress: step={current_step}")
                return WorkflowResult(
                    success=False,
                    status=WorkflowStatus.PENDING,
                    data={
                        "current_step": current_step,
                        "booking_state": booking_state,
                        "langgraph_response": response_text
                    },
                    next_step=current_step,
                    errors=[],
                    metadata={"workflow": "booking", "via": "langgraph"}
                )
        
        except Exception as e:
            logger.error(f"❌ LangGraph booking failed: {e}", exc_info=True)
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.FAILED,
                errors=[f"LangGraph booking error: {str(e)}"],
                metadata={"workflow": "booking", "via": "langgraph"}
            )
    
    async def _execute_resource_selection(
        self, 
        params: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> WorkflowResult:
        """
        Execute resource selection workflow (doctor/specialist/device).
        
        Required params:
            - resource_type: "doctor", "specialist", or "device"
            - service_id: Service ID (to filter available resources)
        
        Optional params:
            - resource_id: Specific resource ID to select
        
        Returns:
            WorkflowResult with list of available resources
        """
        
        logger.info(f"🔧 Starting resource selection workflow")
        
        resource_type = params.get("resource_type")
        service_id = params.get("service_id")
        
        if not resource_type or not service_id:
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.PENDING,
                errors=["resource_type and service_id are required"]
            )
        
        try:
            # Get available resources based on type
            if resource_type == "doctor":
                resources = await self.api_client.get_doctors(limit=20)
            elif resource_type == "specialist":
                resources = await self.api_client.get_specialists(limit=20)
            elif resource_type == "device":
                resources = await self.api_client.get_devices(limit=20)
            else:
                raise ValueError(f"Invalid resource_type: {resource_type}")
            
            if not resources:
                logger.warning(f"⚠️ No {resource_type}s available")
                return WorkflowResult(
                    success=False,
                    status=WorkflowStatus.PARTIAL,
                    data={"resource_type": resource_type, "resources": []},
                    errors=[f"No {resource_type}s available"]
                )
            
            logger.info(f"✅ Found {len(resources)} {resource_type}s")
            
            # If specific resource_id requested, filter to that one
            if params.get("resource_id"):
                selected = [r for r in resources if r.get("id") == params["resource_id"]]
                if selected:
                    return WorkflowResult(
                        success=True,
                        status=WorkflowStatus.SUCCESS,
                        data={
                            "resource_type": resource_type,
                            "selected_resource": selected[0],
                            "resource_id": params["resource_id"]
                        },
                        metadata={"workflow": "resource_selection"}
                    )
            
            return WorkflowResult(
                success=True,
                status=WorkflowStatus.SUCCESS,
                data={
                    "resource_type": resource_type,
                    "resources": resources,
                    "count": len(resources)
                },
                next_step="awaiting_resource_selection",
                metadata={"workflow": "resource_selection"}
            )
        
        except Exception as e:
            logger.error(f"❌ Resource selection failed: {e}")
            return WorkflowResult(
                success=False,
                status=WorkflowStatus.FAILED,
                errors=[f"Resource selection error: {str(e)}"]
            )
