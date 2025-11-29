"""
Agent API Client - Backend API Integration with Token Management
=================================================================
Handles all backend API calls with automatic JWT token management.

Features:
- Automatic login and token refresh
- 12-hour access token management
- Thread-safe token storage
- Retry logic for failed requests
- All CRUD operations (GET, POST, PUT, DELETE)

Author: Agent Orchestrator Team
Version: 2.0.0
"""

import httpx
from fastapi import HTTPException
from loguru import logger
from typing import Dict, Any, Optional

from ..config import get_settings
from ..core.token_manager import get_token_manager
from ..utils.circuit_breaker import get_circuit_breaker, CircuitBreakerOpenError


class AgentApiClient:
    """
    Professional API client with automatic token management.
    Singleton pattern to avoid redundant initialization.
    
    Handles all backend API calls for:
    - Bookings
    - Patients
    - Services
    - Doctors
    - Specialists
    - Devices
    - Slots
    - Ratings
    - Wallet
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentApiClient, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        # Only initialize once
        if not AgentApiClient._initialized:
            self.settings = get_settings()
            self.base_url: str = str(self.settings.agent_api_base_url)
            
            # Initialize token manager with pre-existing token if available
            self.token_manager = get_token_manager(
                login_url=str(self.settings.agent_login_url),
                username=self.settings.agent_api_user,
                password=self.settings.agent_api_password.get_secret_value() if self.settings.agent_api_password else "",
                token_file="tokens.json",
                pre_existing_token=self.settings.agent_api_token
            )
            
            AgentApiClient._initialized = True
            logger.info(f"✅ AgentApiClient initialized for {self.base_url} (singleton)")
        else:
            logger.debug(f"♻️ Reusing existing AgentApiClient instance")
    
    def _client(self) -> httpx.AsyncClient:
        """Get HTTP client instance"""
        return httpx.AsyncClient(timeout=30.0, verify=False)
    
    async def get_jwt(self) -> str:
        """Get valid JWT token (for compatibility with router.py)"""
        return await self.token_manager.get_valid_access_token()
    
    async def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers with valid token"""
        # Token manager handles automatic refresh
        return await self.token_manager.get_auth_headers()
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Redact sensitive headers before logging"""
        try:
            return {
                k: ("<redacted>" if k.lower() in {"authorization", "proxy-authorization"} else v)
                for k, v in (headers or {}).items()
            }
        except Exception:
            return {}
    
    def _log_http_error(
        self,
        method: str,
        url: str,
        request_headers: Dict[str, str],
        request_kwargs: Dict[str, Any],
        response: Optional[httpx.Response] = None,
        exc: Optional[Exception] = None,
    ) -> None:
        """Log rich HTTP error context for debugging (Issue #44)."""
        try:
            # CRITICAL: Get request ID for correlation
            from app.middleware.request_id import get_request_id
            request_id = get_request_id()
            
            safe_headers = self._sanitize_headers(request_headers)
            params = request_kwargs.get("params")
            json_body = request_kwargs.get("json") or request_kwargs.get("data")
            status = response.status_code if response is not None else "N/A"
            resp_headers = dict(response.headers) if response is not None else {}
            # Avoid huge bodies in logs
            resp_text = response.text[:1000] if response is not None else ""
            self_base = getattr(self, "base_url", "")
            logger.error(
                f"[REQ:{request_id}] HTTP ERROR {method.upper()} {url} (base={self_base})\n"
                f"- status: {status}\n"
                f"- request_headers: {safe_headers}\n"
                f"- request_params: {params}\n"
                f"- request_body_preview: {str(json_body)[:500]}\n"
                f"- response_headers: {resp_headers}\n"
                f"- response_text_preview: {resp_text}"
            )
        except Exception as log_exc:
            logger.error(f"Failed to log HTTP error context: {log_exc}")
    
    async def _request_with_retry(
        self,
        method: str,
        path: str,
        timeout_override: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic, circuit breaker, and automatic token refresh.
        
        Args:
            method: HTTP method (get, post, put, delete)
            path: API endpoint path
            timeout_override: Optional timeout override for slow endpoints (seconds)
            **kwargs: Additional request parameters
            
        Returns:
            Response JSON data
        """
        circuit_breaker = get_circuit_breaker("agent_api", failure_threshold=5, recovery_timeout=60)
        max_retries = 3  # Increased to allow for token refresh + retry
        timeout_seconds = timeout_override if timeout_override is not None else 60.0
        token_refreshed = False  # Track if we've already refreshed once
        
        # Log retry strategy and timeout configuration (Issue #18)
        logger.debug(f"🔄 API Request: {method.upper()} {path}")
        logger.debug(f"   ⚙️ Config: timeout={timeout_seconds}s, max_retries={max_retries}, circuit_breaker=enabled")
        logger.debug(f"   🔒 Circuit: failure_threshold=5, recovery_timeout=60s")
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    # Log retry attempt (Issue #19)
                    logger.info(f"🔄 RETRY ATTEMPT {attempt + 1}/{max_retries}: {method.upper()} {path}")
                
                headers = await self._get_headers()
                
                async def _make_request():
                    import time
                    start_time = time.time()
                    
                    # Create client with longer timeout and proper SSL handling
                    timeout_config = httpx.Timeout(timeout_seconds, connect=10.0)
                    
                    async with httpx.AsyncClient(
                        timeout=timeout_config,
                        verify=False,  # Disable SSL verification for development
                        follow_redirects=True,
                        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
                    ) as client:
                        url = f"{self.base_url}{path}"
                        
                        logger.debug(f"📡 Making request to: {url}")
                        logger.debug(f"📋 Headers: {self._sanitize_headers(headers)}")
                        
                        if method == "get":
                            response = await client.get(url, headers=headers, **kwargs)
                        elif method == "post":
                            response = await client.post(url, headers=headers, **kwargs)
                        elif method == "put":
                            response = await client.put(url, headers=headers, **kwargs)
                        elif method == "delete":
                            response = await client.delete(url, headers=headers, **kwargs)
                        
                        response.raise_for_status()
                        elapsed = time.time() - start_time
                        logger.debug(f"⏱️ API timing: {method.upper()} {path} took {elapsed:.3f}s")
                        return response.json()
                
                result = circuit_breaker.call(_make_request)
                response_data = await result  # Await the coroutine
                
                # Log successful first attempt (Issue #19)
                if attempt == 0:
                    logger.debug(f"✅ API SUCCESS (first attempt): {method.upper()} {path}")
                else:
                    logger.info(f"✅ API SUCCESS (after {attempt + 1} attempts): {method.upper()} {path}")
                
                return response_data
                    
            except CircuitBreakerOpenError as e:
                logger.error(f"🔴 API CIRCUIT OPEN: {method.upper()} {path} blocked - {e}")
                logger.error(f"🔴 FALLBACK: Service unavailable, returning error")
                raise HTTPException(status_code=503, detail="Backend API circuit breaker open")
            except httpx.TimeoutException as exc:
                logger.error(f"🔴 API TIMEOUT: {method.upper()} {path} - Request exceeded {timeout_seconds}s")
                logger.error(f"🔴 IMPACT: User request will fail")
                logger.error(f"🔴 URL: {self.base_url}{path}")
                if attempt < max_retries - 1:
                    logger.warning(f"🔄 Retry {attempt + 1}/{max_retries - 1}...")
                    continue
                raise
            except httpx.ConnectError as exc:
                logger.error(f"🔴 API CONNECTION FAILED: Cannot reach {self.base_url}")
                logger.error(f"🔴 ENDPOINT: {path}")
                logger.error(f"🔴 ERROR: {type(exc).__name__}: {exc}")
                logger.error(f"🔴 IMPACT: All API operations unavailable")
                
                # DON'T re-authenticate on connection errors - it's a network issue, not auth
                # Connection errors mean:
                # - Server is down
                # - Network issue
                # - DNS problem
                # - Wrong URL
                # Re-authenticating won't fix these!
                
                if attempt < max_retries - 1:
                    import asyncio
                    logger.warning(f"🔄 Retrying connection in 2 seconds... (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(2)
                    continue
                
                logger.error(f"❌ All {max_retries} connection attempts failed")
                raise
            except httpx.HTTPStatusError as exc:
                # Always refresh token on 401, but only once per request
                if exc.response.status_code == 401 and not token_refreshed:
                    logger.warning(f"🔄 401 error on attempt {attempt + 1}, refreshing token...")
                    
                    # Force token refresh
                    await self.token_manager.login()
                    token_refreshed = True
                    logger.info("✅ Token refreshed, retrying request...")
                    
                    # Don't count this against retry limit - try again with new token
                    continue
                elif exc.response.status_code == 404 and "/patients/search" in path:
                    # 404 is EXPECTED for patient search (patient doesn't exist yet)
                    logger.info(f"ℹ️ Patient not found at {path} - proceeding to registration")
                    raise  # Still raise but logged as INFO, not ERROR
                else:
                    # Rich error logging with full context
                    try:
                        self._log_http_error(method, url, headers, kwargs, response=exc.response, exc=exc)
                    except Exception:
                        # Fallback minimal log if rich logging fails
                        # CRITICAL: Include request ID for correlation with backend logs
                        from app.middleware.request_id import get_request_id
                        request_id = get_request_id()
                        logger.error(f"[REQ:{request_id}] {method.upper()} {path} failed: {exc.response.status_code} - {exc.response.text}")
                    raise
            except Exception as exc:
                # Attempt to log request context even for non-HTTP errors
                logger.error(f"🔴 API ERROR (unexpected): {method.upper()} {path} - {exc}", exc_info=True)
                try:
                    self._log_http_error(method, url, headers, kwargs, response=None, exc=exc)
                except Exception:
                    pass
                raise
        
        raise RuntimeError(f"Failed after {max_retries} attempts")
    
    async def get(self, path: str, params: Optional[Dict] = None, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Perform GET request with retry."""
        return await self._request_with_retry("get", path, params=params, timeout_override=timeout)
    
    async def post(self, path: str, json_body: Optional[Dict] = None, timeout: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Perform POST request with retry."""
        if json_body is not None:
            kwargs["json"] = json_body
        return await self._request_with_retry("post", path, timeout_override=timeout, **kwargs)
    
    async def put(self, path: str, json_body: Optional[Dict] = None, timeout: Optional[float] = None, **kwargs) -> Dict[str, Any]:
        """Perform PUT request with retry."""
        if json_body is not None:
            kwargs["json"] = json_body
        return await self._request_with_retry("put", path, timeout_override=timeout, **kwargs)
    
    async def delete(self, path: str, **kwargs) -> Dict[str, Any]:
        """DELETE request"""
        return await self._request_with_retry("delete", path, **kwargs)
    
    # ========================================================================
    # BOOKING ENDPOINTS
    # ========================================================================
    
    async def get_bookings(
        self,
        patient_id: Optional[int] = None,
        doctor_id: Optional[int] = None,
        state: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get all bookings with optional filters"""
        params = {"limit": limit, "offset": offset}
        if patient_id:
            params["patient_id"] = patient_id
        if doctor_id:
            params["doctor_id"] = doctor_id
        if state:
            params["state"] = state
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to
        
        return await self.get("/booking", params=params)
    
    async def get_booking(self, booking_id: int, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get single booking by ID"""
        return await self.get(f"/booking/{booking_id}", timeout=timeout)
    
    async def create_booking(
        self,
        patient_id: int,
        service_id: int,
        start_date: str,
        start_time: str,
        slot_choice_id: Optional[int] = None,
        doctor_id: Optional[int] = None,
        specialist_id: Optional[int] = None,
        device_id: Optional[int] = None,
        duration_minutes: Optional[int] = None,
        price_adjustment: float = 0.0,
        currency: str = "SAR"
    ) -> Dict[str, Any]:
        """
        Create new booking.
        
        CRITICAL: API requires start_time in HH:MM format (NOT HH:MM:SS)!
        
        IMPORTANT:
        - service_id MUST be a subcategory service ID (not a category ID)
        - start_time is REQUIRED in HH:MM format (e.g., "10:00")
        - slot_choice_id is DEPRECATED (old API, kept for backward compatibility)
        - MUST include exactly ONE of: doctor_id, specialist_id, or device_id
        - duration_minutes should match service duration
        
        Args:
            patient_id: Patient ID
            service_id: Subcategory service ID (from get_subcategory_services)
            start_date: Date in YYYY-MM-DD format (use November 2025 for testing)
            start_time: Time in HH:MM format (e.g., "10:00") - REQUIRED!
            slot_choice_id: DEPRECATED - use start_time instead
            doctor_id/specialist_id/device_id: Resource based on service requirement
            duration_minutes: Service duration (get from service details)
            
        Returns:
            Dict with created booking, including 'id' field
            
        Example:
            booking = await client.create_booking(
                patient_id=123,
                service_id=122,  # subcategory service
                start_date="2025-11-27",
                start_time="10:00",  # HH:MM format!
                doctor_id=3,
                duration_minutes=30
            )
        """
        payload = {
            "patient_id": patient_id,
            "service_id": service_id,
            "start_date": start_date,
            "start_time": start_time,  # REQUIRED!
            "price_adjustment": price_adjustment
        }
        
        # slot_choice_id is deprecated but kept for backward compatibility
        if slot_choice_id:
            payload["slot_choice_id"] = slot_choice_id
        if doctor_id:
            payload["doctor_id"] = doctor_id
        if specialist_id:
            payload["specialist_id"] = specialist_id
        if device_id:
            payload["device_id"] = device_id
        if duration_minutes:
            payload["duration_minutes"] = duration_minutes
        
        # Remove currency from params - not accepted by API
        return await self.post("/booking/create", json=payload)
    
    async def reschedule_booking(
        self,
        session_name: str,
        start_date: str,
        start_time: str,
        scope: str = "this_and_following",
        strategy: str = "refind_slots",
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Reschedule a booking session"""
        payload = {
            "name": session_name,
            "start_date": start_date,
            "start_time": start_time,
            "scope": scope,
            "strategy": strategy,
            "lock_resources": True,
            "respect_retouch_window": True
        }
        if reason:
            payload["reason"] = reason
        
        return await self.post("/sessions/reschedule", json=payload)
    
    # ========================================================================
    # PATIENT ENDPOINTS
    # ========================================================================
    
    async def get_patients(
        self,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get all patients with optional search"""
        params = {"limit": limit, "offset": offset}
        if q:
            params["q"] = q
        return await self.get("/patients", params=params, timeout=timeout)
    
    async def get_patient(self, patient_id: int, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Get single patient by ID"""
        return await self.get(f"/patients/{patient_id}", timeout=timeout)
    
    async def get_patient_by_phone(self, phone: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Get patient directly by phone number (e.g., /patients/552345602)"""
        try:
            # Normalize phone to local format (remove +966, 966, leading 0)
            phone_clean = phone.replace("+", "").replace(" ", "")
            if phone_clean.startswith("966"):
                phone_clean = phone_clean[3:]
            if phone_clean.startswith("0"):
                phone_clean = phone_clean[1:]
            
            # Direct API call to /patients/{phone}
            result = await self.get(f"/patients/{phone_clean}", timeout=timeout)
            return result if result else None
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                logger.info(f"Patient not found by phone: {phone}")
                return None
            raise
        except Exception as e:
            logger.warning(f"Patient phone lookup failed for {phone}: {e}")
            return None
    
    async def search_patient(self, phone: str) -> Optional[Dict[str, Any]]:
        """Search for patient by phone number - tries multiple formats"""
        try:
            # Try multiple phone formats
            phone_formats = set()  # Use set to avoid duplicates
            
            # Clean phone
            phone_clean = phone.replace("+", "").replace(" ", "").replace("-", "")
            
            # Format 1: Original
            phone_formats.add(phone_clean)
            
            # Format 2: Remove Saudi country code (966)
            if phone_clean.startswith("966"):
                phone_formats.add(phone_clean[3:])
            
            # Format 3: Remove other country codes (20 for Egypt, etc.)
            if phone_clean.startswith("20") and len(phone_clean) > 10:
                phone_formats.add(phone_clean[2:])  # Remove "20"
            
            # Format 4: Remove leading 0
            if phone_clean.startswith("0"):
                phone_formats.add(phone_clean[1:])

            
            # Format 6: Add Saudi country code (966)
            if not phone_clean.startswith("966"):
                # Remove leading 0 first if present
                local = phone_clean[1:] if phone_clean.startswith("0") else phone_clean
                phone_formats.add("966" + local)
            
            # Format 7: Try with Saudi mobile prefix (05)
            if not phone_clean.startswith("05") and not phone_clean.startswith("5"):
                # Extract last 9 digits and add 05
                if len(phone_clean) >= 9:
                    last_9 = phone_clean[-9:]
                    phone_formats.add("05" + last_9)
                    phone_formats.add("5" + last_9)
            
            # Format 8: Last 10 digits (common format)
            if len(phone_clean) > 10:
                phone_formats.add(phone_clean[-10:])
            
            # Format 9: Last 9 digits (mobile without leading 0)
            if len(phone_clean) > 9:
                phone_formats.add(phone_clean[-9:])
            
            # Convert set to list for logging
            phone_formats_list = list(phone_formats)
            logger.info(f"🔍 Trying {len(phone_formats_list)} phone formats for: {phone}")
            
            # Try each format
            for fmt in phone_formats_list:
                logger.debug(f"   Trying format: {fmt}")
                result = await self.get_patients(q=fmt, limit=1)
                
                if result.get("count", 0) > 0 and result.get("results"):
                    logger.info(f"✅ Found patient with format: {fmt}")
                    return result["results"][0]
            
            logger.warning(f"❌ Patient not found with any of {len(phone_formats_list)} formats")
            logger.warning(f"   Formats tried: {phone_formats_list}")
            return None
        except Exception as e:
            logger.error(f"⚠️ Patient search error for {phone}: {e}")
            return None
    
    async def create_patient(self, timeout: Optional[float] = None, **patient_data) -> Dict[str, Any]:
        """Create new patient"""
        return await self.post("/customer/create", json_body=patient_data, timeout=timeout)
    
    async def update_patient(self, patient_id: int, timeout: Optional[float] = None, **update_data) -> Dict[str, Any]:
        """Update existing patient - PUT with ID in URL, same payload as create"""
        return await self.put(f"/customer/update/{patient_id}", json_body=update_data, timeout=timeout)
    
    # ========================================================================
    # SERVICE ENDPOINTS  
    # ========================================================================
    
    async def get_services(
        self,
        service_type_id: Optional[int] = None,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> list:
        """
        Get all services with optional filters.
        
        NOTE: The /services endpoint returns SERVICE CATEGORIES by default.
        To get actual bookable services (subcategories), use service_type_id parameter
        with a category ID to fetch its subcategories.
        
        Returns:
            list: Array of service objects (extracted from response['results'])
        
        Example:
            # Get categories
            categories = await client.get_services()
            category_id = categories[0]['id']  # Now returns list directly
            
            # Get subcategories (actual bookable services)
            subcategories = await client.get_services(service_type_id=category_id)
        """
        params = {"limit": limit, "offset": offset}
        if service_type_id:
            params["service_type_id"] = service_type_id
        if q:
            params["q"] = q
        
        response = await self.get("/services", params=params)
        
        # CRITICAL FIX: Extract results array from paginated response
        # Backend returns: {"mode": "...", "count": X, "results": [...]}
        # or with typo: {"mode": "...", "count": X, "resullts": [...]}
        
        if isinstance(response, dict):
            # Try correct spelling first
            if "results" in response:
                return response["results"]
            # Handle backend typo
            elif "resullts" in response:
                logger.warning("⚠️ Backend API has typo: 'resullts' instead of 'results'")
                return response["resullts"]
            else:
                logger.error(f"❌ Unexpected API response structure: {list(response.keys())}")
                return []
        
        # If response is already a list, return it
        return response if isinstance(response, list) else []
    
    async def get_service(self, service_id: int, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        Get single service by ID.
        
        Returns detailed service information including duration_minutes,
        resource requirements (doctor/specialist/device), and pricing.
        """
        return await self.get(f"/services/{service_id}", timeout=timeout)
    
    async def get_subcategory_services(self, category_id: int, limit: int = 100) -> Dict[str, Any]:
        """
        Helper method to get subcategory services from a category.
        
        Args:
            category_id: The ID of the parent category
            limit: Maximum number of subcategories to return
            
        Returns:
            Dict containing 'results' list of subcategory services
            
        Example:
            subcategories = await client.get_subcategory_services(13)
            first_service = subcategories['results'][0]
            service_id = first_service['id']  # Use this for slots/booking
        """
        return await self.get_services(service_type_id=category_id, limit=limit)
    
    # ========================================================================
    # DOCTOR ENDPOINTS
    # ========================================================================
    
    async def get_doctors(self, limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Get all doctors"""
        return await self.get("/doctors", params={"limit": limit, "offset": offset})
    
    async def get_doctor(self, doctor_id: int) -> Dict[str, Any]:
        """Get single doctor by ID"""
        return await self.get(f"/doctors/{doctor_id}")
    
    # ========================================================================
    # SPECIALIST ENDPOINTS
    # ========================================================================
    
    async def get_specialists(
        self,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get all specialists"""
        params = {"limit": limit, "offset": offset}
        if q:
            params["q"] = q
        return await self.get("/specialists", params=params)
    
    async def get_specialist(self, specialist_id: int) -> Dict[str, Any]:
        """Get single specialist by ID"""
        return await self.get(f"/specialists/{specialist_id}")
    
    # ========================================================================
    # DEVICE ENDPOINTS
    # ========================================================================
    
    async def get_devices(
        self,
        q: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get all devices"""
        params = {"limit": limit, "offset": offset}
        if q:
            params["q"] = q
        return await self.get("/devices", params=params)
    
    async def get_device(self, device_id: int) -> Dict[str, Any]:
        """Get single device by ID"""
        return await self.get(f"/devices/{device_id}")
    
    # ========================================================================
    # SLOT ENDPOINTS
    # ========================================================================
    
    async def get_available_slots(
        self,
        service_id: int,
        date: str,
        patient_id: Optional[int] = None,
        doctor_id: Optional[int] = None,
        specialist_id: Optional[int] = None,
        device_id: Optional[int] = None,
        duration_minutes: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Get available time slots for booking.
        
        Example: GET /slots?service_id=127&date=2025-11-27&doctor_id=3&patient_id=185
        
        IMPORTANT:
        - service_id MUST be a subcategory service ID (not a category ID)
        - patient_id is optional but recommended
        - MUST include exactly ONE of: doctor_id, specialist_id, or device_id
          (based on service requirements)
        - date format: YYYY-MM-DD
        
        Args:
            service_id: Subcategory service ID (get from get_subcategory_services)
            date: Date in YYYY-MM-DD format
            patient_id: Patient ID (optional but recommended)
            doctor_id: Required if service requires doctor
            specialist_id: Required if service requires specialist
            device_id: Required if service requires device
            
        Returns:
            Dict with 'slots' array, each slot has:
            - 'time': Time in HH:MM format (e.g., "10:00")
            - 'minute': Minutes from midnight
            - 'slot_id': Usually null
            
        Example:
            result = await client.get_available_slots(
                service_id=127,
                date="2025-11-27",
                patient_id=185,
                doctor_id=3
            )
            # result = {"slots": [{"time": "10:00", "minute": 600, "slot_id": null}, ...]}
            first_time = result['slots'][0]['time']  # "10:00" - use directly for booking!
        """
        params = {
            "service_id": service_id,
            "date": date
        }
        if patient_id:
            params["patient_id"] = patient_id
        if doctor_id:
            params["doctor_id"] = doctor_id
        if specialist_id:
            params["specialist_id"] = specialist_id
        if device_id:
            params["device_id"] = device_id
        if duration_minutes:
            params["duration_minutes"] = duration_minutes
        
        return await self.get("/slots", params=params, timeout=timeout)
    
    # ========================================================================
    # RATING ENDPOINTS
    # ========================================================================
    
    async def get_ratings(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """Get all ratings"""
        params = {"limit": limit, "offset": offset}
        return await self.get("/rating", params=params)
    
    async def get_rating(self, rating_id: int) -> Dict[str, Any]:
        """Get single rating by ID"""
        return await self.get(f"/rating/{rating_id}")
    
    async def create_rating(
        self,
        booking_id: int,
        partner_id: int,
        stars: int,
        feedback: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create new rating"""
        payload = {
            "booking_id": booking_id,
            "partner_id": partner_id,
            "stars": stars
        }
        if feedback:
            payload["feedback"] = feedback
        
        return await self.post("/rating/create", json=payload)

# Singleton helper function
def get_api_client() -> AgentApiClient:
    """
    Get singleton instance of AgentApiClient.
    
    Returns:
        AgentApiClient: Singleton instance
    """
    return AgentApiClient()
