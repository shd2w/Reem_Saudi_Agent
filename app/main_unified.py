"""
UNIFIED MAIN APPLICATION
========================
This is the main entry point for the Agent Orchestrator system.
Supports both WhatsApp and Voice agent workflows with comprehensive security.

Features:
- WhatsApp message processing
- Voice call processing with TTS
- Multi-agent routing (Booking, Patient, Feedback, Resource)
- Session management
- Security middleware (API key, rate limiting, HMAC verification)
- Health checks and startup validation
- Comprehensive error handling
- Logging and monitoring

Author: Agent Orchestrator Team
Version: 1.0.0
"""

import sys
import os

# Allow running as script: python app/main_unified.py
# or as module: python -m app.main_unified
if __name__ == "__main__":
    # Add parent directory to path for relative imports
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
from datetime import datetime

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Import configuration and logging
from .config import get_settings, Settings
from .logging_config import configure_logging

# Import middleware
from .middleware.security import SecurityMiddleware, InputValidationMiddleware
from .middleware.error_handler import ErrorHandlingMiddleware
from .middleware.request_id import RequestIDMiddleware

# Import components
from .memory.session_manager import SessionManager

# Import API routers
from .api.router import router as agent_router
from .api.webhook_handler import webhook as webhook_router
from .api.webhook_handler_hybrid import router as webhook_hybrid_router  # ✅ NEW: Hybrid architecture
from .api.manual_review_routes import router as manual_review_router


# ============================================================================
# STARTUP AND SHUTDOWN HANDLERS
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize and cleanup resources"""
    logger.info("🚀 Application startup - initializing singletons...")
    
    # Initialize all singletons at startup
    from app.core.llm_reasoner import get_llm_reasoner
    from app.api.agent_api import AgentApiClient
    from app.orchestration.router import IntentRouter
    
    # Force initialization
    llm = get_llm_reasoner()
    session_mgr = SessionManager()
    api_client = AgentApiClient()
    router = IntentRouter()
    
    # Validate token on startup
    try:
        await api_client.token_manager.validate_and_refresh_if_needed()
    except Exception as e:
        logger.warning(f"Token validation failed on startup: {e}")
    
    logger.info("✅ All singletons initialized")
    
    yield
    
    # Cleanup on shutdown - BEFORE event loop closes
    logger.info("🛑 Application shutdown - cleaning up resources...")
    
    try:
        # Close Redis connections BEFORE event loop shutdown
        # This prevents "Event loop is closed" errors
        if hasattr(session_mgr, 'redis') and session_mgr.redis:
            await session_mgr.redis.aclose()  # Use aclose() for proper async cleanup
            logger.info("✅ Redis async connection closed")
        
        if hasattr(session_mgr, 'sync_redis') and session_mgr.sync_redis:
            session_mgr.sync_redis.close()
            logger.info("✅ Redis sync connection closed")
        
        # Close any other async resources
        if hasattr(api_client, 'session') and api_client.session:
            await api_client.session.close()
            logger.info("✅ HTTP session closed")
            
    except Exception as e:
        logger.error(f"⚠️ Error during cleanup: {e}")
    
    logger.info("✅ Application shutdown complete - all resources cleaned")

async def validate_environment():
    """
    Validate environment configuration on startup.
    Ensures all required services and configurations are available.
    """
    settings = get_settings()
    issues = []
    
    logger.info("=" * 60)
    logger.info("VALIDATING ENVIRONMENT CONFIGURATION")
    
    # Validate OpenAI API Key
    try:
        if settings.openai_api_key:
            logger.info("✓ OpenAI API Key: CONFIGURED")
        else:
            issues.append("OpenAI API Key not configured")
            logger.warning("✗ OpenAI API Key: MISSING")
    except Exception as e:
        issues.append(f"OpenAI validation error: {e}")
        logger.error(f"✗ OpenAI validation failed: {e}")
    logger.info("=" * 60)
    
    # Check Redis connection - REAL connectivity test
    try:
        session_manager = SessionManager()
        test_key = "startup_health_check"
        test_value = {"test": "ok", "timestamp": str(time.time())}
        
        # Write test
        await session_manager.put_session(test_key, test_value)
        # Read test
        result = await session_manager.get_session(test_key)
        # Cleanup
        await session_manager.delete_session(test_key)
        
        if result.get("test") == "ok":
            logger.info("✓ Redis connection: OK (read/write verified)")
        else:
            raise Exception("Redis read/write verification failed")
    except Exception as exc:
        issues.append(f"Redis connection failed: {exc}")
        logger.error(f"✗ Redis connection: FAILED - {exc}")
    
    # Check required API URLs
    if not settings.agent_api_base_url:
        issues.append("AGENT_API_URL not configured")
        logger.error("✗ Agent API URL: NOT CONFIGURED")
    else:
        logger.info(f"✓ Agent API URL: {settings.agent_api_base_url}")
    
    if not settings.wasender_base_url:
        issues.append("WASENDER_API_URL not configured")
        logger.error("✗ WaSender API URL: NOT CONFIGURED")
    else:
        logger.info(f"✓ WaSender API URL: {settings.wasender_base_url}")
    
    # Check authentication credentials - TEST actual API connectivity
    if settings.agent_api_token:
        # Using pre-existing unlimited token
        try:
            from .core.token_manager import TokenManager
            token_mgr = TokenManager(pre_existing_token=settings.agent_api_token)
            token = await token_mgr.get_valid_token()
            if token:
                logger.info("✓ Agent API: VERIFIED (using pre-existing unlimited token)")
            else:
                logger.warning("⚠ Agent API: Pre-existing token invalid")
        except Exception as e:
            logger.warning(f"⚠ Agent API: Token verification failed - {e}")
    elif not settings.agent_api_user or not settings.agent_api_password:
        logger.warning("⚠ Agent API credentials not configured (no token or user/password)")
    else:
        try:
            # Test actual API connectivity with login credentials
            from .core.token_manager import TokenManager
            token_mgr = TokenManager(
                login_url=settings.agent_login_url,
                username=settings.agent_api_user,
                password=settings.agent_api_password
            )
            token = await token_mgr.get_valid_token()
            if token:
                logger.info("✓ Agent API credentials: VERIFIED (token obtained via login)")
            else:
                logger.warning("⚠ Agent API credentials: CONFIGURED but token fetch failed")
        except Exception as e:
            logger.warning(f"⚠ Agent API credentials: CONFIGURED but verification failed - {e}")
    
    # Check TTS configuration
    if settings.elevenlabs_api_key:
        logger.info("✓ ElevenLabs TTS: CONFIGURED")
    else:
        logger.warning("⚠ ElevenLabs TTS: NOT CONFIGURED (voice features disabled)")
    
    # Check security settings
    if settings.api_key:
        logger.info("✓ API Key authentication: ENABLED")
    else:
        logger.warning("⚠ API Key authentication: DISABLED (not recommended for production)")
    
    logger.info("=" * 60)
    
    if issues:
        logger.error("ENVIRONMENT VALIDATION FAILED:")
        for issue in issues:
            logger.error(f"  - {issue}")
        logger.error("=" * 60)
        logger.warning("Some features may not work correctly. Please check configuration.")
    else:
        logger.info("✓ ALL ENVIRONMENT CHECKS PASSED")
        logger.info("=" * 60)
    
    return len(issues) == 0


async def startup_checks():
    """Run all startup checks and initialization"""
    logger.info("Starting Agent Orchestrator System...")
    
    # Validate environment
    env_valid = await validate_environment()
    
    if not env_valid:
        logger.warning("Starting with configuration warnings...")
    
    # CRITICAL: Login to backend API and get access token
    logger.info("=" * 60)
    logger.info("AUTHENTICATING WITH BACKEND API")
    try:
        from app.api.agent_api import get_api_client
        api_client = get_api_client()
        
        # Check if we already have a valid unlimited token
        token_manager = api_client.token_manager
        if token_manager._access_token and token_manager._access_token_expiry:
            # Check if token is far-future (unlimited) - year > 2100
            if token_manager._access_token_expiry.year > 2100:
                logger.info("✅ Backend API already authenticated with unlimited token")
                logger.info(f"✅ Token expires: {token_manager._access_token_expiry}")
            else:
                # Token exists but might expire soon - try login
                await token_manager.login()
                logger.info("✅ Backend API authentication successful")
                logger.info(f"✅ Access token obtained - expires at {token_manager._access_token_expiry}")
        else:
            # No token - force login
            await token_manager.login()
            logger.info("✅ Backend API authentication successful")
            logger.info(f"✅ Access token obtained - expires at {token_manager._access_token_expiry}")
    except Exception as e:
        logger.error(f"❌ Backend API authentication failed: {e}")
        logger.error("⚠️ All API operations will fail until login succeeds")
        logger.error("⚠️ System will attempt auto-login on first API request")
    logger.info("=" * 60)
    
    # Validate LangGraph graph compilation (Issue #15)
    logger.info("=" * 60)
    logger.info("VALIDATING LANGGRAPH BOOKING AGENT")
    try:
        from app.agents.booking_agent_langgraph import BookingAgentLangGraph
        # Force graph compilation
        graph = BookingAgentLangGraph._get_or_create_graph()
        logger.info("✅ LangGraph graph compiled successfully")
        
        # Try to instantiate a test agent
        test_agent = BookingAgentLangGraph("startup_health_check")
        logger.info("✅ LangGraph agent instantiation successful")
    except Exception as e:
        logger.error(f"❌ LangGraph validation failed: {e}")
        logger.warning("⚠️ Booking functionality may be impaired")
        logger.warning("⚠️ Users will be directed to call support: 920033304")
    logger.info("=" * 60)
    
    logger.info("Agent Orchestrator System started successfully")
    logger.info("Ready to process WhatsApp and Voice requests")


async def shutdown_cleanup():
    """Cleanup resources on shutdown"""
    logger.info("Shutting down Agent Orchestrator System...")
    logger.info("Cleanup completed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await startup_checks()
    yield
    # Shutdown
    await shutdown_cleanup()


# ============================================================================
# APPLICATION INITIALIZATION
# ============================================================================

# Configure logging first
configure_logging()

# Create FastAPI application
app = FastAPI(
    title="Agent Orchestrator - Unified System",
    description="Multi-channel agent orchestration system supporting WhatsApp and Voice",
    version="1.0.0",
    lifespan=lifespan
)

# Get settings
settings = get_settings()

# ============================================================================
# MIDDLEWARE CONFIGURATION (Issue #19: Order matters!)
# ============================================================================
#
# CRITICAL: FastAPI middleware executes in REVERSE order of how it's added!
# - add_middleware(A) then add_middleware(B) means B executes BEFORE A
# - Last added = First executed
#
# EXECUTION ORDER (from first to last):
# 1. ErrorHandlingMiddleware  ← Catches all errors (added last)
# 2. InputValidationMiddleware ← Validates request format
# 3. SecurityMiddleware ← Authentication & rate limiting
# 4. RequestIDMiddleware ← Adds X-Request-ID for tracing
# 5. CORSMiddleware ← Sets CORS headers (added first)

# CORS middleware (added FIRST, executes LAST for response headers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("🔧 MIDDLEWARE [5/5]: CORSMiddleware added (executes LAST for headers)")

# Request ID middleware (added SECOND, executes 4th - after security)
app.add_middleware(RequestIDMiddleware)
logger.info("🔧 MIDDLEWARE [4/5]: RequestIDMiddleware added (adds X-Request-ID)")

# Security middleware (added THIRD, executes 3rd - validates auth)
app.add_middleware(SecurityMiddleware)
logger.info("🔧 MIDDLEWARE [3/5]: SecurityMiddleware added (auth & rate limit)")

# Input validation (added FOURTH, executes 2nd - before error handling)
app.add_middleware(InputValidationMiddleware)
logger.info("🔧 MIDDLEWARE [2/5]: InputValidationMiddleware added (validates format)")

# Error handling (added LAST, executes FIRST - catches everything)
app.add_middleware(ErrorHandlingMiddleware)
logger.info("🔧 MIDDLEWARE [1/5]: ErrorHandlingMiddleware added (executes FIRST)")

logger.info("=" * 70)
logger.info("✅ MIDDLEWARE CONFIGURED")
logger.info("   ADDITION order (last added executes FIRST on request):")
logger.info("     5th: ErrorHandling   ← Added LAST   (wraps everything, executes FIRST)")
logger.info("     4th: InputValidation ← Added 4th    (executes 2nd)")
logger.info("     3rd: Security        ← Added 3rd    (executes 3rd)")
logger.info("     2nd: RequestID       ← Added 2nd    (executes 4th)")
logger.info("     1st: CORS            ← Added FIRST  (executes LAST)")
logger.info("")
logger.info("   REQUEST flow (inbound): ErrorHandling → InputValidation → Security → RequestID → CORS → App")
logger.info("   RESPONSE flow (outbound): App → CORS → RequestID → Security → InputValidation → ErrorHandling")
logger.info("=" * 70)

# ============================================================================
# INITIALIZE COMPONENTS
# ============================================================================

logger.info("Application components initialized")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Agent Orchestrator",
        "version": "3.0.0",  # Hybrid is now default!
        "status": "running",
        "architecture": "hybrid",
        "default_system": "Reem (Hybrid)",
        "channels": ["whatsapp", "voice"],
        "endpoints": {
            "health": "/health",
            "whatsapp_webhook": "/webhook",  # 🚀 Now uses Reem (hybrid)!
            "whatsapp_webhook_legacy": "/api/v1/webhook",  # Old system backup
            "unified": "/api/process"
        },
        "systems": {
            "hybrid": {
                "endpoint": "/webhook",
                "description": "Reem + Workflows (sales-optimized)",
                "status": "active - DEFAULT",
                "components": ["ConversationOrchestrator", "ReemAgent", "WorkflowExecutor"],
                "features": ["Natural conversation", "Sales skills", "Personalization", "Function calling"]
            },
            "legacy": {
                "endpoint": "/api/v1/webhook",
                "description": "Router-based system (backup)",
                "status": "active - FALLBACK",
                "components": ["IntentRouter", "IntelligentBookingAgent"]
            }
        }
    }


@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint.
    Returns status of all integrated services.
    """
    try:
        # Check Redis connection
        session_manager = SessionManager()
        await session_manager.get_session("health_check")
        redis_status = "healthy"
    except Exception:
        redis_status = "unhealthy"
    
    health_status = {
        "application": {
            "name": settings.app_name,
            "environment": settings.app_env,
            "version": "1.0.0",
            "status": "healthy"
        },
        "redis": redis_status,
        "timestamp": datetime.now().isoformat()
    }
    
    overall_healthy = redis_status == "healthy"
    
    return JSONResponse(
        status_code=status.HTTP_200_OK if overall_healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=health_status
    )


@app.get("/api/confidence/stats")
async def get_confidence_stats():
    """
    Get adaptive confidence threshold statistics.
    Shows current thresholds, accuracy per intent, and adjustment history.
    """
    try:
        from .utils.adaptive_confidence import get_confidence_manager
        
        conf_mgr = get_confidence_manager()
        stats = conf_mgr.get_stats_summary()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "adaptive_enabled": settings.enable_adaptive_confidence,
                "current_thresholds": stats["thresholds"],
                "intent_statistics": stats["intents"],
                "total_adjustments": stats["adjustments"],
                "recent_adjustments": stats["recent_adjustments"]
            }
        )
        
    except Exception as exc:
        logger.error(f"Confidence stats retrieval error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


@app.get("/api/session/{session_key}/journey")
async def get_journey_phases(session_key: str):
    """
    Get user journey phase history for a session.
    Shows phase transitions over time (discovery → interest → detail → booking).
    """
    try:
        session_manager = SessionManager()
        session_data = await session_manager.get_session(session_key)
        
        if not session_data:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={"error": "Session not found"}
            )
        
        phase_history = session_data.get("phase_history", [])
        current_phase = session_data.get("journey_phase", "discovery")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "session_key": session_key,
                "current_phase": current_phase,
                "phase_history": phase_history,
                "total_transitions": len(phase_history)
            }
        )
        
    except Exception as exc:
        logger.error(f"Journey phase retrieval error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


@app.get("/api/metrics")
async def get_performance_metrics():
    """
    Get comprehensive performance metrics.
    
    Issues addressed:
    - #28: Memory usage tracking
    - #29: CPU/resource monitoring
    - #30: Cache hit/miss rates
    - #31: Queue depth monitoring
    - #32: Concurrent request tracking
    """
    try:
        from .utils.metrics import get_metrics
        
        metrics = get_metrics()
        report = metrics.get_full_report()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "metrics": report
            }
        )
        
    except Exception as exc:
        logger.error(f"Metrics retrieval error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


@app.post("/api/feedback/user")
async def submit_user_feedback(request: Request):
    """
    Submit user feedback for a specific interaction.
    
    Expected payload:
    {
        "session_id": "whatsapp:+1234567890",
        "feedback": "positive" | "negative" | "neutral",
        "rating": 1-5 (optional),
        "comment": "optional comment"
    }
    """
    try:
        payload = await request.json()
        session_id = payload.get("session_id")
        feedback = payload.get("feedback")
        rating = payload.get("rating")
        comment = payload.get("comment", "")
        
        if not session_id or not feedback:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing session_id or feedback"
            )
        
        # Store feedback in session
        session_manager = SessionManager()
        session_data = session_manager.get(session_id) or {}
        
        if "feedback_history" not in session_data:
            session_data["feedback_history"] = []
        
        session_data["feedback_history"].append({
            "feedback": feedback,
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        })
        
        session_manager.put(session_id, session_data, ttl_minutes=120)
        
        logger.info(f"Feedback received for session {session_id}: {feedback}")
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "success",
                "message": "Thank you for your feedback!"
            }
        )
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"User feedback submission error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


# ============================================================================
# REMINDER ENDPOINTS
# ============================================================================

@app.post("/api/reminders/check")
async def check_reminders():
    """
    Manually trigger reminder check.
    Checks for upcoming appointments and sends reminders.
    """
    try:
        from .services.reminder_service import get_reminder_service
        
        reminder_service = get_reminder_service()
        result = await reminder_service.check_and_send_reminders()
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
        
    except Exception as exc:
        logger.error(f"Reminder check error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


@app.post("/api/reminders/send/{booking_id}")
async def send_manual_reminder(booking_id: str):
    """
    Send a manual reminder for a specific booking.
    
    Args:
        booking_id: Booking ID to send reminder for
    """
    try:
        from .services.reminder_service import get_reminder_service
        
        reminder_service = get_reminder_service()
        result = await reminder_service.send_manual_reminder(booking_id)
        
        if result.get("success"):
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "status": "success",
                    "message": "Reminder sent successfully",
                    "booking_id": booking_id
                }
            )
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "status": "error",
                    "error": result.get("error", "Failed to send reminder")
                }
            )
        
    except Exception as exc:
        logger.error(f"Manual reminder error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )


# Include additional routers
# HYBRID SYSTEM (now default!)
app.include_router(webhook_hybrid_router, tags=["webhook", "hybrid-architecture"])  # ✅ NEW: Now at /webhook
# OLD SYSTEM (backup at /api/v1)
app.include_router(webhook_router, prefix="/api/v1", tags=["webhook-legacy"])  # Old system moved
app.include_router(agent_router, prefix="/api", tags=["agent-api"])
app.include_router(manual_review_router, tags=["manual-review"])

logger.info("=" * 70)
logger.info("✅ API ENDPOINTS REGISTERED")
logger.info("   🚀 HYBRID SYSTEM (Reem):    /webhook           ← NEW DEFAULT!")
logger.info("   📦 Old System (Backup):     /api/v1/webhook    ← Fallback")
logger.info("   🔧 Agent API:               /api/*")
logger.info("   👤 Manual Review:           /manual-review/*")
logger.info("=" * 70)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.app_env == "development" else "An error occurred"
        }
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 60)
    logger.info("STARTING AGENT ORCHESTRATOR SYSTEM")
    logger.info("=" * 60)
    
    uvicorn.run(
        "app.main_unified:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower()
    )
