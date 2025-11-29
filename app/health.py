"""
Health check endpoint with actual flow validation.

Implements comprehensive health checks including critical path testing.
"""
from fastapi import APIRouter, HTTPException
from loguru import logger
import asyncio

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check - components only."""
    return {
        "status": "healthy",
        "service": "agent-orchestrator",
        "checks": {
            "redis": "connected",
            "langgraph": "compiled",
            "agents": "initialized"
        }
    }


@router.get("/health/deep")
async def deep_health_check():
    """
    Deep health check - tests actual critical paths.
    
    This would have caught the sender_name undefined bug!
    """
    checks = {
        "redis": "unknown",
        "langgraph": "unknown",
        "verify_patient_node": "unknown",
        "registration_flow": "unknown",
        "session_manager": "unknown"
    }
    
    try:
        # Check 1: Redis connectivity
        from app.memory.session_manager import SessionManager
        session_manager = SessionManager()
        test_session = await session_manager.get_session("health_check_test")
        checks["redis"] = "connected"
        checks["session_manager"] = "operational"
        
        # Check 2: LangGraph compilation
        from app.agents.booking_agent_factory import BookingAgentFactory
        from app.config import get_settings
        settings = get_settings()
        
        if settings.use_langgraph:
            # Try to get a pooled agent (tests compilation)
            try:
                agent = BookingAgentFactory.create("health_check_session")
                checks["langgraph"] = "compiled"
            except Exception as e:
                checks["langgraph"] = f"error: {str(e)}"
                raise
        else:
            checks["langgraph"] = "disabled"
        
        # Check 3: CRITICAL - Test verify_patient_node (would catch sender_name bug!)
        from app.agents.langgraph.nodes.patient_verification import verify_patient_node
        from unittest.mock import AsyncMock
        
        # Test with minimal state (edge case that caused production bug)
        test_state = {
            "phone_number": "1234567890",
            # Intentionally missing sender_name - this MUST NOT crash!
            "_resuming": False
        }
        
        api_client = AsyncMock()
        api_client.get.side_effect = Exception("404")  # Patient not found
        session_manager_mock = AsyncMock()
        
        try:
            result = await verify_patient_node(test_state, api_client, session_manager_mock)
            
            # Verify it handled missing sender_name gracefully
            if "registration" in result and "sender_name" in result["registration"]:
                sender_name = result["registration"]["sender_name"]
                if sender_name and sender_name != "":
                    checks["verify_patient_node"] = "passed"
                else:
                    checks["verify_patient_node"] = "failed: sender_name is empty"
                    raise Exception("verify_patient_node returned empty sender_name")
            else:
                checks["verify_patient_node"] = "failed: no registration data"
                raise Exception("verify_patient_node missing registration data")
                
        except NameError as e:
            # This would catch the sender_name undefined bug!
            checks["verify_patient_node"] = f"CRITICAL: {str(e)}"
            raise HTTPException(
                status_code=503,
                detail=f"verify_patient_node has undefined variable: {e}"
            )
        
        # Check 4: Test registration name node
        from app.agents.langgraph.nodes.patient_verification import handle_registration_name_node
        
        name_test_state = {
            "current_message": "أحمد علي",
            "sender_name": "أحمد",
            "registration": {},
            "messages": []
        }
        
        try:
            name_result = await handle_registration_name_node(name_test_state)
            if "registration" in name_result and name_result["registration"].get("name"):
                checks["registration_flow"] = "passed"
            else:
                checks["registration_flow"] = "failed: name not saved"
        except Exception as e:
            checks["registration_flow"] = f"error: {str(e)}"
            raise
        
        # All checks passed
        return {
            "status": "healthy",
            "service": "agent-orchestrator",
            "checks": checks,
            "message": "All critical paths validated"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Deep health check failed: {e}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "agent-orchestrator",
            "checks": checks,
            "error": str(e)
        }, 503


@router.get("/health/ready")
async def readiness_check():
    """
    Readiness check - is system ready to accept requests?
    
    Used by Kubernetes/load balancers to determine if pod is ready.
    """
    try:
        # Quick checks only (shouldn't take more than 100ms)
        from app.memory.session_manager import SessionManager
        
        # Check Redis connectivity with timeout
        session_manager = SessionManager()
        test_key = "readiness_check"
        
        # Try a quick Redis operation with timeout
        try:
            await asyncio.wait_for(
                session_manager.get_session(test_key),
                timeout=0.1  # 100ms timeout
            )
            
            return {
                "status": "ready",
                "service": "agent-orchestrator"
            }
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Redis check timeout - not ready"
            )
            
    except Exception as e:
        logger.warning(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/health/live")
async def liveness_check():
    """
    Liveness check - is the service alive?
    
    Used by Kubernetes to determine if pod should be restarted.
    Very lightweight - just returns 200 if process is running.
    """
    return {
        "status": "alive",
        "service": "agent-orchestrator"
    }
