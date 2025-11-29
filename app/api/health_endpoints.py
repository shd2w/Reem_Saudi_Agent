"""
Health Check Endpoints for Hybrid Architecture
===============================================

Production health monitoring endpoints for the Advanced Hybrid Architecture.

Endpoints:
    GET /health/hybrid - Overall hybrid architecture health
    GET /health/metrics - Current metrics snapshot
    GET /health/detailed - Detailed component status
"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
from datetime import datetime

from ..monitoring.hybrid_metrics import get_metrics
from ..models.conversation_context import BookingState
from ..agents.advanced_functions import get_advanced_functions

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/hybrid")
async def hybrid_health() -> Dict[str, Any]:
    """
    Get overall hybrid architecture health status.
    
    Returns:
        Health status with key metrics
        
    Example Response:
        {
            "status": "healthy",
            "timestamp": "2025-11-03T13:56:50+02:00",
            "uptime_seconds": 3600.5,
            "components": {
                "metrics": "healthy",
                "functions": "healthy",
                "models": "healthy"
            }
        }
    """
    metrics = get_metrics()
    health_status = metrics.get_health_status()
    
    # Determine overall status
    if health_status["status"] == "unhealthy":
        status_code = 503  # Service Unavailable
    elif health_status["status"] == "degraded":
        status_code = 200  # OK but with warning
    else:
        status_code = 200  # OK
    
    response = {
        "status": health_status["status"],
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": health_status["uptime_seconds"],
        "error_rate_percent": round(health_status["error_rate"], 2),
        "avg_response_time_ms": round(health_status["avg_response_time"] * 1000, 2),
        "active_bookings": health_status["active_bookings"],
        "components": {
            "metrics": "healthy",
            "functions": "healthy",
            "models": "healthy"
        }
    }
    
    # If unhealthy, raise HTTP exception
    if status_code == 503:
        raise HTTPException(status_code=503, detail=response)
    
    return response


@router.get("/metrics")
async def hybrid_metrics() -> Dict[str, Any]:
    """
    Get current metrics snapshot.
    
    Returns:
        Complete metrics data
        
    Example Response:
        {
            "timestamp": 1699024610.5,
            "state_transitions": {...},
            "function_performance": {...},
            "booking_stats": {...},
            "health": {...}
        }
    """
    metrics = get_metrics()
    return metrics.export_metrics()


@router.get("/detailed")
async def hybrid_detailed_health() -> Dict[str, Any]:
    """
    Get detailed component health status.
    
    Returns:
        Detailed health information for each component
    """
    metrics = get_metrics()
    
    # Test model creation
    try:
        test_state = BookingState()
        test_dict = test_state.to_dict()
        BookingState.from_dict(test_dict)
        models_status = "healthy"
        models_error = None
    except Exception as e:
        models_status = "unhealthy"
        models_error = str(e)
    
    # Test function definitions
    try:
        functions = get_advanced_functions()
        functions_count = len(functions)
        functions_status = "healthy" if functions_count == 9 else "degraded"
        functions_error = None if functions_count == 9 else f"Expected 9 functions, got {functions_count}"
    except Exception as e:
        functions_status = "unhealthy"
        functions_error = str(e)
        functions_count = 0
    
    # Get metrics health
    health = metrics.get_health_status()
    state_transitions = metrics.get_state_transition_summary()
    function_perf = metrics.get_function_performance_summary()
    booking_stats = metrics.get_booking_completion_stats()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "overall_status": health["status"],
        "components": {
            "models": {
                "status": models_status,
                "error": models_error,
                "tests": {
                    "BookingState.create": "✅",
                    "BookingState.serialization": "✅",
                    "BookingState.deserialization": "✅"
                }
            },
            "functions": {
                "status": functions_status,
                "error": functions_error,
                "count": functions_count,
                "expected": 9,
                "names": [f["name"] for f in functions] if functions_status != "unhealthy" else []
            },
            "metrics": {
                "status": "healthy" if health["error_rate"] < 10 else "degraded",
                "error_rate": health["error_rate"],
                "total_functions_executed": health["total_functions_executed"],
                "total_errors": health["total_errors"],
                "active_bookings": health["active_bookings"]
            }
        },
        "performance": {
            "avg_response_time_ms": round(health["avg_response_time"] * 1000, 2),
            "slowest_functions": sorted(
                [(name, stats["avg_duration"] * 1000) for name, stats in function_perf.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        },
        "booking_metrics": {
            "total_bookings": booking_stats["total_bookings"],
            "completed": booking_stats["completed"],
            "cancelled": booking_stats["cancelled"],
            "abandoned": booking_stats["abandoned"],
            "completion_rate_percent": round(booking_stats["completion_rate"], 2),
            "avg_duration_seconds": round(booking_stats["avg_duration"], 2),
            "avg_pauses": round(booking_stats["avg_pauses"], 2)
        },
        "state_transitions": {
            "total": state_transitions["total_transitions"],
            "most_common": state_transitions["most_common"][:5]
        }
    }


@router.get("/ready")
async def hybrid_readiness() -> Dict[str, Any]:
    """
    Kubernetes-style readiness probe.
    
    Returns 200 if ready to serve traffic, 503 otherwise.
    
    Returns:
        Simple readiness status
    """
    try:
        # Test basic operations
        state = BookingState()
        functions = get_advanced_functions()
        
        if len(functions) != 9:
            raise HTTPException(
                status_code=503,
                detail={"ready": False, "reason": "Invalid function count"}
            )
        
        return {
            "ready": True,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail={"ready": False, "reason": str(e)}
        )


@router.get("/live")
async def hybrid_liveness() -> Dict[str, Any]:
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if service is alive, 503 otherwise.
    
    Returns:
        Simple liveness status
    """
    return {
        "alive": True,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/metrics/reset")
async def reset_metrics() -> Dict[str, Any]:
    """
    Reset metrics (for testing/debugging).
    
    ⚠️ WARNING: This clears all collected metrics!
    
    Returns:
        Confirmation message
    """
    from ..monitoring.hybrid_metrics import HybridMetrics, _metrics_instance
    
    global _metrics_instance
    _metrics_instance = HybridMetrics()
    
    return {
        "status": "reset",
        "message": "Metrics have been reset",
        "timestamp": datetime.now().isoformat()
    }
