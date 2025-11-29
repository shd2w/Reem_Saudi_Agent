"""
Hybrid Architecture Monitoring & Metrics
=========================================

Production-grade monitoring for the Advanced Hybrid Architecture.
Tracks performance, state transitions, and system health.

Usage:
    from app.monitoring.hybrid_metrics import HybridMetrics
    
    metrics = HybridMetrics()
    metrics.record_state_transition("idle", "active", session_id)
    metrics.record_function_execution("execute_booking_step", duration=0.5)
"""
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class StateTransitionMetric:
    """Metric for booking state transition"""
    from_state: str
    to_state: str
    session_id: str
    timestamp: float
    duration_in_previous_state: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionMetric:
    """Metric for function execution"""
    function_name: str
    duration: float
    success: bool
    timestamp: float
    session_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BookingFlowMetric:
    """Metric for complete booking flow"""
    session_id: str
    started_at: float
    completed_at: Optional[float] = None
    status: str = "in_progress"  # in_progress, completed, cancelled, abandoned
    steps_completed: List[str] = field(default_factory=list)
    pause_count: int = 0
    resume_count: int = 0
    total_duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class HybridMetrics:
    """
    Production monitoring for Advanced Hybrid Architecture.
    
    Tracks:
    - Booking state transitions
    - Function execution performance
    - Booking flow completion rates
    - Pause/resume patterns
    - Error rates
    - System health
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Initialize metrics collector.
        
        Args:
            max_history: Maximum number of metrics to keep in memory
        """
        self.max_history = max_history
        
        # State transition tracking
        self.state_transitions: deque = deque(maxlen=max_history)
        self.state_transition_counts: Dict[str, int] = defaultdict(int)
        
        # Function execution tracking
        self.function_executions: deque = deque(maxlen=max_history)
        self.function_durations: Dict[str, List[float]] = defaultdict(list)
        self.function_errors: Dict[str, int] = defaultdict(int)
        
        # Booking flow tracking
        self.active_bookings: Dict[str, BookingFlowMetric] = {}
        self.completed_bookings: deque = deque(maxlen=max_history)
        
        # Session state tracking (for duration calculations)
        self._session_states: Dict[str, tuple] = {}  # {session_id: (state, timestamp)}
        
        logger.info("✅ HybridMetrics initialized")
    
    def record_state_transition(
        self,
        from_state: str,
        to_state: str,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a booking state transition.
        
        Args:
            from_state: Previous state
            to_state: New state
            session_id: Session identifier
            metadata: Additional context
        """
        now = time.time()
        
        # Calculate duration in previous state
        duration = None
        if session_id in self._session_states:
            prev_state, prev_timestamp = self._session_states[session_id]
            if prev_state == from_state:
                duration = now - prev_timestamp
        
        # Create metric
        metric = StateTransitionMetric(
            from_state=from_state,
            to_state=to_state,
            session_id=session_id,
            timestamp=now,
            duration_in_previous_state=duration,
            metadata=metadata or {}
        )
        
        # Store metric
        self.state_transitions.append(metric)
        transition_key = f"{from_state}→{to_state}"
        self.state_transition_counts[transition_key] += 1
        
        # Update session state
        self._session_states[session_id] = (to_state, now)
        
        # Update booking flow
        self._update_booking_flow(session_id, to_state, now)
        
        duration_str = f"{duration:.2f}s" if duration else "N/A"
        logger.info(
            f"📊 State transition: {from_state} → {to_state} "
            f"(session: {session_id[:8]}, duration: {duration_str})"
        )
    
    def record_function_execution(
        self,
        function_name: str,
        duration: float,
        success: bool = True,
        session_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record function execution metrics.
        
        Args:
            function_name: Name of executed function
            duration: Execution time in seconds
            success: Whether execution succeeded
            session_id: Session identifier
            error: Error message if failed
            metadata: Additional context
        """
        metric = FunctionMetric(
            function_name=function_name,
            duration=duration,
            success=success,
            timestamp=time.time(),
            session_id=session_id,
            error=error,
            metadata=metadata or {}
        )
        
        # Store metric
        self.function_executions.append(metric)
        
        if success:
            self.function_durations[function_name].append(duration)
            # Keep only last 100 for each function
            if len(self.function_durations[function_name]) > 100:
                self.function_durations[function_name] = self.function_durations[function_name][-100:]
        else:
            self.function_errors[function_name] += 1
        
        status = "✅" if success else "❌"
        session_str = session_id[:8] if session_id else "N/A"
        logger.info(
            f"📊 Function {status}: {function_name} "
            f"({duration:.3f}s, session: {session_str})"
        )
    
    def start_booking_flow(
        self,
        session_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Mark the start of a booking flow.
        
        Args:
            session_id: Session identifier
            metadata: Additional context
        """
        if session_id not in self.active_bookings:
            self.active_bookings[session_id] = BookingFlowMetric(
                session_id=session_id,
                started_at=time.time(),
                metadata=metadata or {}
            )
            logger.info(f"📊 Booking flow started: {session_id[:8]}")
    
    def complete_booking_flow(
        self,
        session_id: str,
        status: str = "completed"
    ):
        """
        Mark booking flow as complete.
        
        Args:
            session_id: Session identifier
            status: Final status (completed, cancelled, abandoned)
        """
        if session_id in self.active_bookings:
            booking = self.active_bookings[session_id]
            booking.completed_at = time.time()
            booking.status = status
            booking.total_duration = booking.completed_at - booking.started_at
            
            # Move to completed
            self.completed_bookings.append(booking)
            del self.active_bookings[session_id]
            
            logger.info(
                f"📊 Booking flow {status}: {session_id[:8]} "
                f"(duration: {booking.total_duration:.2f}s, "
                f"steps: {len(booking.steps_completed)}, "
                f"pauses: {booking.pause_count})"
            )
    
    def _update_booking_flow(self, session_id: str, state: str, timestamp: float):
        """Update booking flow based on state transition"""
        if state == "active" and session_id not in self.active_bookings:
            self.start_booking_flow(session_id)
        
        if session_id in self.active_bookings:
            booking = self.active_bookings[session_id]
            
            if state == "paused":
                booking.pause_count += 1
            elif state == "active":
                if booking.pause_count > booking.resume_count:
                    booking.resume_count += 1
            elif state == "completed":
                self.complete_booking_flow(session_id, "completed")
            elif state == "cancelled":
                self.complete_booking_flow(session_id, "cancelled")
    
    def get_state_transition_summary(self) -> Dict[str, Any]:
        """
        Get summary of state transitions.
        
        Returns:
            Summary statistics
        """
        return {
            "total_transitions": len(self.state_transitions),
            "transition_counts": dict(self.state_transition_counts),
            "most_common": sorted(
                self.state_transition_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def get_function_performance_summary(self) -> Dict[str, Any]:
        """
        Get function performance summary.
        
        Returns:
            Performance statistics per function
        """
        summary = {}
        
        for func_name, durations in self.function_durations.items():
            if durations:
                summary[func_name] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "p50_duration": sorted(durations)[len(durations) // 2],
                    "p95_duration": sorted(durations)[int(len(durations) * 0.95)],
                    "errors": self.function_errors.get(func_name, 0)
                }
        
        return summary
    
    def get_booking_completion_stats(self) -> Dict[str, Any]:
        """
        Get booking flow completion statistics.
        
        Returns:
            Completion rate and timing statistics
        """
        total = len(self.completed_bookings)
        
        if total == 0:
            return {
                "total_completed": 0,
                "completion_rate": 0.0,
                "avg_duration": 0.0,
                "avg_pauses": 0.0
            }
        
        completed = sum(1 for b in self.completed_bookings if b.status == "completed")
        cancelled = sum(1 for b in self.completed_bookings if b.status == "cancelled")
        abandoned = sum(1 for b in self.completed_bookings if b.status == "abandoned")
        
        durations = [b.total_duration for b in self.completed_bookings if b.total_duration]
        pauses = [b.pause_count for b in self.completed_bookings]
        
        return {
            "total_bookings": total,
            "completed": completed,
            "cancelled": cancelled,
            "abandoned": abandoned,
            "completion_rate": (completed / total * 100) if total > 0 else 0.0,
            "avg_duration": sum(durations) / len(durations) if durations else 0.0,
            "avg_pauses": sum(pauses) / len(pauses) if pauses else 0.0,
            "avg_duration_completed": sum(
                b.total_duration for b in self.completed_bookings 
                if b.status == "completed" and b.total_duration
            ) / completed if completed > 0 else 0.0
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Health metrics and status
        """
        # Calculate error rates
        total_functions = len(self.function_executions)
        total_errors = sum(self.function_errors.values())
        error_rate = (total_errors / total_functions * 100) if total_functions > 0 else 0.0
        
        # Calculate avg response time
        all_durations = [m.duration for m in self.function_executions if m.success]
        avg_response_time = sum(all_durations) / len(all_durations) if all_durations else 0.0
        
        # Determine health status
        if error_rate > 10.0:
            status = "unhealthy"
        elif error_rate > 5.0:
            status = "degraded"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "active_bookings": len(self.active_bookings),
            "total_functions_executed": total_functions,
            "total_errors": total_errors,
            "uptime_seconds": time.time() - (
                self.state_transitions[0].timestamp 
                if self.state_transitions else time.time()
            )
        }
    
    def export_metrics(self) -> Dict[str, Any]:
        """
        Export all metrics for external monitoring systems.
        
        Returns:
            Complete metrics snapshot
        """
        return {
            "timestamp": time.time(),
            "state_transitions": self.get_state_transition_summary(),
            "function_performance": self.get_function_performance_summary(),
            "booking_stats": self.get_booking_completion_stats(),
            "health": self.get_health_status()
        }
    
    def log_summary(self):
        """Log comprehensive metrics summary"""
        logger.info("\n" + "="*60)
        logger.info("  HYBRID ARCHITECTURE METRICS SUMMARY")
        logger.info("="*60)
        
        # Health
        health = self.get_health_status()
        logger.info(f"\n🏥 Health Status: {health['status'].upper()}")
        logger.info(f"   Error Rate: {health['error_rate']:.2f}%")
        logger.info(f"   Avg Response: {health['avg_response_time']:.3f}s")
        logger.info(f"   Active Bookings: {health['active_bookings']}")
        
        # State Transitions
        transitions = self.get_state_transition_summary()
        logger.info(f"\n🔄 State Transitions: {transitions['total_transitions']} total")
        for transition, count in transitions['most_common']:
            logger.info(f"   {transition}: {count}")
        
        # Function Performance
        perf = self.get_function_performance_summary()
        logger.info(f"\n⚡ Function Performance:")
        for func, stats in sorted(perf.items(), key=lambda x: x[1]['avg_duration'], reverse=True)[:5]:
            logger.info(
                f"   {func}: {stats['avg_duration']:.3f}s avg "
                f"(count: {stats['count']}, errors: {stats['errors']})"
            )
        
        # Booking Stats
        bookings = self.get_booking_completion_stats()
        logger.info(f"\n📊 Booking Completion:")
        logger.info(f"   Total: {bookings['total_bookings']}")
        logger.info(f"   Completed: {bookings['completed']}")
        logger.info(f"   Cancelled: {bookings['cancelled']}")
        logger.info(f"   Completion Rate: {bookings['completion_rate']:.1f}%")
        logger.info(f"   Avg Duration: {bookings['avg_duration']:.2f}s")
        logger.info(f"   Avg Pauses: {bookings['avg_pauses']:.1f}")
        
        logger.info("\n" + "="*60 + "\n")


# Global singleton instance
_metrics_instance: Optional[HybridMetrics] = None


def get_metrics() -> HybridMetrics:
    """
    Get global metrics instance (singleton).
    
    Returns:
        HybridMetrics instance
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = HybridMetrics()
    return _metrics_instance
