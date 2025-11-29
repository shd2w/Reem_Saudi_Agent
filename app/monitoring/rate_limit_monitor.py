"""
Rate Limit Monitoring and Alerting System

Tracks rate limit hits and alerts operations team when thresholds exceeded.
"""
import time
from typing import Dict, List, Optional
from loguru import logger
import redis.asyncio as redis
from dataclasses import dataclass
from datetime import datetime


@dataclass
class RateLimitEvent:
    """Single rate limit event."""
    timestamp: float
    limit_type: str  # "RATE_SPIKE", "ACCOUNT_LIMIT", "QUOTA_EXHAUSTED"
    retry_after: int
    endpoint: str
    phone_number: Optional[str] = None


class RateLimitMonitor:
    """
    Monitor rate limit hits and trigger alerts.
    
    Features:
    - Track hits per hour/day
    - Alert on threshold violations
    - Predict quota exhaustion
    - Generate usage reports
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.events_key = "rate_limit:events"
        self.stats_key = "rate_limit:stats"
        
        # Alert thresholds
        self.WARNING_THRESHOLD_HOUR = 10   # 10 hits/hour = warning
        self.CRITICAL_THRESHOLD_HOUR = 50  # 50 hits/hour = critical
        self.WARNING_THRESHOLD_DAY = 100   # 100 hits/day = warning
        self.CRITICAL_THRESHOLD_DAY = 500  # 500 hits/day = critical
    
    async def record_rate_limit(
        self,
        limit_type: str,
        retry_after: int,
        endpoint: str = "send_message",
        phone_number: Optional[str] = None
    ):
        """
        Record a rate limit event.
        
        Args:
            limit_type: Type of limit (RATE_SPIKE, ACCOUNT_LIMIT, QUOTA_EXHAUSTED)
            retry_after: Seconds until retry allowed
            endpoint: API endpoint that was rate limited
            phone_number: Optional phone number
        """
        event = RateLimitEvent(
            timestamp=time.time(),
            limit_type=limit_type,
            retry_after=retry_after,
            endpoint=endpoint,
            phone_number=phone_number
        )
        
        # Store event in Redis (expire after 7 days)
        event_data = {
            "timestamp": event.timestamp,
            "limit_type": limit_type,
            "retry_after": retry_after,
            "endpoint": endpoint,
            "phone": phone_number or "unknown"
        }
        
        # Add to time-series sorted set
        await self.redis.zadd(
            self.events_key,
            {str(event_data): event.timestamp}
        )
        
        # Set TTL (7 days)
        await self.redis.expire(self.events_key, 7 * 24 * 3600)
        
        # Update counters
        await self._update_counters(limit_type)
        
        # Check if alerts needed
        await self._check_alerts()
        
        logger.warning(
            f"📊 [MONITOR] Rate limit recorded: {limit_type} "
            f"(retry_after: {retry_after}s, endpoint: {endpoint})"
        )
    
    async def _update_counters(self, limit_type: str):
        """Update hourly and daily counters."""
        now = time.time()
        hour_key = f"{self.stats_key}:hour:{int(now / 3600)}"
        day_key = f"{self.stats_key}:day:{int(now / 86400)}"
        
        # Increment counters
        await self.redis.incr(f"{hour_key}:total")
        await self.redis.incr(f"{hour_key}:{limit_type}")
        await self.redis.incr(f"{day_key}:total")
        await self.redis.incr(f"{day_key}:{limit_type}")
        
        # Set TTL
        await self.redis.expire(hour_key, 7200)  # 2 hours
        await self.redis.expire(day_key, 172800)  # 2 days
    
    async def _check_alerts(self):
        """Check if alert thresholds exceeded."""
        now = time.time()
        hour_key = f"{self.stats_key}:hour:{int(now / 3600)}"
        day_key = f"{self.stats_key}:day:{int(now / 86400)}"
        
        # Get current counts
        hits_this_hour = await self.redis.get(f"{hour_key}:total") or 0
        hits_this_hour = int(hits_this_hour)
        
        hits_today = await self.redis.get(f"{day_key}:total") or 0
        hits_today = int(hits_today)
        
        # Check thresholds
        if hits_this_hour >= self.CRITICAL_THRESHOLD_HOUR:
            await self._send_alert(
                level="CRITICAL",
                message=f"WaSender rate limit hit {hits_this_hour} times this hour! "
                        f"(threshold: {self.CRITICAL_THRESHOLD_HOUR})",
                metrics={"hits_this_hour": hits_this_hour}
            )
        elif hits_this_hour >= self.WARNING_THRESHOLD_HOUR:
            await self._send_alert(
                level="WARNING",
                message=f"WaSender rate limit hit {hits_this_hour} times this hour "
                        f"(threshold: {self.WARNING_THRESHOLD_HOUR})",
                metrics={"hits_this_hour": hits_this_hour}
            )
        
        if hits_today >= self.CRITICAL_THRESHOLD_DAY:
            await self._send_alert(
                level="CRITICAL",
                message=f"WaSender rate limit hit {hits_today} times today! "
                        f"(threshold: {self.CRITICAL_THRESHOLD_DAY})",
                metrics={"hits_today": hits_today}
            )
        elif hits_today >= self.WARNING_THRESHOLD_DAY:
            await self._send_alert(
                level="WARNING",
                message=f"WaSender rate limit hit {hits_today} times today "
                        f"(threshold: {self.WARNING_THRESHOLD_DAY})",
                metrics={"hits_today": hits_today}
            )
    
    async def _send_alert(self, level: str, message: str, metrics: Dict):
        """
        Send alert to operations team.
        
        This is a placeholder - integrate with your alerting system:
        - PagerDuty
        - Slack
        - Email
        - SMS
        - etc.
        """
        if level == "CRITICAL":
            logger.error(f"🚨 [ALERT-CRITICAL] {message}")
            # TODO: Integrate with PagerDuty/Slack/etc.
            # await pagerduty.trigger_incident(message, metrics)
        else:
            logger.warning(f"⚠️ [ALERT-WARNING] {message}")
            # TODO: Send to Slack channel
            # await slack.send_message("#ops-alerts", message)
    
    async def get_stats(self, period: str = "hour") -> Dict:
        """
        Get rate limit statistics.
        
        Args:
            period: "hour", "day", or "week"
        
        Returns:
            Statistics dict
        """
        now = time.time()
        
        if period == "hour":
            key_prefix = f"{self.stats_key}:hour:{int(now / 3600)}"
            period_label = "this hour"
        elif period == "day":
            key_prefix = f"{self.stats_key}:day:{int(now / 86400)}"
            period_label = "today"
        elif period == "week":
            # Aggregate last 7 days
            stats = {
                "total": 0,
                "RATE_SPIKE": 0,
                "ACCOUNT_LIMIT": 0,
                "QUOTA_EXHAUSTED": 0
            }
            for i in range(7):
                day = int((now - i * 86400) / 86400)
                key_prefix = f"{self.stats_key}:day:{day}"
                
                total = await self.redis.get(f"{key_prefix}:total") or 0
                stats["total"] += int(total)
                
                for limit_type in ["RATE_SPIKE", "ACCOUNT_LIMIT", "QUOTA_EXHAUSTED"]:
                    count = await self.redis.get(f"{key_prefix}:{limit_type}") or 0
                    stats[limit_type] += int(count)
            
            stats["period"] = "last 7 days"
            return stats
        else:
            raise ValueError(f"Invalid period: {period}")
        
        # Get counts
        total = await self.redis.get(f"{key_prefix}:total") or 0
        rate_spike = await self.redis.get(f"{key_prefix}:RATE_SPIKE") or 0
        account_limit = await self.redis.get(f"{key_prefix}:ACCOUNT_LIMIT") or 0
        quota_exhausted = await self.redis.get(f"{key_prefix}:QUOTA_EXHAUSTED") or 0
        
        return {
            "period": period_label,
            "total": int(total),
            "RATE_SPIKE": int(rate_spike),
            "ACCOUNT_LIMIT": int(account_limit),
            "QUOTA_EXHAUSTED": int(quota_exhausted)
        }
    
    async def predict_quota_exhaustion(self) -> Optional[Dict]:
        """
        Predict when quota will be exhausted based on current rate.
        
        Returns:
            Prediction dict or None
        """
        # Get events from last hour
        now = time.time()
        one_hour_ago = now - 3600
        
        events = await self.redis.zrangebyscore(
            self.events_key,
            min=one_hour_ago,
            max=now
        )
        
        if not events:
            return None
        
        # Count quota exhaustion events
        quota_exhausted_count = sum(
            1 for e in events 
            if '"QUOTA_EXHAUSTED"' in str(e)
        )
        
        if quota_exhausted_count > 0:
            return {
                "status": "CRITICAL",
                "message": f"Quota exhausted {quota_exhausted_count} times in last hour",
                "recommendation": "Upgrade WaSender plan or reduce sending rate"
            }
        
        # Calculate rate
        rate_per_hour = len(events)
        
        # Estimate quota (assuming 1000/day = 42/hour)
        estimated_quota_per_hour = 42
        
        if rate_per_hour > estimated_quota_per_hour * 0.8:
            hours_until_exhaustion = estimated_quota_per_hour / rate_per_hour
            
            return {
                "status": "WARNING",
                "message": f"Rate limit hits approaching quota ({rate_per_hour}/hour)",
                "hours_until_exhaustion": round(hours_until_exhaustion, 1),
                "recommendation": "Monitor closely or reduce sending rate"
            }
        
        return {
            "status": "OK",
            "message": "Rate limit usage within normal range",
            "rate_per_hour": rate_per_hour
        }
    
    async def get_dashboard_data(self) -> Dict:
        """Get comprehensive dashboard data."""
        hour_stats = await self.get_stats("hour")
        day_stats = await self.get_stats("day")
        week_stats = await self.get_stats("week")
        prediction = await self.predict_quota_exhaustion()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "stats": {
                "hour": hour_stats,
                "day": day_stats,
                "week": week_stats
            },
            "prediction": prediction,
            "thresholds": {
                "warning_hour": self.WARNING_THRESHOLD_HOUR,
                "critical_hour": self.CRITICAL_THRESHOLD_HOUR,
                "warning_day": self.WARNING_THRESHOLD_DAY,
                "critical_day": self.CRITICAL_THRESHOLD_DAY
            }
        }
