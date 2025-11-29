"""
Performance Metrics Tracking
=============================
Comprehensive metrics for monitoring system performance, resource usage, and bottlenecks.

Issues Addressed:
- #28: Memory usage tracking
- #29: CPU/resource monitoring
- #30: Cache hit/miss rates
- #31: Queue depth monitoring
- #32: Concurrent request tracking
"""

import time
import psutil
import asyncio
from typing import Dict, Optional
from collections import defaultdict, deque
from datetime import datetime
from loguru import logger


class PerformanceMetrics:
    """
    Singleton class for tracking system performance metrics.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if PerformanceMetrics._initialized:
            return
        
        # Memory tracking
        self.process = psutil.Process()
        self.memory_samples = deque(maxlen=100)  # Last 100 samples
        
        # CPU tracking
        self.cpu_samples = deque(maxlen=100)
        
        # Cache metrics (Issue #30)
        self.cache_hits = defaultdict(int)
        self.cache_misses = defaultdict(int)
        
        # Request tracking (Issue #32)
        self.concurrent_requests = 0
        self.max_concurrent_requests = 0
        self.total_requests = 0
        self.request_start_times = {}  # request_id -> start_time
        
        # Queue depth (Issue #31)
        self.queue_depths = defaultdict(lambda: deque(maxlen=100))
        
        # Response time tracking
        self.response_times = deque(maxlen=1000)
        
        # Component-specific metrics
        self.component_metrics = defaultdict(lambda: {
            "calls": 0,
            "total_time": 0,
            "errors": 0
        })
        
        PerformanceMetrics._initialized = True
        logger.info("📊 Performance metrics tracking initialized")
    
    # ========================================================================
    # MEMORY TRACKING (Issue #28)
    # ========================================================================
    
    def record_memory_usage(self):
        """Record current memory usage"""
        try:
            mem_info = self.process.memory_info()
            memory_mb = mem_info.rss / 1024 / 1024  # Convert to MB
            
            self.memory_samples.append({
                "timestamp": time.time(),
                "rss_mb": memory_mb,
                "vms_mb": mem_info.vms / 1024 / 1024,
                "percent": self.process.memory_percent()
            })
            
            return memory_mb
        except Exception as exc:
            logger.error(f"Failed to record memory usage: {exc}")
            return None
    
    def get_memory_stats(self) -> Dict:
        """Get memory usage statistics"""
        if not self.memory_samples:
            self.record_memory_usage()
        
        if not self.memory_samples:
            return {"error": "No memory samples"}
        
        recent = list(self.memory_samples)
        rss_values = [s["rss_mb"] for s in recent]
        
        return {
            "current_mb": rss_values[-1] if rss_values else 0,
            "avg_mb": sum(rss_values) / len(rss_values),
            "max_mb": max(rss_values),
            "min_mb": min(rss_values),
            "samples": len(rss_values)
        }
    
    # ========================================================================
    # CPU TRACKING (Issue #29)
    # ========================================================================
    
    def record_cpu_usage(self):
        """Record current CPU usage"""
        try:
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            self.cpu_samples.append({
                "timestamp": time.time(),
                "percent": cpu_percent,
                "num_threads": self.process.num_threads()
            })
            
            return cpu_percent
        except Exception as exc:
            logger.error(f"Failed to record CPU usage: {exc}")
            return None
    
    def get_cpu_stats(self) -> Dict:
        """Get CPU usage statistics"""
        if not self.cpu_samples:
            self.record_cpu_usage()
        
        if not self.cpu_samples:
            return {"error": "No CPU samples"}
        
        recent = list(self.cpu_samples)
        cpu_values = [s["percent"] for s in recent]
        
        return {
            "current_percent": cpu_values[-1] if cpu_values else 0,
            "avg_percent": sum(cpu_values) / len(cpu_values),
            "max_percent": max(cpu_values),
            "num_threads": recent[-1]["num_threads"] if recent else 0,
            "samples": len(cpu_values)
        }
    
    # ========================================================================
    # CACHE METRICS (Issue #30)
    # ========================================================================
    
    def record_cache_hit(self, cache_name: str):
        """Record a cache hit"""
        self.cache_hits[cache_name] += 1
        logger.debug(f"✅ CACHE HIT: {cache_name}")
    
    def record_cache_miss(self, cache_name: str):
        """Record a cache miss"""
        self.cache_misses[cache_name] += 1
        logger.debug(f"❌ CACHE MISS: {cache_name}")
    
    def get_cache_stats(self, cache_name: Optional[str] = None) -> Dict:
        """Get cache hit/miss statistics"""
        if cache_name:
            hits = self.cache_hits[cache_name]
            misses = self.cache_misses[cache_name]
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            return {
                "cache": cache_name,
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate_percent": round(hit_rate, 2)
            }
        else:
            # All caches
            all_caches = set(list(self.cache_hits.keys()) + list(self.cache_misses.keys()))
            return {
                cache: self.get_cache_stats(cache)
                for cache in all_caches
            }
    
    # ========================================================================
    # QUEUE DEPTH (Issue #31)
    # ========================================================================
    
    def record_queue_depth(self, queue_name: str, depth: int):
        """Record queue depth"""
        self.queue_depths[queue_name].append({
            "timestamp": time.time(),
            "depth": depth
        })
        
        # Log if queue is growing
        if depth > 10:
            logger.warning(f"⚠️ QUEUE DEPTH HIGH: {queue_name} has {depth} items")
    
    def get_queue_stats(self, queue_name: Optional[str] = None) -> Dict:
        """Get queue depth statistics"""
        if queue_name:
            samples = list(self.queue_depths[queue_name])
            if not samples:
                return {"queue": queue_name, "error": "No samples"}
            
            depths = [s["depth"] for s in samples]
            return {
                "queue": queue_name,
                "current": depths[-1] if depths else 0,
                "avg": sum(depths) / len(depths),
                "max": max(depths),
                "samples": len(depths)
            }
        else:
            # All queues
            return {
                queue: self.get_queue_stats(queue)
                for queue in self.queue_depths.keys()
            }
    
    # ========================================================================
    # CONCURRENT REQUESTS (Issue #32)
    # ========================================================================
    
    def start_request(self, request_id: str):
        """Mark request as started"""
        self.concurrent_requests += 1
        self.total_requests += 1
        self.request_start_times[request_id] = time.time()
        
        # Track peak concurrency
        if self.concurrent_requests > self.max_concurrent_requests:
            self.max_concurrent_requests = self.concurrent_requests
            logger.info(f"📊 NEW PEAK CONCURRENCY: {self.concurrent_requests} requests")
        
        # Log high concurrency
        if self.concurrent_requests > 10:
            logger.warning(f"⚠️ HIGH CONCURRENCY: {self.concurrent_requests} concurrent requests")
    
    def end_request(self, request_id: str):
        """Mark request as completed"""
        self.concurrent_requests = max(0, self.concurrent_requests - 1)
        
        # Record response time
        if request_id in self.request_start_times:
            duration = time.time() - self.request_start_times[request_id]
            self.response_times.append(duration)
            del self.request_start_times[request_id]
            return duration
        return None
    
    def get_request_stats(self) -> Dict:
        """Get request statistics"""
        response_times_list = list(self.response_times)
        
        return {
            "concurrent": self.concurrent_requests,
            "max_concurrent": self.max_concurrent_requests,
            "total_requests": self.total_requests,
            "avg_response_time_ms": (sum(response_times_list) / len(response_times_list) * 1000) if response_times_list else 0,
            "max_response_time_ms": max(response_times_list) * 1000 if response_times_list else 0,
            "min_response_time_ms": min(response_times_list) * 1000 if response_times_list else 0
        }
    
    # ========================================================================
    # COMPONENT METRICS
    # ========================================================================
    
    def record_component_call(self, component: str, duration: float, error: bool = False):
        """Record a component call"""
        self.component_metrics[component]["calls"] += 1
        self.component_metrics[component]["total_time"] += duration
        if error:
            self.component_metrics[component]["errors"] += 1
    
    def get_component_stats(self) -> Dict:
        """Get component performance statistics"""
        stats = {}
        for component, metrics in self.component_metrics.items():
            calls = metrics["calls"]
            avg_time = (metrics["total_time"] / calls) if calls > 0 else 0
            error_rate = (metrics["errors"] / calls * 100) if calls > 0 else 0
            
            stats[component] = {
                "calls": calls,
                "avg_time_ms": round(avg_time * 1000, 2),
                "total_time_s": round(metrics["total_time"], 2),
                "errors": metrics["errors"],
                "error_rate_percent": round(error_rate, 2)
            }
        
        return stats
    
    # ========================================================================
    # COMPREHENSIVE REPORT
    # ========================================================================
    
    def get_full_report(self) -> Dict:
        """Get comprehensive performance report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "memory": self.get_memory_stats(),
            "cpu": self.get_cpu_stats(),
            "cache": self.get_cache_stats(),
            "queues": self.get_queue_stats(),
            "requests": self.get_request_stats(),
            "components": self.get_component_stats()
        }
    
    def log_metrics_summary(self):
        """Log a summary of current metrics"""
        logger.info("=" * 60)
        logger.info("📊 PERFORMANCE METRICS SUMMARY")
        
        # Memory
        mem = self.get_memory_stats()
        if "error" not in mem:
            logger.info(f"💾 Memory: {mem['current_mb']:.1f} MB (avg: {mem['avg_mb']:.1f} MB, max: {mem['max_mb']:.1f} MB)")
        
        # CPU
        cpu = self.get_cpu_stats()
        if "error" not in cpu:
            logger.info(f"⚙️ CPU: {cpu['current_percent']:.1f}% (avg: {cpu['avg_percent']:.1f}%, threads: {cpu['num_threads']})")
        
        # Requests
        req = self.get_request_stats()
        logger.info(f"📨 Requests: {req['concurrent']} concurrent (peak: {req['max_concurrent']}, total: {req['total_requests']})")
        logger.info(f"⏱️ Response Time: avg {req['avg_response_time_ms']:.0f}ms (max: {req['max_response_time_ms']:.0f}ms)")
        
        # Cache
        cache_stats = self.get_cache_stats()
        if cache_stats:
            logger.info("💾 Cache Hit Rates:")
            for cache_name, stats in cache_stats.items():
                if isinstance(stats, dict) and "hit_rate_percent" in stats:
                    logger.info(f"   - {cache_name}: {stats['hit_rate_percent']:.1f}% ({stats['hits']}/{stats['total']})")
        
        # Queues
        queue_stats = self.get_queue_stats()
        if queue_stats:
            logger.info("📋 Queue Depths:")
            for queue_name, stats in queue_stats.items():
                if isinstance(stats, dict) and "current" in stats:
                    logger.info(f"   - {queue_name}: {stats['current']} items (avg: {stats['avg']:.1f}, max: {stats['max']})")
        
        logger.info("=" * 60)


# Global instance
_metrics = None

def get_metrics() -> PerformanceMetrics:
    """Get or create global metrics instance"""
    global _metrics
    if _metrics is None:
        _metrics = PerformanceMetrics()
    return _metrics


# Context manager for tracking component performance
class track_component:
    """Context manager for tracking component performance"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.start_time = None
        self.metrics = get_metrics()
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        error = exc_type is not None
        self.metrics.record_component_call(self.component_name, duration, error)
        
        if duration > 1.0:  # Log slow operations
            logger.warning(f"⏱️ SLOW COMPONENT: {self.component_name} took {duration:.2f}s")
        
        return False  # Don't suppress exceptions
