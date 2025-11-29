"""
Service Cache Manager

Caches services from API to avoid excessive API calls.
Refreshes every 5 minutes automatically.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from loguru import logger


class ServiceCache:
    """
    Singleton cache for services to avoid redundant API calls.
    
    Features:
    - Caches services for 5 minutes
    - Thread-safe with asyncio lock
    - Auto-refresh on expiry
    - Reduces API calls from every request to every 5 minutes
    """
    
    _instance: Optional['ServiceCache'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._services: List[Dict] = []
        self._last_fetch: Optional[datetime] = None
        self._cache_duration = timedelta(minutes=5)
        self._fetch_lock = asyncio.Lock()
        self._initialized = True
        
        logger.info("🗄️ ServiceCache initialized (5-minute cache)")
    
    @property
    def is_expired(self) -> bool:
        """Check if cache is expired"""
        if self._last_fetch is None:
            return True
        return datetime.now() - self._last_fetch > self._cache_duration
    
    @property
    def cache_age_seconds(self) -> int:
        """Get cache age in seconds"""
        if self._last_fetch is None:
            return -1
        return int((datetime.now() - self._last_fetch).total_seconds())
    
    async def get_services(self, api_client, force_refresh: bool = False) -> List[Dict]:
        """
        Get services from cache or fetch from API if expired.
        
        Args:
            api_client: AgentApiClient instance
            force_refresh: Force refresh even if cache is valid
        
        Returns:
            List of service dictionaries
        """
        # Check if cache is valid and not forced refresh
        if not force_refresh and not self.is_expired and self._services:
            logger.info(f"✅ Using CACHED services ({len(self._services)} items) - age: {self.cache_age_seconds}s")
            return self._services
        
        # Use lock to prevent multiple simultaneous fetches
        async with self._fetch_lock:
            # Double-check after acquiring lock (another task might have refreshed)
            if not force_refresh and not self.is_expired and self._services:
                logger.info(f"✅ Using CACHED services ({len(self._services)} items) - refreshed by another task")
                return self._services
            
            # Fetch from API
            try:
                reason = "forced" if force_refresh else "expired" if self._last_fetch else "initial"
                logger.info(f"🔍 FETCHING SERVICES from API (reason: {reason})")
                
                # UPDATED: Use get_services() which now returns list directly
                services_list = await api_client.get_services(limit=100)
                
                if services_list:
                    self._services = services_list
                    self._last_fetch = datetime.now()
                    logger.info(f"✅ SERVICE CACHE UPDATED: {len(self._services)} services cached for 5 minutes")
                    logger.info(f"💾 Next refresh at: {(self._last_fetch + self._cache_duration).strftime('%H:%M:%S')}")
                else:
                    logger.warning("⚠️ API returned empty services - keeping old cache if exists")
                    if not self._services:
                        logger.error("🚨 No services available in cache or API!")
                
                return self._services
                
            except Exception as e:
                logger.error(f"❌ Failed to fetch services: {e}")
                # Return cached services even if expired (better than nothing)
                if self._services:
                    logger.warning(f"⚠️ Using STALE cache ({len(self._services)} items) - age: {self.cache_age_seconds}s")
                return self._services
    
    async def clear_cache(self):
        """Clear the cache (useful for testing)"""
        async with self._fetch_lock:
            self._services = []
            self._last_fetch = None
            logger.info("🗑️ Service cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cached_count": len(self._services),
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "age_seconds": self.cache_age_seconds,
            "is_expired": self.is_expired,
            "cache_duration_minutes": self._cache_duration.total_seconds() / 60
        }


# Singleton instance
service_cache = ServiceCache()
