"""
Token Manager - Centralized JWT Token Management
=================================================
Handles automatic login, token storage, refresh, and expiry detection.

Features:
- Automatic login with credentials
- 12-hour access token management
- 6-hour refresh token support
- Automatic token refresh before expiry
- Thread-safe token storage
- File-based token persistence (tokens.json)
- Memory caching for performance

Author: Agent Orchestrator Team
Version: 2.0.0
"""

import asyncio
import json
import base64
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
from loguru import logger
import httpx


class TokenManager:
    """
    Centralized token management system.
    
    Manages:
    - Access tokens (12-hour validity)
    - Refresh tokens (6-hour validity)
    - Automatic refresh before expiry
    - Persistent storage in tokens.json
    """
    
    def __init__(
        self,
        login_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token_file: str = "tokens.json",
        pre_existing_token: Optional[str] = None
    ):
        """
        Initialize token manager.
        
        Args:
            login_url: API login endpoint (optional if using pre_existing_token)
            username: Login username (optional if using pre_existing_token)
            password: Login password (optional if using pre_existing_token)
            token_file: Path to token storage file
            pre_existing_token: Optional pre-existing unlimited token (no refresh needed)
        """
        self.login_url = login_url
        self.username = username
        self.password = password
        self.token_file = Path(token_file)
        
        # In-memory token storage
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._access_token_expiry: Optional[datetime] = None
        self._refresh_token_expiry: Optional[datetime] = None
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        # If pre-existing token provided, use it but ALSO validate credentials for refresh capability
        if pre_existing_token:
            # Validate credentials are provided for token refresh capability
            if not all([login_url, username, password]):
                logger.warning("⚠️ Pre-existing token provided but no credentials - cannot refresh if token expires!")
            
            self._access_token = pre_existing_token
            self._access_token_expiry = self._parse_jwt_expiry(pre_existing_token)
            if self._access_token_expiry:
                self._save_tokens_to_file()
                logger.info(f"✅ TokenManager initialized with pre-existing token (expires: {self._access_token_expiry})")
                if login_url and username:
                    logger.info(f"✅ Token refresh available using {username}@{login_url}")
            else:
                # Token might be unlimited/non-expiring - set far future expiry
                self._access_token_expiry = datetime.now() + timedelta(days=3650)  # 10 years
                logger.info("✅ TokenManager initialized with unlimited token (no expiry detected)")
        else:
            # Validate required credentials for login-based auth
            if not all([login_url, username, password]):
                raise ValueError("login_url, username, and password are required when not using pre_existing_token")
            
            # Load tokens from file if exists
            self._load_tokens_from_file()
            logger.info(f"TokenManager initialized for {username}")
    
    def _parse_jwt_expiry(self, jwt_token: str) -> Optional[datetime]:
        """
        Parse JWT token to extract expiry time.
        
        Args:
            jwt_token: JWT token string
            
        Returns:
            Expiry datetime or None if parsing fails
        """
        try:
            parts = jwt_token.split(".")
            if len(parts) < 2:
                return None
            
            # Decode payload (add padding if needed)
            payload_b64 = parts[1] + "=" * (-len(parts[1]) % 4)
            payload_bytes = base64.urlsafe_b64decode(payload_b64.encode("utf-8"))
            payload = json.loads(payload_bytes.decode("utf-8"))
            
            # Extract expiry timestamp
            exp = payload.get("exp")
            if isinstance(exp, int):
                return datetime.fromtimestamp(exp, tz=timezone.utc)
            
            return None
        except Exception as exc:
            logger.warning(f"Failed to parse JWT expiry: {exc}")
            return None
    
    def _load_tokens_from_file(self) -> None:
        """Load tokens from persistent storage file"""
        try:
            if self.token_file.exists():
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
                
                self._access_token = data.get("access_token")
                self._refresh_token = data.get("refresh_token")
                
                # Parse expiry times
                if self._access_token:
                    self._access_token_expiry = self._parse_jwt_expiry(self._access_token)
                
                if self._refresh_token:
                    self._refresh_token_expiry = self._parse_jwt_expiry(self._refresh_token)
                
                logger.info("Tokens loaded from file")
        except Exception as exc:
            logger.warning(f"Failed to load tokens from file: {exc}")
    
    def _save_tokens_to_file(self) -> None:
        """Save tokens to persistent storage file"""
        try:
            data = {
                "access_token": self._access_token,
                "refresh_token": self._refresh_token,
                "access_token_expiry": self._access_token_expiry.isoformat() if self._access_token_expiry else None,
                "refresh_token_expiry": self._refresh_token_expiry.isoformat() if self._refresh_token_expiry else None,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Tokens saved to file")
        except Exception as exc:
            logger.error(f"Failed to save tokens to file: {exc}")
    
    def _is_token_valid(self, expiry: Optional[datetime], buffer_minutes: int = 10) -> bool:
        """
        Check if token is still valid.
        
        Args:
            expiry: Token expiry datetime
            buffer_minutes: Refresh buffer before actual expiry (increased to 10 min)
            
        Returns:
            True if token is valid
            
        Note: Increased buffer from 5 to 10 minutes to further reduce 401s
              due to clock skew or server-side timing differences.
        """
        if not expiry:
            return False
        
        now = datetime.now(timezone.utc)
        buffer = timedelta(minutes=buffer_minutes)
        
        return now < (expiry - buffer)
    
    async def login(self) -> Dict[str, Any]:
        """
        Perform login and obtain new tokens with retry logic.
        
        Returns:
            Login response data
        """
        max_retries = 3
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 2 ** attempt  # Exponential backoff: 2s, 4s
                    logger.info(f"🔄 Login retry {attempt + 1}/{max_retries} after {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.info(f"Logging in as {self.username}...")
                
                # Increased timeout for login endpoint
                async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                    payload = {
                        "login": self.username,
                        "password": self.password
                    }
                    
                    response = await client.post(self.login_url, json=payload)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Extract tokens
                    self._access_token = data.get("token") or data.get("access_token")
                    self._refresh_token = data.get("refresh_token")
                    
                    if not self._access_token:
                        raise RuntimeError("Login response did not contain access token")
                    
                    # Parse expiry times
                    self._access_token_expiry = self._parse_jwt_expiry(self._access_token)
                    if not self._access_token_expiry:
                        # Default to 12 hours if parsing fails
                        self._access_token_expiry = datetime.now(timezone.utc) + timedelta(hours=12)
                    
                    if self._refresh_token:
                        self._refresh_token_expiry = self._parse_jwt_expiry(self._refresh_token)
                        if not self._refresh_token_expiry:
                            # Default to 6 hours if parsing fails
                            self._refresh_token_expiry = datetime.now(timezone.utc) + timedelta(hours=6)
                    
                    # Save to file
                    self._save_tokens_to_file()
                    
                    if attempt > 0:
                        logger.info(f"✅ Login successful on attempt {attempt + 1}! Access token expires at {self._access_token_expiry}")
                    else:
                        logger.info(f"✅ Login successful! Access token expires at {self._access_token_expiry}")
                    
                    return data
                    
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ConnectTimeout) as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    logger.warning(f"⏱️ Login connection issue on attempt {attempt + 1}: {type(exc).__name__}")
                    continue
                else:
                    logger.error(f"❌ Login failed after {max_retries} attempts")
                    logger.error(f"❌ Last error: {type(exc).__name__}: {str(exc)}")
                    raise RuntimeError(f"Login failed after {max_retries} attempts: {type(exc).__name__}") from exc
            except httpx.HTTPStatusError as exc:
                logger.error(f"❌ Login failed with status {exc.response.status_code}: {exc.response.text[:200]}")
                raise  # Don't retry on auth errors
            except Exception as exc:
                last_exception = exc
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Login error on attempt {attempt + 1}: {type(exc).__name__}: {str(exc)[:100]}")
                    continue
                else:
                    logger.error(f"❌ Login error after {max_retries} attempts: {type(exc).__name__}: {str(exc)}")
                    raise
        
        # Should never reach here, but just in case
        if last_exception:
            raise last_exception
        raise RuntimeError("Login failed for unknown reason")
    
    async def refresh_access_token(self, refresh_url: str) -> Dict[str, Any]:
        """
        Refresh access token using refresh token.
        
        Args:
            refresh_url: Refresh token endpoint
            
        Returns:
            Refresh response data
        """
        try:
            if not self._refresh_token:
                logger.warning("No refresh token available, performing full login")
                return await self.login()
            
            logger.info("Refreshing access token...")
            
            async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
                payload = {
                    "refresh_token": self._refresh_token
                }
                
                response = await client.post(refresh_url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                
                # Update access token
                self._access_token = data.get("token") or data.get("access_token")
                
                if not self._access_token:
                    raise RuntimeError("Refresh response did not contain access token")
                
                # Parse new expiry
                self._access_token_expiry = self._parse_jwt_expiry(self._access_token)
                if not self._access_token_expiry:
                    self._access_token_expiry = datetime.now(timezone.utc) + timedelta(hours=12)
                
                # Save to file
                self._save_tokens_to_file()
                
                logger.info(f"✅ Access token refreshed! Expires at {self._access_token_expiry}")
                
                return data
                
        except httpx.HTTPStatusError as exc:
            logger.error(f"Token refresh failed: {exc.response.status_code}")
            # If refresh fails, try full login
            logger.info("Refresh failed, attempting full login...")
            return await self.login()
        except Exception as exc:
            logger.error(f"Token refresh error: {exc}")
            # Fallback to full login
            return await self.login()
    
    async def validate_and_refresh_if_needed(self) -> None:
        """
        Validate token on startup (token is unlimited/long-lived).
        Call this during application startup.
        """
        if not self._access_token:
            logger.info("No token found, performing login...")
            await self.login()
            return
        
        # Token is unlimited/long-lived - just log status
        logger.info(f"✅ Using pre-configured unlimited token")
        
        # Optional: Still parse expiry for monitoring but don't enforce
        if self._access_token_expiry:
            logger.debug(f"Token expiry timestamp: {self._access_token_expiry}")
    
    async def get_valid_access_token(self, refresh_url: Optional[str] = None) -> str:
        """
        Get a valid access token, refreshing if necessary.
        
        Args:
            refresh_url: Optional refresh endpoint URL
            
        Returns:
            Valid access token
        """
        async with self._lock:
            # Check if current token is valid
            if self._is_token_valid(self._access_token_expiry):
                return self._access_token
            
            # Token expired or about to expire
            logger.info("Access token expired or expiring soon")
            
            # If using unlimited/pre-existing token without login credentials, just return it
            if self._access_token and not all([self.login_url, self.username, self.password]):
                logger.warning("⚠️ Token expired but no login credentials available (using unlimited token)")
                return self._access_token
            
            # Try refresh if we have a refresh token and URL
            if refresh_url and self._is_token_valid(self._refresh_token_expiry):
                try:
                    await self.refresh_access_token(refresh_url)
                    return self._access_token
                except Exception as exc:
                    logger.warning(f"Refresh failed: {exc}, falling back to login")
            
            # Perform full login
            await self.login()
            return self._access_token
    
    async def get_valid_token(self, refresh_url: Optional[str] = None) -> str:
        """
        Alias for get_valid_access_token for backward compatibility.
        
        Args:
            refresh_url: Optional refresh endpoint URL
            
        Returns:
            Valid access token
        """
        return await self.get_valid_access_token(refresh_url)
    
    async def get_auth_headers(self, refresh_url: Optional[str] = None) -> Dict[str, str]:
        """
        Get authorization headers with valid token.
        
        Args:
            refresh_url: Optional refresh endpoint URL
            
        Returns:
            Headers dictionary with Authorization
        """
        token = await self.get_valid_access_token(refresh_url)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def get_token_status(self) -> Dict[str, Any]:
        """
        Get current token status.
        
        Returns:
            Dictionary with token information
        """
        now = datetime.now(timezone.utc)
        
        return {
            "has_access_token": bool(self._access_token),
            "has_refresh_token": bool(self._refresh_token),
            "access_token_valid": self._is_token_valid(self._access_token_expiry),
            "refresh_token_valid": self._is_token_valid(self._refresh_token_expiry),
            "access_token_expiry": self._access_token_expiry.isoformat() if self._access_token_expiry else None,
            "refresh_token_expiry": self._refresh_token_expiry.isoformat() if self._refresh_token_expiry else None,
            "current_time": now.isoformat()
        }
    
    def should_refresh_proactively(self, buffer_minutes: int = 5) -> bool:
        """
        Check if token should be refreshed proactively (before expiry).
        
        Args:
            buffer_minutes: Refresh when token expires in less than this many minutes
            
        Returns:
            True if token should be refreshed now
        """
        if not self._access_token_expiry:
            return False
        
        now = datetime.now(timezone.utc)
        time_until_expiry = self._access_token_expiry - now
        
        # Refresh if expiring in less than buffer_minutes
        return time_until_expiry < timedelta(minutes=buffer_minutes)
    
    async def proactive_refresh_loop(self, refresh_url: str, check_interval_seconds: int = 60):
        """
        Background task that proactively refreshes tokens before expiry.
        
        Args:
            refresh_url: Refresh token endpoint
            check_interval_seconds: How often to check token status (default: 60s)
            
        Note: This should be run as a background task in the application
        """
        logger.info(f"🔄 Proactive token refresh loop started (checking every {check_interval_seconds}s)")
        
        while True:
            try:
                await asyncio.sleep(check_interval_seconds)
                
                # Check if we should refresh
                if self.should_refresh_proactively(buffer_minutes=5):
                    logger.info("⏰ Token expiring soon - proactively refreshing...")
                    
                    try:
                        if self._is_token_valid(self._refresh_token_expiry):
                            await self.refresh_access_token(refresh_url)
                            logger.info("✅ Proactive token refresh successful")
                        else:
                            logger.info("Refresh token expired - performing full login")
                            await self.login()
                    except Exception as refresh_exc:
                        logger.error(f"❌ Proactive refresh failed: {refresh_exc}")
                        # Don't crash the loop - will retry on next iteration
                        
            except asyncio.CancelledError:
                logger.info("Proactive refresh loop cancelled")
                break
            except Exception as exc:
                logger.error(f"Error in proactive refresh loop: {exc}")
                # Continue loop despite errors
                await asyncio.sleep(check_interval_seconds)
    
    async def logout(self, logout_url: str) -> None:
        """
        Logout and clear tokens.
        
        Args:
            logout_url: Logout endpoint URL
        """
        try:
            if self._access_token:
                async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
                    headers = {"Authorization": f"Bearer {self._access_token}"}
                    await client.post(logout_url, headers=headers)
        except Exception as exc:
            logger.warning(f"Logout request failed: {exc}")
        finally:
            # Clear tokens regardless of logout success
            self._access_token = None
            self._refresh_token = None
            self._access_token_expiry = None
            self._refresh_token_expiry = None
            
            # Delete token file
            if self.token_file.exists():
                self.token_file.unlink()
            
            logger.info("Logged out and tokens cleared")


# Global token manager instance
_token_manager: Optional[TokenManager] = None


def get_token_manager(
    login_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    token_file: str = "tokens.json",
    pre_existing_token: Optional[str] = None
) -> TokenManager:
    """
    Get or create global token manager instance.
    
    Args:
        login_url: API login endpoint (optional if using pre_existing_token)
        username: Login username (optional if using pre_existing_token)
        password: Login password (optional if using pre_existing_token)
        token_file: Path to token storage file
        pre_existing_token: Optional pre-existing unlimited token (no refresh needed)
        
    Returns:
        TokenManager instance
    """
    global _token_manager
    
    if _token_manager is None:
        _token_manager = TokenManager(login_url, username, password, token_file, pre_existing_token)
    
    return _token_manager
