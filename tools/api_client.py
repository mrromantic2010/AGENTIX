"""
API Client Tool for Agentix agents.

This tool provides HTTP API client capabilities including:
- REST API calls (GET, POST, PUT, DELETE)
- Authentication handling
- Request/response validation
- Rate limiting and retry logic
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import aiohttp
import json
from pydantic import BaseModel, Field, validator

from .base import BaseTool, ToolConfig, ToolResult, ToolStatus


class APIClientConfig(ToolConfig):
    """Configuration for API client tool."""
    
    # Base settings
    base_url: Optional[str] = None
    default_headers: Dict[str, str] = Field(default_factory=dict)
    
    # Authentication
    auth_type: str = "none"  # none, bearer, basic, api_key
    auth_token: Optional[str] = None
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    api_key_header: str = "X-API-Key"
    
    # Request settings
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Security
    allowed_domains: List[str] = Field(default_factory=list)
    blocked_domains: List[str] = Field(default_factory=list)
    verify_ssl: bool = True
    
    @validator('auth_type')
    def validate_auth_type(cls, v):
        allowed_types = ["none", "bearer", "basic", "api_key"]
        if v not in allowed_types:
            raise ValueError(f"auth_type must be one of {allowed_types}")
        return v


class APIClientTool(BaseTool):
    """
    API Client Tool for HTTP API interactions.
    
    This tool provides:
    - RESTful API calls with multiple HTTP methods
    - Authentication support (Bearer, Basic, API Key)
    - Request/response validation
    - Rate limiting and retry logic
    - Security validation
    """
    
    def __init__(self, config: APIClientConfig):
        super().__init__(config)
        self.api_config = config
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self.request_timestamps: List[datetime] = []
        
        # Setup default headers
        self.default_headers = {
            'User-Agent': f'Agentix-APIClient/{self.version}',
            'Content-Type': 'application/json',
            **config.default_headers
        }
        
        # Setup authentication
        self._setup_authentication()
    
    def _setup_authentication(self):
        """Setup authentication headers."""
        if self.api_config.auth_type == "bearer" and self.api_config.auth_token:
            self.default_headers['Authorization'] = f'Bearer {self.api_config.auth_token}'
        
        elif self.api_config.auth_type == "api_key" and self.api_config.auth_token:
            self.default_headers[self.api_config.api_key_header] = self.api_config.auth_token
        
        # Basic auth is handled per-request
    
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute API request with the given parameters."""
        
        method = parameters.get('method', 'GET').upper()
        url = parameters.get('url', '')
        
        if not url:
            return ToolResult(
                status=ToolStatus.FAILURE,
                error="URL is required",
                tool_name=self.name,
                tool_version=self.version
            )
        
        try:
            # Validate URL security
            if not self._validate_url_security(url):
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error="URL failed security validation",
                    tool_name=self.name,
                    tool_version=self.version
                )
            
            # Check rate limiting
            if not self._check_rate_limit():
                return ToolResult(
                    status=ToolStatus.FAILURE,
                    error="Rate limit exceeded",
                    tool_name=self.name,
                    tool_version=self.version
                )
            
            # Initialize session if needed
            if not self.session:
                connector = aiohttp.TCPConnector(verify_ssl=self.api_config.verify_ssl)
                self.session = aiohttp.ClientSession(connector=connector)
            
            # Make API request
            response_data = await self._make_request(method, url, parameters)
            
            return ToolResult(
                status=ToolStatus.SUCCESS,
                data=response_data,
                tool_name=self.name,
                tool_version=self.version
            )
            
        except Exception as e:
            self.logger.error(f"API request failed: {str(e)}")
            return ToolResult(
                status=ToolStatus.FAILURE,
                error=f"API request failed: {str(e)}",
                tool_name=self.name,
                tool_version=self.version
            )
    
    async def _make_request(self, method: str, url: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Make the actual HTTP request."""
        
        # Prepare request parameters
        headers = {**self.default_headers}
        headers.update(parameters.get('headers', {}))
        
        # Handle different content types
        data = None
        json_data = None
        
        if 'data' in parameters:
            if isinstance(parameters['data'], dict):
                json_data = parameters['data']
            else:
                data = parameters['data']
                headers['Content-Type'] = 'text/plain'
        
        # Query parameters
        params = parameters.get('params', {})
        
        # Basic authentication
        auth = None
        if self.api_config.auth_type == "basic":
            if self.api_config.auth_username and self.api_config.auth_password:
                auth = aiohttp.BasicAuth(
                    self.api_config.auth_username,
                    self.api_config.auth_password
                )
        
        # Build full URL
        if self.api_config.base_url and not url.startswith(('http://', 'https://')):
            full_url = f"{self.api_config.base_url.rstrip('/')}/{url.lstrip('/')}"
        else:
            full_url = url
        
        # Make request with retries
        last_error = None
        for attempt in range(self.api_config.max_retries):
            try:
                async with self.session.request(
                    method=method,
                    url=full_url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    data=data,
                    auth=auth,
                    timeout=aiohttp.ClientTimeout(total=self.api_config.timeout_seconds)
                ) as response:
                    
                    # Record request for rate limiting
                    self.request_timestamps.append(datetime.now())
                    
                    # Parse response
                    response_text = await response.text()
                    
                    try:
                        response_json = json.loads(response_text) if response_text else {}
                    except json.JSONDecodeError:
                        response_json = {'raw_response': response_text}
                    
                    return {
                        'status_code': response.status,
                        'headers': dict(response.headers),
                        'data': response_json,
                        'url': str(response.url),
                        'method': method,
                        'success': 200 <= response.status < 300,
                        'response_time': 0.0,  # Would be calculated in production
                        'attempt': attempt + 1
                    }
                    
            except asyncio.TimeoutError:
                last_error = f"Request timed out after {self.api_config.timeout_seconds} seconds"
                self.logger.warning(f"API request timeout on attempt {attempt + 1}")
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"API request failed on attempt {attempt + 1}: {str(e)}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.api_config.max_retries - 1:
                await asyncio.sleep(self.api_config.retry_delay * (attempt + 1))
        
        # All attempts failed
        raise Exception(f"Request failed after {self.api_config.max_retries} attempts: {last_error}")
    
    def _validate_url_security(self, url: str) -> bool:
        """Validate URL for security compliance."""
        try:
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check blocked domains
            for blocked_domain in self.api_config.blocked_domains:
                if blocked_domain.lower() in domain:
                    self.logger.warning(f"Blocked domain access attempted: {domain}")
                    return False
            
            # Check allowed domains (if specified)
            if self.api_config.allowed_domains:
                allowed = False
                for allowed_domain in self.api_config.allowed_domains:
                    if allowed_domain.lower() in domain:
                        allowed = True
                        break
                
                if not allowed:
                    self.logger.warning(f"Domain not in allowed list: {domain}")
                    return False
            
            # Basic URL validation
            if parsed.scheme not in ['http', 'https']:
                self.logger.warning(f"Invalid URL scheme: {parsed.scheme}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"URL validation error: {str(e)}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """Check if request is within rate limits."""
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=self.api_config.rate_limit_window)
        
        # Remove old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if ts > window_start
        ]
        
        # Check if we're within limits
        if len(self.request_timestamps) >= self.api_config.rate_limit_requests:
            self.logger.warning("Rate limit exceeded")
            return False
        
        return True
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate API request parameters."""
        # Check required parameters
        if 'url' not in parameters:
            return False
        
        # Validate HTTP method
        method = parameters.get('method', 'GET').upper()
        valid_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'HEAD', 'OPTIONS']
        if method not in valid_methods:
            return False
        
        # Validate headers if provided
        headers = parameters.get('headers')
        if headers and not isinstance(headers, dict):
            return False
        
        # Validate params if provided
        params = parameters.get('params')
        if params and not isinstance(params, dict):
            return False
        
        return True
    
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("API client session closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    def get_rate_limit_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        current_time = datetime.now()
        window_start = current_time - timedelta(seconds=self.api_config.rate_limit_window)
        
        # Count requests in current window
        recent_requests = [
            ts for ts in self.request_timestamps 
            if ts > window_start
        ]
        
        return {
            'requests_in_window': len(recent_requests),
            'max_requests': self.api_config.rate_limit_requests,
            'window_seconds': self.api_config.rate_limit_window,
            'remaining_requests': max(0, self.api_config.rate_limit_requests - len(recent_requests)),
            'window_reset_time': (window_start + timedelta(seconds=self.api_config.rate_limit_window)).isoformat()
        }
