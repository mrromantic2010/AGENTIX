"""
Validation utilities for Agentix framework.
"""

import re
import html
import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
from pathlib import Path

from .exceptions import ValidationError


def validate_input(data: Any, schema: Dict[str, Any]) -> bool:
    """
    Validate input data against a schema.
    
    Args:
        data: Data to validate
        schema: Validation schema
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    
    if 'type' in schema:
        if not _validate_type(data, schema['type']):
            raise ValidationError(f"Expected type {schema['type']}, got {type(data).__name__}")
    
    if 'required' in schema and schema['required']:
        if data is None or data == "":
            raise ValidationError("Required field is missing or empty")
    
    if 'min_length' in schema and isinstance(data, str):
        if len(data) < schema['min_length']:
            raise ValidationError(f"String too short: {len(data)} < {schema['min_length']}")
    
    if 'max_length' in schema and isinstance(data, str):
        if len(data) > schema['max_length']:
            raise ValidationError(f"String too long: {len(data)} > {schema['max_length']}")
    
    if 'pattern' in schema and isinstance(data, str):
        if not re.match(schema['pattern'], data):
            raise ValidationError(f"String does not match pattern: {schema['pattern']}")
    
    if 'min_value' in schema and isinstance(data, (int, float)):
        if data < schema['min_value']:
            raise ValidationError(f"Value too small: {data} < {schema['min_value']}")
    
    if 'max_value' in schema and isinstance(data, (int, float)):
        if data > schema['max_value']:
            raise ValidationError(f"Value too large: {data} > {schema['max_value']}")
    
    if 'allowed_values' in schema:
        if data not in schema['allowed_values']:
            raise ValidationError(f"Value not allowed: {data}")
    
    if 'custom_validator' in schema:
        validator_func = schema['custom_validator']
        if not validator_func(data):
            raise ValidationError("Custom validation failed")
    
    return True


def validate_output(data: Any, schema: Dict[str, Any]) -> bool:
    """
    Validate output data against a schema.
    
    Args:
        data: Data to validate
        schema: Validation schema
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    return validate_input(data, schema)


def _validate_type(data: Any, expected_type: str) -> bool:
    """Validate data type."""
    type_mapping = {
        'string': str,
        'int': int,
        'float': float,
        'bool': bool,
        'list': list,
        'dict': dict,
        'none': type(None)
    }
    
    if expected_type in type_mapping:
        return isinstance(data, type_mapping[expected_type])
    
    return True


def sanitize_data(data: Any, sanitization_rules: Optional[Dict[str, Any]] = None) -> Any:
    """
    Sanitize data according to rules.
    
    Args:
        data: Data to sanitize
        sanitization_rules: Rules for sanitization
        
    Returns:
        Sanitized data
    """
    
    if sanitization_rules is None:
        sanitization_rules = get_default_sanitization_rules()
    
    if isinstance(data, str):
        return _sanitize_string(data, sanitization_rules)
    elif isinstance(data, dict):
        return _sanitize_dict(data, sanitization_rules)
    elif isinstance(data, list):
        return _sanitize_list(data, sanitization_rules)
    else:
        return data


def _sanitize_string(text: str, rules: Dict[str, Any]) -> str:
    """Sanitize string data."""
    
    # HTML escape
    if rules.get('html_escape', True):
        text = html.escape(text)
    
    # Remove control characters
    if rules.get('remove_control_chars', True):
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\t\n\r')
    
    # Limit length
    max_length = rules.get('max_string_length', 10000)
    if len(text) > max_length:
        text = text[:max_length]
    
    # Remove dangerous patterns
    if rules.get('remove_dangerous_patterns', True):
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\('
        ]
        
        for pattern in dangerous_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    return text


def _sanitize_dict(data: Dict[str, Any], rules: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize dictionary data."""
    
    sanitized = {}
    max_keys = rules.get('max_dict_keys', 1000)
    
    for i, (key, value) in enumerate(data.items()):
        if i >= max_keys:
            break
        
        # Sanitize key
        if isinstance(key, str):
            key = _sanitize_string(key, rules)
        
        # Sanitize value
        sanitized[key] = sanitize_data(value, rules)
    
    return sanitized


def _sanitize_list(data: List[Any], rules: Dict[str, Any]) -> List[Any]:
    """Sanitize list data."""
    
    max_items = rules.get('max_list_items', 1000)
    sanitized = []
    
    for i, item in enumerate(data):
        if i >= max_items:
            break
        
        sanitized.append(sanitize_data(item, rules))
    
    return sanitized


def get_default_sanitization_rules() -> Dict[str, Any]:
    """Get default sanitization rules."""
    return {
        'html_escape': True,
        'remove_control_chars': True,
        'remove_dangerous_patterns': True,
        'max_string_length': 10000,
        'max_dict_keys': 1000,
        'max_list_items': 1000
    }


def validate_url(url: str, allowed_schemes: Optional[List[str]] = None,
                allowed_domains: Optional[List[str]] = None,
                blocked_domains: Optional[List[str]] = None) -> bool:
    """
    Validate URL for security.
    
    Args:
        url: URL to validate
        allowed_schemes: Allowed URL schemes
        allowed_domains: Allowed domains (if specified, only these are allowed)
        blocked_domains: Blocked domains
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    
    try:
        parsed = urlparse(url)
    except Exception:
        raise ValidationError("Invalid URL format")
    
    # Check scheme
    if allowed_schemes is None:
        allowed_schemes = ['http', 'https']
    
    if parsed.scheme not in allowed_schemes:
        raise ValidationError(f"URL scheme not allowed: {parsed.scheme}")
    
    # Check domain
    domain = parsed.netloc.lower()
    
    if blocked_domains:
        for blocked_domain in blocked_domains:
            if blocked_domain.lower() in domain:
                raise ValidationError(f"Blocked domain: {domain}")
    
    if allowed_domains:
        allowed = False
        for allowed_domain in allowed_domains:
            if allowed_domain.lower() in domain:
                allowed = True
                break
        
        if not allowed:
            raise ValidationError(f"Domain not in allowed list: {domain}")
    
    return True


def validate_file_path(file_path: str, allowed_directories: Optional[List[str]] = None,
                      blocked_directories: Optional[List[str]] = None,
                      allowed_extensions: Optional[List[str]] = None) -> bool:
    """
    Validate file path for security.
    
    Args:
        file_path: File path to validate
        allowed_directories: Allowed directories
        blocked_directories: Blocked directories
        allowed_extensions: Allowed file extensions
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    
    try:
        path = Path(file_path).resolve()
        path_str = str(path)
    except Exception:
        raise ValidationError("Invalid file path")
    
    # Check blocked directories
    if blocked_directories:
        for blocked_dir in blocked_directories:
            if path_str.startswith(blocked_dir):
                raise ValidationError(f"Access to blocked directory: {blocked_dir}")
    
    # Check allowed directories
    if allowed_directories:
        allowed = False
        for allowed_dir in allowed_directories:
            if path_str.startswith(allowed_dir):
                allowed = True
                break
        
        if not allowed:
            raise ValidationError(f"Path not in allowed directories: {path_str}")
    
    # Check file extension
    if allowed_extensions and path.suffix:
        if path.suffix.lower() not in allowed_extensions:
            raise ValidationError(f"File extension not allowed: {path.suffix}")
    
    return True


def validate_json(data: str, max_size: int = 1024 * 1024) -> bool:
    """
    Validate JSON data.
    
    Args:
        data: JSON string to validate
        max_size: Maximum size in bytes
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    
    if len(data.encode('utf-8')) > max_size:
        raise ValidationError(f"JSON data too large: {len(data)} > {max_size}")
    
    try:
        json.loads(data)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON: {str(e)}")
    
    return True


def validate_email(email: str) -> bool:
    """
    Validate email address.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, email):
        raise ValidationError("Invalid email format")
    
    return True


def validate_api_key(api_key: str, min_length: int = 10, max_length: int = 200) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        min_length: Minimum length
        max_length: Maximum length
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    
    if len(api_key) < min_length:
        raise ValidationError(f"API key too short: {len(api_key)} < {min_length}")
    
    if len(api_key) > max_length:
        raise ValidationError(f"API key too long: {len(api_key)} > {max_length}")
    
    # Check for basic format (alphanumeric and common special chars)
    if not re.match(r'^[a-zA-Z0-9._-]+$', api_key):
        raise ValidationError("API key contains invalid characters")
    
    return True


class InputValidator:
    """Reusable input validator with configurable rules."""
    
    def __init__(self, rules: Optional[Dict[str, Any]] = None):
        self.rules = rules or {}
    
    def validate(self, data: Any, field_name: str = "input") -> Any:
        """Validate and sanitize input data."""
        
        try:
            # Apply validation rules
            if field_name in self.rules:
                validate_input(data, self.rules[field_name])
            
            # Apply sanitization
            sanitized_data = sanitize_data(data)
            
            return sanitized_data
            
        except ValidationError as e:
            raise ValidationError(f"Validation failed for {field_name}: {str(e)}")
    
    def add_rule(self, field_name: str, rule: Dict[str, Any]):
        """Add validation rule for a field."""
        self.rules[field_name] = rule
    
    def remove_rule(self, field_name: str):
        """Remove validation rule for a field."""
        if field_name in self.rules:
            del self.rules[field_name]
