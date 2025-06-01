"""
Input validation guardrails for Agentix agents.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pydantic import BaseModel, Field

from ..utils.exceptions import ValidationError
from ..utils.validation import sanitize_data, validate_url, validate_file_path


class InputValidationConfig(BaseModel):
    """Configuration for input validation."""
    
    # Basic validation
    max_input_length: int = 10000
    min_input_length: int = 0
    allowed_input_types: List[str] = Field(default_factory=lambda: ["str", "dict", "list", "int", "float", "bool"])
    
    # Content filtering
    block_profanity: bool = True
    block_personal_info: bool = True
    block_malicious_patterns: bool = True
    
    # Security validation
    validate_urls: bool = True
    validate_file_paths: bool = True
    allowed_url_schemes: List[str] = Field(default_factory=lambda: ["http", "https"])
    blocked_domains: List[str] = Field(default_factory=list)
    allowed_file_extensions: List[str] = Field(default_factory=lambda: [".txt", ".json", ".csv", ".md"])
    
    # Custom validation
    custom_validators: List[str] = Field(default_factory=list)
    validation_rules: Dict[str, Any] = Field(default_factory=dict)
    
    # Sanitization
    enable_sanitization: bool = True
    sanitization_rules: Dict[str, Any] = Field(default_factory=dict)


class InputValidator:
    """
    Input validation guardrail for agent inputs.
    
    This validator provides:
    - Basic input validation (type, length, format)
    - Content filtering (profanity, personal info, malicious patterns)
    - Security validation (URLs, file paths)
    - Custom validation rules
    - Input sanitization
    """
    
    def __init__(self, config: InputValidationConfig):
        self.config = config
        self.logger = logging.getLogger("agentix.guardrails.input_validation")
        
        # Compile regex patterns for performance
        self._compile_patterns()
        
        # Load custom validators
        self.custom_validators: Dict[str, Callable] = {}
        self._load_custom_validators()
    
    def _compile_patterns(self):
        """Compile regex patterns for validation."""
        
        # Profanity patterns (basic examples)
        self.profanity_patterns = [
            re.compile(r'\b(damn|hell|crap)\b', re.IGNORECASE),
            # Add more patterns as needed
        ]
        
        # Personal information patterns
        self.personal_info_patterns = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # Credit card
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # Phone number
        ]
        
        # Malicious patterns
        self.malicious_patterns = [
            re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
            re.compile(r'javascript:', re.IGNORECASE),
            re.compile(r'vbscript:', re.IGNORECASE),
            re.compile(r'on\w+\s*=', re.IGNORECASE),
            re.compile(r'eval\s*\(', re.IGNORECASE),
            re.compile(r'exec\s*\(', re.IGNORECASE),
            re.compile(r'system\s*\(', re.IGNORECASE),
            re.compile(r'shell_exec\s*\(', re.IGNORECASE),
        ]
    
    def _load_custom_validators(self):
        """Load custom validation functions."""
        # This would load custom validators from configuration
        # For now, it's a placeholder
        pass
    
    def validate(self, input_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate input data.
        
        Args:
            input_data: Data to validate
            context: Additional context for validation
            
        Returns:
            Validation result with sanitized data
            
        Raises:
            ValidationError: If validation fails
        """
        
        validation_result = {
            'valid': True,
            'sanitized_data': input_data,
            'warnings': [],
            'errors': [],
            'metadata': {
                'validation_timestamp': datetime.now().isoformat(),
                'validator_version': '1.0.0'
            }
        }
        
        try:
            # Basic validation
            self._validate_basic(input_data, validation_result)
            
            # Content filtering
            if isinstance(input_data, str):
                self._validate_content(input_data, validation_result)
            
            # Security validation
            self._validate_security(input_data, validation_result)
            
            # Custom validation
            self._validate_custom(input_data, validation_result, context)
            
            # Sanitization
            if self.config.enable_sanitization and validation_result['valid']:
                validation_result['sanitized_data'] = self._sanitize_input(
                    validation_result['sanitized_data']
                )
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Input validation failed: {str(e)}")
        
        return validation_result
    
    def _validate_basic(self, input_data: Any, result: Dict[str, Any]):
        """Perform basic validation checks."""
        
        # Type validation
        input_type = type(input_data).__name__
        if input_type not in self.config.allowed_input_types:
            result['valid'] = False
            result['errors'].append(f"Input type '{input_type}' not allowed")
            return
        
        # Length validation for strings
        if isinstance(input_data, str):
            if len(input_data) < self.config.min_input_length:
                result['valid'] = False
                result['errors'].append(f"Input too short: {len(input_data)} < {self.config.min_input_length}")
                return
            
            if len(input_data) > self.config.max_input_length:
                result['valid'] = False
                result['errors'].append(f"Input too long: {len(input_data)} > {self.config.max_input_length}")
                return
        
        # Size validation for collections
        elif isinstance(input_data, (list, dict)):
            if len(input_data) > 1000:  # Reasonable default limit
                result['warnings'].append(f"Large collection size: {len(input_data)} items")
    
    def _validate_content(self, text: str, result: Dict[str, Any]):
        """Validate text content for inappropriate material."""
        
        # Profanity check
        if self.config.block_profanity:
            for pattern in self.profanity_patterns:
                if pattern.search(text):
                    result['warnings'].append("Potential profanity detected")
                    break
        
        # Personal information check
        if self.config.block_personal_info:
            for pattern in self.personal_info_patterns:
                if pattern.search(text):
                    result['valid'] = False
                    result['errors'].append("Personal information detected")
                    return
        
        # Malicious pattern check
        if self.config.block_malicious_patterns:
            for pattern in self.malicious_patterns:
                if pattern.search(text):
                    result['valid'] = False
                    result['errors'].append("Malicious pattern detected")
                    return
    
    def _validate_security(self, input_data: Any, result: Dict[str, Any]):
        """Perform security validation."""
        
        if isinstance(input_data, str):
            # URL validation
            if self.config.validate_urls and self._looks_like_url(input_data):
                try:
                    validate_url(
                        input_data,
                        allowed_schemes=self.config.allowed_url_schemes,
                        blocked_domains=self.config.blocked_domains
                    )
                except ValidationError as e:
                    result['valid'] = False
                    result['errors'].append(f"URL validation failed: {str(e)}")
                    return
            
            # File path validation
            if self.config.validate_file_paths and self._looks_like_file_path(input_data):
                try:
                    validate_file_path(
                        input_data,
                        allowed_extensions=self.config.allowed_file_extensions
                    )
                except ValidationError as e:
                    result['valid'] = False
                    result['errors'].append(f"File path validation failed: {str(e)}")
                    return
        
        elif isinstance(input_data, dict):
            # Recursively validate dictionary values
            for key, value in input_data.items():
                if isinstance(value, str):
                    self._validate_security(value, result)
                    if not result['valid']:
                        return
    
    def _validate_custom(self, input_data: Any, result: Dict[str, Any], 
                        context: Optional[Dict[str, Any]]):
        """Apply custom validation rules."""
        
        # Apply validation rules from configuration
        for rule_name, rule_config in self.config.validation_rules.items():
            try:
                if not self._apply_validation_rule(input_data, rule_config, context):
                    result['valid'] = False
                    result['errors'].append(f"Custom validation rule '{rule_name}' failed")
                    return
            except Exception as e:
                result['warnings'].append(f"Custom validation rule '{rule_name}' error: {str(e)}")
        
        # Apply custom validator functions
        for validator_name, validator_func in self.custom_validators.items():
            try:
                if not validator_func(input_data, context):
                    result['valid'] = False
                    result['errors'].append(f"Custom validator '{validator_name}' failed")
                    return
            except Exception as e:
                result['warnings'].append(f"Custom validator '{validator_name}' error: {str(e)}")
    
    def _apply_validation_rule(self, input_data: Any, rule_config: Dict[str, Any],
                              context: Optional[Dict[str, Any]]) -> bool:
        """Apply a single validation rule."""
        
        rule_type = rule_config.get('type', 'pattern')
        
        if rule_type == 'pattern' and isinstance(input_data, str):
            pattern = rule_config.get('pattern', '')
            return bool(re.match(pattern, input_data))
        
        elif rule_type == 'length' and isinstance(input_data, str):
            min_len = rule_config.get('min_length', 0)
            max_len = rule_config.get('max_length', float('inf'))
            return min_len <= len(input_data) <= max_len
        
        elif rule_type == 'whitelist':
            allowed_values = rule_config.get('allowed_values', [])
            return input_data in allowed_values
        
        elif rule_type == 'blacklist':
            blocked_values = rule_config.get('blocked_values', [])
            return input_data not in blocked_values
        
        return True
    
    def _sanitize_input(self, input_data: Any) -> Any:
        """Sanitize input data."""
        
        sanitization_rules = self.config.sanitization_rules
        if not sanitization_rules:
            # Use default sanitization rules
            sanitization_rules = {
                'html_escape': True,
                'remove_control_chars': True,
                'max_string_length': self.config.max_input_length
            }
        
        return sanitize_data(input_data, sanitization_rules)
    
    def _looks_like_url(self, text: str) -> bool:
        """Check if text looks like a URL."""
        return text.startswith(('http://', 'https://', 'ftp://', 'www.'))
    
    def _looks_like_file_path(self, text: str) -> bool:
        """Check if text looks like a file path."""
        return ('/' in text or '\\' in text) and ('.' in text)
    
    def add_custom_validator(self, name: str, validator_func: Callable):
        """Add a custom validation function."""
        self.custom_validators[name] = validator_func
    
    def remove_custom_validator(self, name: str):
        """Remove a custom validation function."""
        if name in self.custom_validators:
            del self.custom_validators[name]
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'config': self.config.dict(),
            'custom_validators': list(self.custom_validators.keys()),
            'validation_rules': list(self.config.validation_rules.keys())
        }
