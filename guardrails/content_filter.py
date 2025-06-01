"""
Content filtering guardrails for Agentix agents.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pydantic import BaseModel, Field

from ..utils.exceptions import ValidationError


class ContentFilterConfig(BaseModel):
    """Configuration for content filter."""
    
    # Redaction settings
    redact_personal_info: bool = True
    redact_financial_info: bool = True
    redact_medical_info: bool = True
    redact_contact_info: bool = True
    
    # Replacement settings
    replacement_text: str = "[REDACTED]"
    preserve_format: bool = True
    
    # Custom patterns
    custom_patterns: Dict[str, str] = Field(default_factory=dict)
    
    # Filtering options
    case_sensitive: bool = False
    whole_words_only: bool = False


class ContentFilter:
    """
    Content filtering guardrail for sensitive information redaction.
    
    This filter provides:
    - Personal information redaction
    - Financial information protection
    - Medical information filtering
    - Custom pattern filtering
    - Format-preserving redaction
    """
    
    def __init__(self, config: ContentFilterConfig):
        self.config = config
        self.logger = logging.getLogger("agentix.guardrails.content_filter")
        
        # Compile filtering patterns
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for filtering."""
        
        flags = re.IGNORECASE if not self.config.case_sensitive else 0
        
        # Personal information patterns
        self.personal_patterns = {
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b', flags),
            'ssn_no_dash': re.compile(r'\b\d{9}\b', flags),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', flags),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', flags),
            'drivers_license': re.compile(r'\b[A-Z]{1,2}\d{6,8}\b', flags),
        }
        
        # Financial information patterns
        self.financial_patterns = {
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', flags),
            'bank_account': re.compile(r'\b\d{8,17}\b', flags),
            'routing_number': re.compile(r'\b\d{9}\b', flags),
            'iban': re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b', flags),
        }
        
        # Medical information patterns
        self.medical_patterns = {
            'medical_record': re.compile(r'\bMRN[-:\s]*\d+\b', flags),
            'patient_id': re.compile(r'\bPatient[-\s]ID[-:\s]*\d+\b', flags),
            'insurance_id': re.compile(r'\bInsurance[-\s]ID[-:\s]*[A-Z0-9]+\b', flags),
        }
        
        # Contact information patterns
        self.contact_patterns = {
            'address': re.compile(r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b', flags),
            'zip_code': re.compile(r'\b\d{5}(?:-\d{4})?\b', flags),
        }
        
        # Custom patterns from configuration
        self.custom_patterns = {}
        for name, pattern in self.config.custom_patterns.items():
            try:
                self.custom_patterns[name] = re.compile(pattern, flags)
            except re.error as e:
                self.logger.warning(f"Invalid custom pattern '{name}': {str(e)}")
    
    def filter_content(self, content: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Filter content and redact sensitive information.
        
        Args:
            content: Content to filter
            context: Additional context for filtering
            
        Returns:
            Filtering result with redacted content
        """
        
        filter_result = {
            'filtered_content': content,
            'redactions': [],
            'patterns_found': {},
            'original_length': len(content),
            'metadata': {
                'filter_timestamp': datetime.now().isoformat(),
                'filter_version': '1.0.0'
            }
        }
        
        try:
            # Apply personal information filtering
            if self.config.redact_personal_info:
                filter_result = self._apply_pattern_filtering(
                    filter_result, self.personal_patterns, 'personal'
                )
            
            # Apply financial information filtering
            if self.config.redact_financial_info:
                filter_result = self._apply_pattern_filtering(
                    filter_result, self.financial_patterns, 'financial'
                )
            
            # Apply medical information filtering
            if self.config.redact_medical_info:
                filter_result = self._apply_pattern_filtering(
                    filter_result, self.medical_patterns, 'medical'
                )
            
            # Apply contact information filtering
            if self.config.redact_contact_info:
                filter_result = self._apply_pattern_filtering(
                    filter_result, self.contact_patterns, 'contact'
                )
            
            # Apply custom patterns
            if self.custom_patterns:
                filter_result = self._apply_pattern_filtering(
                    filter_result, self.custom_patterns, 'custom'
                )
            
            # Update final length
            filter_result['filtered_length'] = len(filter_result['filtered_content'])
            filter_result['reduction_percentage'] = (
                (filter_result['original_length'] - filter_result['filtered_length']) / 
                filter_result['original_length'] * 100
            ) if filter_result['original_length'] > 0 else 0
        
        except Exception as e:
            self.logger.error(f"Content filtering failed: {str(e)}")
            filter_result['error'] = str(e)
        
        return filter_result
    
    def _apply_pattern_filtering(self, filter_result: Dict[str, Any], 
                               patterns: Dict[str, re.Pattern], 
                               category: str) -> Dict[str, Any]:
        """Apply a set of filtering patterns."""
        
        content = filter_result['filtered_content']
        
        for pattern_name, pattern in patterns.items():
            matches = list(pattern.finditer(content))
            
            if matches:
                # Record pattern matches
                full_pattern_name = f"{category}_{pattern_name}"
                filter_result['patterns_found'][full_pattern_name] = len(matches)
                
                # Apply redaction
                for match in reversed(matches):  # Reverse to maintain positions
                    start, end = match.span()
                    original_text = match.group()
                    
                    # Generate replacement text
                    replacement = self._generate_replacement(original_text, pattern_name)
                    
                    # Record redaction
                    filter_result['redactions'].append({
                        'pattern': full_pattern_name,
                        'position': start,
                        'original_length': len(original_text),
                        'replacement_length': len(replacement),
                        'original_preview': original_text[:10] + '...' if len(original_text) > 10 else original_text
                    })
                    
                    # Apply redaction
                    content = content[:start] + replacement + content[end:]
        
        filter_result['filtered_content'] = content
        return filter_result
    
    def _generate_replacement(self, original_text: str, pattern_name: str) -> str:
        """Generate appropriate replacement text."""
        
        if not self.config.preserve_format:
            return self.config.replacement_text
        
        # Format-preserving replacements
        if pattern_name in ['ssn', 'ssn_no_dash']:
            return 'XXX-XX-XXXX' if '-' in original_text else 'XXXXXXXXX'
        
        elif pattern_name == 'phone':
            if '-' in original_text:
                return 'XXX-XXX-XXXX'
            elif '.' in original_text:
                return 'XXX.XXX.XXXX'
            else:
                return 'XXXXXXXXXX'
        
        elif pattern_name == 'email':
            # Preserve domain structure
            parts = original_text.split('@')
            if len(parts) == 2:
                return f"{'X' * len(parts[0])}@{parts[1]}"
            return self.config.replacement_text
        
        elif pattern_name == 'credit_card':
            # Preserve card format
            if '-' in original_text:
                return 'XXXX-XXXX-XXXX-XXXX'
            elif ' ' in original_text:
                return 'XXXX XXXX XXXX XXXX'
            else:
                return 'XXXXXXXXXXXXXXXX'
        
        elif pattern_name == 'zip_code':
            if '-' in original_text:
                return 'XXXXX-XXXX'
            else:
                return 'XXXXX'
        
        else:
            # Default: preserve length with X's
            return 'X' * len(original_text)
    
    def add_custom_pattern(self, name: str, pattern: str, description: str = ""):
        """Add a custom filtering pattern."""
        
        try:
            flags = re.IGNORECASE if not self.config.case_sensitive else 0
            compiled_pattern = re.compile(pattern, flags)
            self.custom_patterns[name] = compiled_pattern
            self.config.custom_patterns[name] = pattern
            
            self.logger.info(f"Added custom pattern '{name}': {description}")
        
        except re.error as e:
            raise ValidationError(f"Invalid regex pattern '{name}': {str(e)}")
    
    def remove_custom_pattern(self, name: str):
        """Remove a custom filtering pattern."""
        
        if name in self.custom_patterns:
            del self.custom_patterns[name]
            
        if name in self.config.custom_patterns:
            del self.config.custom_patterns[name]
        
        self.logger.info(f"Removed custom pattern '{name}'")
    
    def test_pattern(self, pattern: str, test_text: str) -> Dict[str, Any]:
        """Test a regex pattern against sample text."""
        
        try:
            flags = re.IGNORECASE if not self.config.case_sensitive else 0
            compiled_pattern = re.compile(pattern, flags)
            matches = list(compiled_pattern.finditer(test_text))
            
            return {
                'valid_pattern': True,
                'matches_found': len(matches),
                'matches': [
                    {
                        'text': match.group(),
                        'start': match.start(),
                        'end': match.end()
                    }
                    for match in matches
                ]
            }
        
        except re.error as e:
            return {
                'valid_pattern': False,
                'error': str(e)
            }
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get content filter statistics."""
        
        return {
            'config': self.config.dict(),
            'pattern_counts': {
                'personal': len(self.personal_patterns),
                'financial': len(self.financial_patterns),
                'medical': len(self.medical_patterns),
                'contact': len(self.contact_patterns),
                'custom': len(self.custom_patterns)
            },
            'total_patterns': (
                len(self.personal_patterns) + 
                len(self.financial_patterns) + 
                len(self.medical_patterns) + 
                len(self.contact_patterns) + 
                len(self.custom_patterns)
            )
        }
    
    def preview_filtering(self, content: str, max_preview_length: int = 500) -> Dict[str, Any]:
        """Preview what would be filtered without actually filtering."""
        
        preview_result = {
            'would_filter': False,
            'potential_redactions': [],
            'preview_content': content[:max_preview_length] + '...' if len(content) > max_preview_length else content
        }
        
        # Check all pattern categories
        all_patterns = {}
        
        if self.config.redact_personal_info:
            all_patterns.update({f"personal_{k}": v for k, v in self.personal_patterns.items()})
        
        if self.config.redact_financial_info:
            all_patterns.update({f"financial_{k}": v for k, v in self.financial_patterns.items()})
        
        if self.config.redact_medical_info:
            all_patterns.update({f"medical_{k}": v for k, v in self.medical_patterns.items()})
        
        if self.config.redact_contact_info:
            all_patterns.update({f"contact_{k}": v for k, v in self.contact_patterns.items()})
        
        all_patterns.update({f"custom_{k}": v for k, v in self.custom_patterns.items()})
        
        # Find potential matches
        for pattern_name, pattern in all_patterns.items():
            matches = list(pattern.finditer(content))
            
            if matches:
                preview_result['would_filter'] = True
                
                for match in matches:
                    preview_result['potential_redactions'].append({
                        'pattern': pattern_name,
                        'text_preview': match.group()[:20] + '...' if len(match.group()) > 20 else match.group(),
                        'position': match.start(),
                        'length': len(match.group())
                    })
        
        return preview_result
