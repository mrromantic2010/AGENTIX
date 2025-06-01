"""
Safety checker guardrails for Agentix agents.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from ..utils.exceptions import SecurityError


class SafetyConfig(BaseModel):
    """Configuration for safety checker."""
    
    # Content safety
    check_harmful_content: bool = True
    check_hate_speech: bool = True
    check_violence: bool = True
    check_self_harm: bool = True
    
    # Privacy protection
    check_personal_info: bool = True
    check_financial_info: bool = True
    check_medical_info: bool = True
    
    # Security checks
    check_malicious_code: bool = True
    check_injection_attacks: bool = True
    check_unauthorized_access: bool = True
    
    # Compliance
    enforce_content_policy: bool = True
    log_safety_violations: bool = True
    
    # Thresholds
    safety_threshold: float = 0.8
    confidence_threshold: float = 0.7


class SafetyChecker:
    """
    Safety checker guardrail for comprehensive safety validation.
    
    This checker provides:
    - Content safety assessment
    - Privacy protection
    - Security threat detection
    - Compliance enforcement
    """
    
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.logger = logging.getLogger("agentix.guardrails.safety_checker")
        self.violation_log: List[Dict[str, Any]] = []
    
    def check_safety(self, content: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive safety check.
        
        Args:
            content: Content to check
            context: Additional context
            
        Returns:
            Safety assessment result
        """
        
        safety_result = {
            'safe': True,
            'safety_score': 1.0,
            'violations': [],
            'warnings': [],
            'categories_checked': [],
            'metadata': {
                'check_timestamp': datetime.now().isoformat(),
                'checker_version': '1.0.0'
            }
        }
        
        try:
            # Content safety checks
            if self.config.check_harmful_content:
                self._check_harmful_content(content, safety_result)
                safety_result['categories_checked'].append('harmful_content')
            
            # Privacy checks
            if self.config.check_personal_info:
                self._check_privacy_violations(content, safety_result)
                safety_result['categories_checked'].append('privacy')
            
            # Security checks
            if self.config.check_malicious_code:
                self._check_security_threats(content, safety_result)
                safety_result['categories_checked'].append('security')
            
            # Apply safety threshold
            if safety_result['safety_score'] < self.config.safety_threshold:
                safety_result['safe'] = False
                safety_result['violations'].append(
                    f"Safety score {safety_result['safety_score']:.2f} below threshold {self.config.safety_threshold}"
                )
            
            # Log violations if configured
            if self.config.log_safety_violations and safety_result['violations']:
                self._log_violation(content, safety_result, context)
        
        except Exception as e:
            safety_result['safe'] = False
            safety_result['violations'].append(f"Safety check error: {str(e)}")
            self.logger.error(f"Safety check failed: {str(e)}")
        
        return safety_result
    
    def _check_harmful_content(self, content: Any, result: Dict[str, Any]):
        """Check for harmful content."""
        
        if not isinstance(content, str):
            return
        
        content_lower = content.lower()
        
        # Hate speech indicators
        if self.config.check_hate_speech:
            hate_keywords = [
                'hate', 'racist', 'sexist', 'homophobic', 'xenophobic',
                'discrimination', 'prejudice', 'bigotry'
            ]
            
            hate_score = sum(1 for keyword in hate_keywords if keyword in content_lower)
            if hate_score > 2:
                result['safe'] = False
                result['violations'].append("Potential hate speech detected")
                result['safety_score'] *= 0.3
                return
            elif hate_score > 0:
                result['warnings'].append("Mild hate speech indicators detected")
                result['safety_score'] *= 0.8
        
        # Violence indicators
        if self.config.check_violence:
            violence_keywords = [
                'kill', 'murder', 'assault', 'attack', 'violence',
                'weapon', 'bomb', 'shoot', 'stab', 'hurt'
            ]
            
            violence_score = sum(1 for keyword in violence_keywords if keyword in content_lower)
            if violence_score > 3:
                result['safe'] = False
                result['violations'].append("Violent content detected")
                result['safety_score'] *= 0.2
                return
            elif violence_score > 1:
                result['warnings'].append("Violence indicators detected")
                result['safety_score'] *= 0.7
        
        # Self-harm indicators
        if self.config.check_self_harm:
            self_harm_keywords = [
                'suicide', 'self-harm', 'cutting', 'overdose',
                'end my life', 'kill myself', 'hurt myself'
            ]
            
            for keyword in self_harm_keywords:
                if keyword in content_lower:
                    result['safe'] = False
                    result['violations'].append("Self-harm content detected")
                    result['safety_score'] *= 0.1
                    return
    
    def _check_privacy_violations(self, content: Any, result: Dict[str, Any]):
        """Check for privacy violations."""
        
        if not isinstance(content, str):
            return
        
        import re
        
        # Personal information patterns
        personal_patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        }
        
        violations_found = []
        
        for info_type, pattern in personal_patterns.items():
            if re.search(pattern, content):
                violations_found.append(info_type)
        
        if violations_found:
            result['safe'] = False
            result['violations'].append(f"Personal information detected: {', '.join(violations_found)}")
            result['safety_score'] *= 0.4
        
        # Financial information
        if self.config.check_financial_info:
            financial_keywords = [
                'bank account', 'routing number', 'pin number',
                'social security', 'tax id', 'account number'
            ]
            
            financial_score = sum(1 for keyword in financial_keywords if keyword in content.lower())
            if financial_score > 0:
                result['warnings'].append("Potential financial information detected")
                result['safety_score'] *= 0.9
        
        # Medical information
        if self.config.check_medical_info:
            medical_keywords = [
                'medical record', 'diagnosis', 'prescription',
                'patient id', 'health insurance', 'medical history'
            ]
            
            medical_score = sum(1 for keyword in medical_keywords if keyword in content.lower())
            if medical_score > 1:
                result['warnings'].append("Potential medical information detected")
                result['safety_score'] *= 0.9
    
    def _check_security_threats(self, content: Any, result: Dict[str, Any]):
        """Check for security threats."""
        
        if not isinstance(content, str):
            return
        
        import re
        
        # Malicious code patterns
        malicious_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'shell_exec\s*\(',
            r'passthru\s*\(',
            r'file_get_contents\s*\(',
            r'curl_exec\s*\('
        ]
        
        malicious_found = []
        for pattern in malicious_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                malicious_found.append(pattern)
        
        if malicious_found:
            result['safe'] = False
            result['violations'].append("Malicious code patterns detected")
            result['safety_score'] *= 0.2
            return
        
        # Injection attack patterns
        if self.config.check_injection_attacks:
            injection_patterns = [
                r"'.*OR.*'.*'",  # SQL injection
                r"UNION.*SELECT",  # SQL injection
                r"DROP.*TABLE",  # SQL injection
                r"<.*>.*<\/.*>",  # XSS
                r"alert\s*\(",  # XSS
                r"document\.cookie",  # XSS
            ]
            
            injection_found = []
            for pattern in injection_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    injection_found.append(pattern)
            
            if injection_found:
                result['safe'] = False
                result['violations'].append("Injection attack patterns detected")
                result['safety_score'] *= 0.3
                return
        
        # Unauthorized access attempts
        if self.config.check_unauthorized_access:
            access_keywords = [
                'admin password', 'root access', 'bypass security',
                'privilege escalation', 'unauthorized access',
                'hack', 'crack', 'exploit'
            ]
            
            access_score = sum(1 for keyword in access_keywords if keyword in content.lower())
            if access_score > 2:
                result['safe'] = False
                result['violations'].append("Unauthorized access indicators detected")
                result['safety_score'] *= 0.4
            elif access_score > 0:
                result['warnings'].append("Security-related keywords detected")
                result['safety_score'] *= 0.8
    
    def _log_violation(self, content: Any, safety_result: Dict[str, Any], 
                      context: Optional[Dict[str, Any]]):
        """Log safety violation."""
        
        violation_entry = {
            'timestamp': datetime.now().isoformat(),
            'content_preview': str(content)[:200] if content else '',
            'violations': safety_result['violations'],
            'safety_score': safety_result['safety_score'],
            'context': context or {},
            'categories_checked': safety_result['categories_checked']
        }
        
        self.violation_log.append(violation_entry)
        
        # Keep only recent violations (last 1000)
        if len(self.violation_log) > 1000:
            self.violation_log = self.violation_log[-1000:]
        
        # Log to system logger
        self.logger.warning(f"Safety violation detected: {safety_result['violations']}")
    
    def get_violation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent safety violations."""
        return self.violation_log[-limit:]
    
    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety checker statistics."""
        
        total_violations = len(self.violation_log)
        recent_violations = len([v for v in self.violation_log 
                               if datetime.fromisoformat(v['timestamp']) > 
                               datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)])
        
        return {
            'total_violations': total_violations,
            'recent_violations_today': recent_violations,
            'config': self.config.dict(),
            'violation_categories': self._get_violation_categories()
        }
    
    def _get_violation_categories(self) -> Dict[str, int]:
        """Get violation counts by category."""
        
        categories = {}
        
        for violation in self.violation_log:
            for category in violation['categories_checked']:
                categories[category] = categories.get(category, 0) + 1
        
        return categories
    
    def clear_violation_history(self):
        """Clear violation history."""
        self.violation_log.clear()
        self.logger.info("Safety violation history cleared")


class ContentFilter:
    """Content filtering component."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("agentix.guardrails.content_filter")
    
    def filter_content(self, content: str) -> Dict[str, Any]:
        """Filter content and return filtered version."""
        
        filtered_content = content
        modifications = []
        
        # Remove sensitive patterns
        import re
        
        # Replace SSNs
        ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
        if re.search(ssn_pattern, filtered_content):
            filtered_content = re.sub(ssn_pattern, '[SSN REDACTED]', filtered_content)
            modifications.append('SSN redacted')
        
        # Replace credit card numbers
        cc_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'
        if re.search(cc_pattern, filtered_content):
            filtered_content = re.sub(cc_pattern, '[CREDIT CARD REDACTED]', filtered_content)
            modifications.append('Credit card number redacted')
        
        # Replace email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        if re.search(email_pattern, filtered_content):
            filtered_content = re.sub(email_pattern, '[EMAIL REDACTED]', filtered_content)
            modifications.append('Email address redacted')
        
        return {
            'filtered_content': filtered_content,
            'modifications': modifications,
            'original_length': len(content),
            'filtered_length': len(filtered_content)
        }


# Content filter configuration
class ContentFilterConfig(BaseModel):
    """Configuration for content filter."""
    
    redact_personal_info: bool = True
    redact_financial_info: bool = True
    redact_medical_info: bool = True
    replacement_text: str = "[REDACTED]"
