"""
Output validation guardrails for Agentix agents.
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from pydantic import BaseModel, Field

from ..utils.exceptions import ValidationError


class OutputValidationConfig(BaseModel):
    """Configuration for output validation."""
    
    # Format validation
    validate_json_format: bool = True
    validate_response_structure: bool = True
    required_fields: List[str] = Field(default_factory=list)
    
    # Content validation
    max_output_length: int = 50000
    min_output_length: int = 1
    block_sensitive_info: bool = True
    validate_factual_consistency: bool = False
    
    # Quality checks
    check_coherence: bool = True
    check_relevance: bool = True
    min_quality_score: float = 0.7
    
    # Safety checks
    block_harmful_content: bool = True
    block_misinformation: bool = True
    validate_citations: bool = False
    
    # Custom validation
    custom_validators: List[str] = Field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None


class OutputValidator:
    """
    Output validation guardrail for agent outputs.
    
    This validator provides:
    - Format and structure validation
    - Content quality assessment
    - Safety and appropriateness checks
    - Factual consistency validation
    - Custom validation rules
    """
    
    def __init__(self, config: OutputValidationConfig):
        self.config = config
        self.logger = logging.getLogger("agentix.guardrails.output_validation")
        
        # Compile patterns for validation
        self._compile_patterns()
        
        # Load custom validators
        self.custom_validators: Dict[str, Callable] = {}
        self._load_custom_validators()
    
    def _compile_patterns(self):
        """Compile regex patterns for validation."""
        
        # Sensitive information patterns
        self.sensitive_patterns = [
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),  # SSN
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),  # Credit card
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),  # Email
            re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),  # Phone number
        ]
        
        # Harmful content patterns
        self.harmful_patterns = [
            re.compile(r'\b(kill|murder|suicide|bomb|weapon)\b', re.IGNORECASE),
            re.compile(r'\b(hate|racist|sexist|discriminat)\w*\b', re.IGNORECASE),
            # Add more patterns as needed
        ]
        
        # Misinformation indicators
        self.misinformation_patterns = [
            re.compile(r'\b(definitely|absolutely|100%|always|never)\s+(true|false|correct|wrong)\b', re.IGNORECASE),
            re.compile(r'\bproven\s+fact\b', re.IGNORECASE),
            re.compile(r'\bscientists\s+say\b', re.IGNORECASE),
        ]
    
    def _load_custom_validators(self):
        """Load custom validation functions."""
        # Placeholder for loading custom validators
        pass
    
    def validate(self, output_data: Any, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output data.
        
        Args:
            output_data: Data to validate
            context: Additional context for validation
            
        Returns:
            Validation result
        """
        
        validation_result = {
            'valid': True,
            'quality_score': 1.0,
            'warnings': [],
            'errors': [],
            'suggestions': [],
            'metadata': {
                'validation_timestamp': datetime.now().isoformat(),
                'validator_version': '1.0.0'
            }
        }
        
        try:
            # Format validation
            self._validate_format(output_data, validation_result)
            
            # Content validation
            if isinstance(output_data, str):
                self._validate_content(output_data, validation_result)
            
            # Quality assessment
            self._assess_quality(output_data, validation_result, context)
            
            # Safety checks
            self._check_safety(output_data, validation_result)
            
            # Custom validation
            self._validate_custom(output_data, validation_result, context)
            
            # Final quality score check
            if validation_result['quality_score'] < self.config.min_quality_score:
                validation_result['valid'] = False
                validation_result['errors'].append(
                    f"Quality score {validation_result['quality_score']:.2f} below minimum {self.config.min_quality_score}"
                )
        
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
            self.logger.error(f"Output validation failed: {str(e)}")
        
        return validation_result
    
    def _validate_format(self, output_data: Any, result: Dict[str, Any]):
        """Validate output format and structure."""
        
        # JSON format validation
        if self.config.validate_json_format and isinstance(output_data, str):
            try:
                json.loads(output_data)
            except json.JSONDecodeError:
                # Not JSON, which might be fine
                pass
        
        # Length validation
        if isinstance(output_data, str):
            if len(output_data) < self.config.min_output_length:
                result['valid'] = False
                result['errors'].append(f"Output too short: {len(output_data)} < {self.config.min_output_length}")
                return
            
            if len(output_data) > self.config.max_output_length:
                result['valid'] = False
                result['errors'].append(f"Output too long: {len(output_data)} > {self.config.max_output_length}")
                return
        
        # Structure validation
        if self.config.validate_response_structure and isinstance(output_data, dict):
            self._validate_structure(output_data, result)
        
        # Schema validation
        if self.config.output_schema:
            self._validate_schema(output_data, result)
    
    def _validate_structure(self, output_dict: Dict[str, Any], result: Dict[str, Any]):
        """Validate response structure."""
        
        # Check required fields
        for field in self.config.required_fields:
            if field not in output_dict:
                result['valid'] = False
                result['errors'].append(f"Required field missing: {field}")
                return
        
        # Check for common response fields
        expected_fields = ['content', 'response', 'result', 'data']
        has_content_field = any(field in output_dict for field in expected_fields)
        
        if not has_content_field:
            result['warnings'].append("No standard content field found in response")
    
    def _validate_schema(self, output_data: Any, result: Dict[str, Any]):
        """Validate output against schema."""
        
        try:
            # Basic schema validation (simplified)
            schema = self.config.output_schema
            
            if 'type' in schema:
                expected_type = schema['type']
                actual_type = type(output_data).__name__
                
                if expected_type != actual_type:
                    result['valid'] = False
                    result['errors'].append(f"Type mismatch: expected {expected_type}, got {actual_type}")
                    return
            
            if 'properties' in schema and isinstance(output_data, dict):
                for prop, prop_schema in schema['properties'].items():
                    if prop in output_data:
                        # Recursively validate properties
                        prop_result = {'valid': True, 'errors': []}
                        self._validate_property(output_data[prop], prop_schema, prop_result)
                        
                        if not prop_result['valid']:
                            result['valid'] = False
                            result['errors'].extend(prop_result['errors'])
        
        except Exception as e:
            result['warnings'].append(f"Schema validation error: {str(e)}")
    
    def _validate_property(self, value: Any, schema: Dict[str, Any], result: Dict[str, Any]):
        """Validate a single property against its schema."""
        
        if 'type' in schema:
            expected_type = schema['type']
            actual_type = type(value).__name__
            
            if expected_type != actual_type:
                result['valid'] = False
                result['errors'].append(f"Property type mismatch: expected {expected_type}, got {actual_type}")
    
    def _validate_content(self, text: str, result: Dict[str, Any]):
        """Validate text content."""
        
        # Sensitive information check
        if self.config.block_sensitive_info:
            for pattern in self.sensitive_patterns:
                if pattern.search(text):
                    result['valid'] = False
                    result['errors'].append("Sensitive information detected in output")
                    return
        
        # Basic coherence check
        if self.config.check_coherence:
            coherence_score = self._assess_coherence(text)
            if coherence_score < 0.5:
                result['warnings'].append(f"Low coherence score: {coherence_score:.2f}")
                result['quality_score'] *= 0.8
    
    def _assess_quality(self, output_data: Any, result: Dict[str, Any], 
                       context: Optional[Dict[str, Any]]):
        """Assess output quality."""
        
        quality_factors = []
        
        if isinstance(output_data, str):
            # Length appropriateness
            length_score = self._assess_length_appropriateness(output_data, context)
            quality_factors.append(('length', length_score))
            
            # Coherence
            if self.config.check_coherence:
                coherence_score = self._assess_coherence(output_data)
                quality_factors.append(('coherence', coherence_score))
            
            # Relevance
            if self.config.check_relevance and context:
                relevance_score = self._assess_relevance(output_data, context)
                quality_factors.append(('relevance', relevance_score))
        
        # Calculate overall quality score
        if quality_factors:
            total_score = sum(score for _, score in quality_factors)
            result['quality_score'] = total_score / len(quality_factors)
            
            # Add individual scores to metadata
            result['metadata']['quality_factors'] = dict(quality_factors)
        
        # Quality suggestions
        if result['quality_score'] < 0.8:
            result['suggestions'].append("Consider improving response quality")
    
    def _assess_coherence(self, text: str) -> float:
        """Assess text coherence (simplified implementation)."""
        
        # Basic coherence indicators
        sentences = text.split('.')
        if len(sentences) < 2:
            return 0.8  # Short text, assume coherent
        
        # Check for transition words
        transition_words = ['however', 'therefore', 'furthermore', 'moreover', 'additionally']
        transition_count = sum(1 for word in transition_words if word in text.lower())
        
        # Simple scoring based on transitions and sentence structure
        coherence_score = min(1.0, 0.5 + (transition_count / len(sentences)) * 0.5)
        
        return coherence_score
    
    def _assess_length_appropriateness(self, text: str, context: Optional[Dict[str, Any]]) -> float:
        """Assess if text length is appropriate for the context."""
        
        text_length = len(text)
        
        # Default appropriate range
        min_appropriate = 50
        max_appropriate = 2000
        
        # Adjust based on context
        if context:
            query_length = len(context.get('query', ''))
            if query_length > 100:
                max_appropriate = 5000  # Longer queries may need longer responses
        
        if text_length < min_appropriate:
            return 0.5  # Too short
        elif text_length > max_appropriate:
            return 0.7  # Too long
        else:
            return 1.0  # Appropriate length
    
    def _assess_relevance(self, text: str, context: Dict[str, Any]) -> float:
        """Assess relevance to the input context."""
        
        query = context.get('query', '').lower()
        if not query:
            return 0.8  # No query to compare against
        
        text_lower = text.lower()
        
        # Simple keyword overlap
        query_words = set(query.split())
        text_words = set(text_lower.split())
        
        if not query_words:
            return 0.8
        
        overlap = len(query_words & text_words)
        relevance_score = min(1.0, overlap / len(query_words))
        
        return relevance_score
    
    def _check_safety(self, output_data: Any, result: Dict[str, Any]):
        """Perform safety checks on output."""
        
        if isinstance(output_data, str):
            # Harmful content check
            if self.config.block_harmful_content:
                for pattern in self.harmful_patterns:
                    if pattern.search(output_data):
                        result['valid'] = False
                        result['errors'].append("Harmful content detected")
                        return
            
            # Misinformation check
            if self.config.block_misinformation:
                misinformation_count = sum(1 for pattern in self.misinformation_patterns 
                                         if pattern.search(output_data))
                if misinformation_count > 2:
                    result['warnings'].append("Potential misinformation indicators detected")
                    result['quality_score'] *= 0.9
            
            # Citation validation
            if self.config.validate_citations:
                self._validate_citations(output_data, result)
    
    def _validate_citations(self, text: str, result: Dict[str, Any]):
        """Validate citations in the output."""
        
        # Look for citation patterns
        citation_patterns = [
            r'\[(\d+)\]',  # [1], [2], etc.
            r'\(([^)]+\d{4}[^)]*)\)',  # (Author 2023)
        ]
        
        citations_found = []
        for pattern in citation_patterns:
            citations_found.extend(re.findall(pattern, text))
        
        if citations_found:
            result['metadata']['citations_found'] = len(citations_found)
        else:
            result['suggestions'].append("Consider adding citations for factual claims")
    
    def _validate_custom(self, output_data: Any, result: Dict[str, Any],
                        context: Optional[Dict[str, Any]]):
        """Apply custom validation rules."""
        
        for validator_name, validator_func in self.custom_validators.items():
            try:
                validation_result = validator_func(output_data, context)
                
                if isinstance(validation_result, bool):
                    if not validation_result:
                        result['valid'] = False
                        result['errors'].append(f"Custom validator '{validator_name}' failed")
                        return
                elif isinstance(validation_result, dict):
                    # Detailed validation result
                    if not validation_result.get('valid', True):
                        result['valid'] = False
                        result['errors'].append(f"Custom validator '{validator_name}' failed")
                        return
                    
                    # Merge warnings and suggestions
                    result['warnings'].extend(validation_result.get('warnings', []))
                    result['suggestions'].extend(validation_result.get('suggestions', []))
                    
                    # Adjust quality score
                    if 'quality_score' in validation_result:
                        result['quality_score'] *= validation_result['quality_score']
            
            except Exception as e:
                result['warnings'].append(f"Custom validator '{validator_name}' error: {str(e)}")
    
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
            'custom_validators': list(self.custom_validators.keys())
        }
