"""
OpenRouter LLM client for Agentix.

OpenRouter provides access to multiple LLM models through a unified API,
including models from OpenAI, Anthropic, Google, Meta, Mistral, and more.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import aiohttp
from datetime import datetime

from ..config.llm_config import LLMConfig, LLMProvider
from ..utils.exceptions import LLMError
from .base_client import BaseLLMClient, LLMResponse, LLMStreamResponse


class OpenRouterClient(BaseLLMClient):
    """
    OpenRouter LLM client implementation.
    
    OpenRouter provides access to many models:
    - OpenAI: openai/gpt-4-turbo, openai/gpt-3.5-turbo
    - Anthropic: anthropic/claude-3-opus, anthropic/claude-3-sonnet
    - Google: google/gemini-pro, google/gemini-pro-vision
    - Meta: meta-llama/llama-2-70b-chat, meta-llama/codellama-34b-instruct
    - Mistral: mistralai/mixtral-8x7b-instruct, mistralai/mistral-7b-instruct
    - And many more...
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize OpenRouter client."""
        super().__init__(config)
        
        if config.provider != LLMProvider.OPENROUTER:
            raise ValueError("Config must be for OpenRouter provider")
        
        if not config.api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.api_url = config.get_api_url()
        self.headers = config.get_headers()
        self.logger = logging.getLogger(f"agentix.llm.openrouter")
        
        # OpenRouter-specific settings
        self.site_url = config.provider_config.get('site_url', 'https://agentix.dev')
        self.app_name = config.provider_config.get('app_name', 'Agentix Framework')
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      **kwargs) -> LLMResponse:
        """
        Generate a response using OpenRouter.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with the generated content
        """
        
        # Prepare request payload
        payload = self._prepare_payload(messages, **kwargs)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"OpenRouter API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    
                    # Extract response
                    if 'choices' not in result or not result['choices']:
                        raise LLMError("No choices in OpenRouter response")
                    
                    choice = result['choices'][0]
                    content = choice.get('message', {}).get('content', '')
                    
                    # Extract usage information
                    usage = result.get('usage', {})
                    
                    return LLMResponse(
                        content=content,
                        model=result.get('model', self.config.model),
                        usage={
                            'prompt_tokens': usage.get('prompt_tokens', 0),
                            'completion_tokens': usage.get('completion_tokens', 0),
                            'total_tokens': usage.get('total_tokens', 0)
                        },
                        finish_reason=choice.get('finish_reason', 'stop'),
                        metadata={
                            'provider': 'openrouter',
                            'response_id': result.get('id'),
                            'created': result.get('created'),
                            'provider_name': result.get('provider', {}).get('name'),
                            'model_name': result.get('model')
                        }
                    )
        
        except aiohttp.ClientError as e:
            raise LLMError(f"OpenRouter client error: {str(e)}")
        except Exception as e:
            raise LLMError(f"OpenRouter generation error: {str(e)}")
    
    async def stream_generate(self, 
                             messages: List[Dict[str, str]], 
                             **kwargs) -> AsyncGenerator[LLMStreamResponse, None]:
        """
        Generate a streaming response using OpenRouter.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional generation parameters
            
        Yields:
            LLMStreamResponse objects with incremental content
        """
        
        # Prepare payload with streaming enabled
        payload = self._prepare_payload(messages, stream=True, **kwargs)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"OpenRouter API error {response.status}: {error_text}")
                    
                    # Process streaming response
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        
                        if not line or not line.startswith('data: '):
                            continue
                        
                        if line == 'data: [DONE]':
                            break
                        
                        try:
                            # Parse JSON data
                            data = json.loads(line[6:])  # Remove 'data: ' prefix
                            
                            if 'choices' in data and data['choices']:
                                choice = data['choices'][0]
                                delta = choice.get('delta', {})
                                content = delta.get('content', '')
                                
                                if content:
                                    yield LLMStreamResponse(
                                        content=content,
                                        is_complete=False,
                                        metadata={
                                            'provider': 'openrouter',
                                            'model': data.get('model', self.config.model),
                                            'response_id': data.get('id')
                                        }
                                    )
                                
                                # Check if streaming is complete
                                if choice.get('finish_reason'):
                                    yield LLMStreamResponse(
                                        content='',
                                        is_complete=True,
                                        metadata={
                                            'provider': 'openrouter',
                                            'finish_reason': choice.get('finish_reason'),
                                            'model': data.get('model', self.config.model)
                                        }
                                    )
                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
        
        except aiohttp.ClientError as e:
            raise LLMError(f"OpenRouter streaming error: {str(e)}")
        except Exception as e:
            raise LLMError(f"OpenRouter streaming error: {str(e)}")
    
    def _prepare_payload(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Prepare the request payload for OpenRouter API."""
        
        # Get base configuration
        provider_config = self.config.get_provider_config()
        
        # Build payload
        payload = {
            'model': self.config.model,
            'messages': messages,
            'temperature': kwargs.get('temperature', provider_config.get('temperature')),
            'max_tokens': kwargs.get('max_tokens', provider_config.get('max_tokens')),
            'top_p': kwargs.get('top_p', provider_config.get('top_p')),
            'frequency_penalty': kwargs.get('frequency_penalty', provider_config.get('frequency_penalty')),
            'presence_penalty': kwargs.get('presence_penalty', provider_config.get('presence_penalty')),
            'stream': kwargs.get('stream', provider_config.get('stream', False))
        }
        
        # Add OpenRouter-specific parameters
        if 'top_k' in provider_config:
            payload['top_k'] = provider_config['top_k']
        
        if 'repetition_penalty' in provider_config:
            payload['repetition_penalty'] = provider_config['repetition_penalty']
        
        if 'min_p' in provider_config:
            payload['min_p'] = provider_config['min_p']
        
        if 'top_a' in provider_config:
            payload['top_a'] = provider_config['top_a']
        
        # Add function calling support if enabled
        if self.config.enable_function_calling and 'functions' in kwargs:
            payload['functions'] = kwargs['functions']
            if 'function_call' in kwargs:
                payload['function_call'] = kwargs['function_call']
        
        # Add tools support (newer format)
        if 'tools' in kwargs:
            payload['tools'] = kwargs['tools']
            if 'tool_choice' in kwargs:
                payload['tool_choice'] = kwargs['tool_choice']
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        return payload
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models from OpenRouter.
        
        Returns:
            List of model information dictionaries
        """
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_url}/models",
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"OpenRouter models API error {response.status}: {error_text}")
                    
                    result = await response.json()
                    return result.get('data', [])
        
        except Exception as e:
            self.logger.error(f"Failed to get OpenRouter models: {str(e)}")
            return []
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model information dictionary or None if not found
        """
        
        models = await self.get_available_models()
        
        for model in models:
            if model.get('id') == model_name:
                return model
        
        return None
    
    def get_popular_models(self) -> Dict[str, str]:
        """
        Get a dictionary of popular models available on OpenRouter.
        
        Returns:
            Dictionary mapping model categories to model names
        """
        
        return {
            # OpenAI Models
            'gpt-4-turbo': 'openai/gpt-4-turbo',
            'gpt-4': 'openai/gpt-4',
            'gpt-3.5-turbo': 'openai/gpt-3.5-turbo',
            
            # Anthropic Models
            'claude-3-opus': 'anthropic/claude-3-opus',
            'claude-3-sonnet': 'anthropic/claude-3-sonnet',
            'claude-3-haiku': 'anthropic/claude-3-haiku',
            
            # Google Models
            'gemini-pro': 'google/gemini-pro',
            'gemini-pro-vision': 'google/gemini-pro-vision',
            
            # Meta Models
            'llama-2-70b': 'meta-llama/llama-2-70b-chat',
            'codellama-34b': 'meta-llama/codellama-34b-instruct',
            
            # Mistral Models
            'mixtral-8x7b': 'mistralai/mixtral-8x7b-instruct',
            'mistral-7b': 'mistralai/mistral-7b-instruct',
            
            # Other Popular Models
            'nous-hermes-2-mixtral': 'nousresearch/nous-hermes-2-mixtral-8x7b-dpo',
            'dolphin-mixtral': 'cognitivecomputations/dolphin-mixtral-8x7b',
            'openchat-7b': 'openchat/openchat-7b',
            'zephyr-7b': 'huggingfaceh4/zephyr-7b-beta'
        }


# Convenience function to create OpenRouter client
def create_openrouter_client(api_key: str, 
                           model: str = "openai/gpt-4-turbo",
                           **kwargs) -> OpenRouterClient:
    """
    Create an OpenRouter client with the specified configuration.
    
    Args:
        api_key: OpenRouter API key
        model: Model name to use
        **kwargs: Additional configuration options
        
    Returns:
        Configured OpenRouterClient instance
    """
    
    from ..config.llm_config import LLMConfig, LLMProvider
    
    config = LLMConfig(
        provider=LLMProvider.OPENROUTER,
        model=model,
        api_key=api_key,
        **kwargs
    )
    
    return OpenRouterClient(config)
