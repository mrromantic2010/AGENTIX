"""
Anthropic Claude LLM client for Agentix.

This module provides direct integration with Anthropic's Claude models,
supporting all Claude-3 variants (Opus, Sonnet, Haiku) and Claude-2 models.
"""

import asyncio
import json
import logging
from typing import Dict, Any, List, Optional, AsyncGenerator, Union
import aiohttp
from datetime import datetime

from ..config.llm_config import LLMConfig, LLMProvider
from ..utils.exceptions import LLMError, RateLimitError, AuthenticationError
from .base_client import BaseLLMClient, LLMResponse, LLMStreamResponse


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude LLM client implementation.

    Supports all Claude models:
    - Claude-3 Opus: claude-3-opus-20240229 (Most capable)
    - Claude-3 Sonnet: claude-3-sonnet-20240229 (Balanced)
    - Claude-3 Haiku: claude-3-haiku-20240307 (Fastest)
    - Claude-2.1: claude-2.1 (Previous generation)
    - Claude-2.0: claude-2.0 (Legacy)
    """

    # Claude model specifications (updated with latest models)
    CLAUDE_MODELS = {
        'claude-3-5-sonnet-20241022': {
            'name': 'Claude-3.5 Sonnet',
            'max_tokens': 8192,
            'context_window': 200000,
            'supports_vision': True,
            'supports_tools': True,
            'cost_per_1k_input': 0.003,
            'cost_per_1k_output': 0.015
        },
        'claude-3-5-sonnet-latest': {
            'name': 'Claude-3.5 Sonnet (Latest)',
            'max_tokens': 8192,
            'context_window': 200000,
            'supports_vision': True,
            'supports_tools': True,
            'cost_per_1k_input': 0.003,
            'cost_per_1k_output': 0.015
        },
        'claude-3-opus-20240229': {
            'name': 'Claude-3 Opus',
            'max_tokens': 4096,
            'context_window': 200000,
            'supports_vision': True,
            'supports_tools': True,
            'cost_per_1k_input': 0.015,
            'cost_per_1k_output': 0.075
        },
        'claude-3-sonnet-20240229': {
            'name': 'Claude-3 Sonnet',
            'max_tokens': 4096,
            'context_window': 200000,
            'supports_vision': True,
            'supports_tools': True,
            'cost_per_1k_input': 0.003,
            'cost_per_1k_output': 0.015
        },
        'claude-3-haiku-20240307': {
            'name': 'Claude-3 Haiku',
            'max_tokens': 4096,
            'context_window': 200000,
            'supports_vision': True,
            'supports_tools': True,
            'cost_per_1k_input': 0.00025,
            'cost_per_1k_output': 0.00125
        },
        'claude-2.1': {
            'name': 'Claude-2.1',
            'max_tokens': 4096,
            'context_window': 200000,
            'supports_vision': False,
            'supports_tools': False,
            'cost_per_1k_input': 0.008,
            'cost_per_1k_output': 0.024
        }
    }

    def __init__(self, config: LLMConfig):
        """Initialize Anthropic client."""
        super().__init__(config)

        if config.provider != LLMProvider.ANTHROPIC:
            raise ValueError("Config must be for Anthropic provider")

        if not config.api_key:
            raise ValueError("Anthropic API key is required")

        # Validate model
        if config.model not in self.CLAUDE_MODELS:
            available = ', '.join(self.CLAUDE_MODELS.keys())
            raise ValueError(f"Unsupported Claude model: {config.model}. Available: {available}")

        self.api_url = config.get_api_url() or "https://api.anthropic.com"
        self.headers = self._build_headers()
        self.logger = logging.getLogger(f"agentix.llm.anthropic")

        # Model-specific settings
        self.model_info = self.CLAUDE_MODELS[config.model]
        self.supports_vision = self.model_info['supports_vision']
        self.supports_tools = self.model_info['supports_tools']

        # Rate limiting
        self.rate_limit_requests_per_minute = 1000
        self.rate_limit_tokens_per_minute = 100000
        self.request_count = 0
        self.token_count = 0
        self.last_reset = datetime.now()

    def _build_headers(self) -> Dict[str, str]:
        """Build headers for Anthropic API requests."""
        return {
            'Content-Type': 'application/json',
            'x-api-key': self.config.api_key,
            'anthropic-version': '2023-06-01',
            'User-Agent': 'Agentix/1.0'
        }

    async def generate(self,
                      messages: List[Dict[str, str]],
                      **kwargs) -> LLMResponse:
        """
        Generate a response using Anthropic Claude.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters

        Returns:
            LLMResponse with the generated content
        """

        # Check rate limits
        await self._check_rate_limits()

        # Prepare request payload
        payload = self._prepare_payload(messages, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/v1/messages",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:

                    # Handle rate limiting
                    if response.status == 429:
                        retry_after = int(response.headers.get('retry-after', 60))
                        raise RateLimitError(
                            f"Anthropic rate limit exceeded. Retry after {retry_after} seconds",
                            limit=self.rate_limit_requests_per_minute,
                            window=60
                        )

                    # Handle authentication errors
                    if response.status == 401:
                        raise AuthenticationError(
                            "Invalid Anthropic API key",
                            provider="anthropic"
                        )

                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Anthropic API error {response.status}: {error_text}")

                    result = await response.json()

                    # Extract response
                    if 'content' not in result or not result['content']:
                        raise LLMError("No content in Anthropic response")

                    # Claude returns content as a list
                    content_blocks = result['content']
                    content = ""

                    for block in content_blocks:
                        if block.get('type') == 'text':
                            content += block.get('text', '')

                    # Extract usage information
                    usage = result.get('usage', {})
                    input_tokens = usage.get('input_tokens', 0)
                    output_tokens = usage.get('output_tokens', 0)

                    # Update rate limiting counters
                    self.token_count += input_tokens + output_tokens
                    self.request_count += 1

                    return LLMResponse(
                        content=content,
                        model=result.get('model', self.config.model),
                        usage={
                            'prompt_tokens': input_tokens,
                            'completion_tokens': output_tokens,
                            'total_tokens': input_tokens + output_tokens
                        },
                        finish_reason=result.get('stop_reason', 'stop'),
                        metadata={
                            'provider': 'anthropic',
                            'response_id': result.get('id'),
                            'model_info': self.model_info,
                            'supports_vision': self.supports_vision,
                            'supports_tools': self.supports_tools
                        }
                    )

        except aiohttp.ClientError as e:
            raise LLMError(f"Anthropic client error: {str(e)}")
        except Exception as e:
            if isinstance(e, (LLMError, RateLimitError, AuthenticationError)):
                raise
            raise LLMError(f"Anthropic generation error: {str(e)}")

    async def stream_generate(self,
                             messages: List[Dict[str, str]],
                             **kwargs) -> AsyncGenerator[LLMStreamResponse, None]:
        """
        Generate a streaming response using Anthropic Claude.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional generation parameters

        Yields:
            LLMStreamResponse objects with incremental content
        """

        # Check rate limits
        await self._check_rate_limits()

        # Prepare payload with streaming enabled
        payload = self._prepare_payload(messages, stream=True, **kwargs)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/v1/messages",
                    headers=self.headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:

                    if response.status == 429:
                        retry_after = int(response.headers.get('retry-after', 60))
                        raise RateLimitError(
                            f"Anthropic rate limit exceeded. Retry after {retry_after} seconds"
                        )

                    if response.status == 401:
                        raise AuthenticationError("Invalid Anthropic API key", provider="anthropic")

                    if response.status != 200:
                        error_text = await response.text()
                        raise LLMError(f"Anthropic API error {response.status}: {error_text}")

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

                            event_type = data.get('type')

                            if event_type == 'content_block_delta':
                                delta = data.get('delta', {})
                                if delta.get('type') == 'text_delta':
                                    content = delta.get('text', '')

                                    if content:
                                        yield LLMStreamResponse(
                                            content=content,
                                            is_complete=False,
                                            metadata={
                                                'provider': 'anthropic',
                                                'model': self.config.model,
                                                'event_type': event_type
                                            }
                                        )

                            elif event_type == 'message_stop':
                                yield LLMStreamResponse(
                                    content='',
                                    is_complete=True,
                                    metadata={
                                        'provider': 'anthropic',
                                        'finish_reason': 'stop',
                                        'model': self.config.model
                                    }
                                )

                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue

                    # Update counters
                    self.request_count += 1

        except aiohttp.ClientError as e:
            raise LLMError(f"Anthropic streaming error: {str(e)}")
        except Exception as e:
            if isinstance(e, (LLMError, RateLimitError, AuthenticationError)):
                raise
            raise LLMError(f"Anthropic streaming error: {str(e)}")

    def _prepare_payload(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Prepare the request payload for Anthropic API."""

        # Get base configuration
        provider_config = self.config.get_provider_config()

        # Separate system message from conversation
        system_message = ""
        conversation_messages = []

        for message in messages:
            if message.get('role') == 'system':
                system_message = message.get('content', '')
            else:
                conversation_messages.append(message)

        # Build payload
        payload = {
            'model': self.config.model,
            'messages': conversation_messages,
            'max_tokens': kwargs.get('max_tokens', provider_config.get('max_tokens', 4096)),
            'temperature': kwargs.get('temperature', provider_config.get('temperature')),
            'top_p': kwargs.get('top_p', provider_config.get('top_p')),
            'stream': kwargs.get('stream', provider_config.get('stream', False))
        }

        # Add system message if present
        if system_message:
            payload['system'] = system_message

        # Add tool support for Claude-3 models
        if self.supports_tools and 'tools' in kwargs:
            payload['tools'] = kwargs['tools']
            if 'tool_choice' in kwargs:
                payload['tool_choice'] = kwargs['tool_choice']

        # Add stop sequences
        if 'stop' in kwargs:
            payload['stop_sequences'] = kwargs['stop']

        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}

        return payload

    async def _check_rate_limits(self):
        """Check and enforce rate limits."""
        now = datetime.now()

        # Reset counters every minute
        if (now - self.last_reset).total_seconds() >= 60:
            self.request_count = 0
            self.token_count = 0
            self.last_reset = now

        # Check request rate limit
        if self.request_count >= self.rate_limit_requests_per_minute:
            raise RateLimitError(
                "Request rate limit exceeded",
                limit=self.rate_limit_requests_per_minute,
                window=60
            )

        # Check token rate limit
        if self.token_count >= self.rate_limit_tokens_per_minute:
            raise RateLimitError(
                "Token rate limit exceeded",
                limit=self.rate_limit_tokens_per_minute,
                window=60
            )

    def get_model_capabilities(self) -> Dict[str, Any]:
        """Get capabilities of the current Claude model."""
        return {
            **self.model_info,
            'provider': 'anthropic',
            'model': self.config.model,
            'streaming': True,
            'function_calling': self.supports_tools,
            'vision': self.supports_vision
        }

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage."""
        input_cost = (input_tokens / 1000) * self.model_info['cost_per_1k_input']
        output_cost = (output_tokens / 1000) * self.model_info['cost_per_1k_output']
        return input_cost + output_cost


# Convenience function to create Anthropic client
def create_anthropic_client(api_key: str,
                          model: str = "claude-3-sonnet-20240229",
                          **kwargs) -> AnthropicClient:
    """
    Create an Anthropic client with the specified configuration.

    Args:
        api_key: Anthropic API key
        model: Claude model name to use
        **kwargs: Additional configuration options

    Returns:
        Configured AnthropicClient instance
    """

    from ..config.llm_config import LLMConfig, LLMProvider

    config = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model=model,
        api_key=api_key,
        **kwargs
    )

    return AnthropicClient(config)
