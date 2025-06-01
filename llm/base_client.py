"""
Base LLM client classes for Agentix.

This module provides the base classes and interfaces for LLM clients.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

from ..config.llm_config import LLMConfig


@dataclass
class LLMResponse:
    """Response from an LLM generation request."""
    
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class LLMStreamResponse:
    """Response chunk from an LLM streaming request."""
    
    content: str
    is_complete: bool
    metadata: Dict[str, Any]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseLLMClient(ABC):
    """
    Base class for all LLM clients.
    
    This provides the common interface that all LLM providers must implement.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the LLM client with configuration."""
        self.config = config
    
    @abstractmethod
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      **kwargs) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            LLMResponse with the generated content
        """
        pass
    
    @abstractmethod
    async def stream_generate(self, 
                             messages: List[Dict[str, str]], 
                             **kwargs) -> AsyncGenerator[LLMStreamResponse, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional generation parameters
            
        Yields:
            LLMStreamResponse objects with incremental content
        """
        pass
    
    def format_messages(self, 
                       prompt: str, 
                       system_prompt: Optional[str] = None,
                       conversation_history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
        """
        Format a prompt into the messages format expected by the LLM.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            conversation_history: Optional conversation history
            
        Returns:
            List of formatted message dictionaries
        """
        
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add the current user prompt
        messages.append({
            "role": "user", 
            "content": prompt
        })
        
        return messages
    
    async def simple_generate(self, 
                             prompt: str,
                             system_prompt: Optional[str] = None,
                             **kwargs) -> str:
        """
        Simple generation method that takes a prompt and returns a string.
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters
            
        Returns:
            Generated content as string
        """
        
        messages = self.format_messages(prompt, system_prompt)
        response = await self.generate(messages, **kwargs)
        return response.content
    
    async def chat_generate(self,
                           messages: List[Dict[str, str]],
                           **kwargs) -> str:
        """
        Chat generation method for conversation-style interactions.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional generation parameters
            
        Returns:
            Generated content as string
        """
        
        response = await self.generate(messages, **kwargs)
        return response.content
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        
        return {
            "provider": self.config.provider.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "supports_streaming": self.config.enable_streaming,
            "supports_function_calling": self.config.enable_function_calling
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        This is a rough estimation. For accurate token counting,
        use the provider's specific tokenizer.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated number of tokens
        """
        
        # Rough estimation: ~4 characters per token for English text
        return len(text) // 4
    
    def validate_messages(self, messages: List[Dict[str, str]]) -> bool:
        """
        Validate that messages are in the correct format.
        
        Args:
            messages: List of message dictionaries to validate
            
        Returns:
            True if valid, False otherwise
        """
        
        if not isinstance(messages, list):
            return False
        
        for message in messages:
            if not isinstance(message, dict):
                return False
            
            if "role" not in message or "content" not in message:
                return False
            
            if message["role"] not in ["system", "user", "assistant", "function"]:
                return False
            
            if not isinstance(message["content"], str):
                return False
        
        return True


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for testing and development.
    
    This client returns predefined responses without making API calls.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize mock client."""
        super().__init__(config)
        
        self.responses = {
            "default": "This is a mock response from the LLM client.",
            "greeting": "Hello! I'm a mock AI assistant. How can I help you today?",
            "ai": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence.",
            "machine learning": "Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
            "error": "I'm sorry, but I encountered an error processing your request."
        }
    
    async def generate(self, 
                      messages: List[Dict[str, str]], 
                      **kwargs) -> LLMResponse:
        """Generate a mock response."""
        
        if not self.validate_messages(messages):
            raise ValueError("Invalid messages format")
        
        # Get the last user message
        user_message = ""
        for message in reversed(messages):
            if message["role"] == "user":
                user_message = message["content"].lower()
                break
        
        # Select response based on content
        if "hello" in user_message or "hi" in user_message:
            content = self.responses["greeting"]
        elif "ai" in user_message or "artificial intelligence" in user_message:
            content = self.responses["ai"]
        elif "machine learning" in user_message or "ml" in user_message:
            content = self.responses["machine learning"]
        else:
            content = self.responses["default"]
        
        return LLMResponse(
            content=content,
            model=self.config.model,
            usage={
                "prompt_tokens": self.estimate_tokens(str(messages)),
                "completion_tokens": self.estimate_tokens(content),
                "total_tokens": self.estimate_tokens(str(messages)) + self.estimate_tokens(content)
            },
            finish_reason="stop",
            metadata={
                "provider": "mock",
                "mock_response": True
            }
        )
    
    async def stream_generate(self, 
                             messages: List[Dict[str, str]], 
                             **kwargs) -> AsyncGenerator[LLMStreamResponse, None]:
        """Generate a mock streaming response."""
        
        # Get the full response first
        response = await self.generate(messages, **kwargs)
        content = response.content
        
        # Stream it word by word
        words = content.split()
        
        for i, word in enumerate(words):
            # Add space before word (except first)
            chunk = f" {word}" if i > 0 else word
            
            yield LLMStreamResponse(
                content=chunk,
                is_complete=False,
                metadata={
                    "provider": "mock",
                    "chunk_index": i,
                    "total_chunks": len(words)
                }
            )
            
            # Small delay to simulate streaming
            import asyncio
            await asyncio.sleep(0.1)
        
        # Final completion signal
        yield LLMStreamResponse(
            content="",
            is_complete=True,
            metadata={
                "provider": "mock",
                "finish_reason": "stop"
            }
        )
