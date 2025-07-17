"""
vLLM Client Integration for Local LLM Operations.

This module provides a client for interacting with locally hosted vLLM servers,
enabling abstractive summarization, reranking, and other LLM-based operations
while maintaining 100% local execution and data privacy.
"""

import asyncio
import json
from typing import List, Dict, Optional, Union, Any, AsyncIterator
from dataclasses import dataclass
from enum import Enum
import aiohttp
import structlog
from pydantic import BaseModel, Field, ConfigDict
import backoff
from contextlib import asynccontextmanager

from ..utils.config import settings
from ..utils.circuit_breaker import CircuitBreaker

logger = structlog.get_logger(__name__)


class CompletionMode(str, Enum):
    """Completion request modes."""
    CHAT = "chat"
    COMPLETION = "completion"
    STREAMING = "streaming"


class Role(str, Enum):
    """Chat message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ChatMessage:
    """Chat message structure."""
    role: Role
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary for API calls."""
        return {"role": self.role.value, "content": self.content}


class CompletionRequest(BaseModel):
    """vLLM completion request structure."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    model: str
    messages: Optional[List[Dict[str, str]]] = None
    prompt: Optional[str] = None
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logprobs: Optional[int] = None
    echo: bool = False
    n: int = Field(default=1, ge=1, le=10)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API request."""
        data = self.model_dump(exclude_none=True)
        # vLLM expects either messages or prompt, not both
        if self.messages:
            data.pop("prompt", None)
        else:
            data.pop("messages", None)
        return data


class CompletionResponse(BaseModel):
    """vLLM completion response structure."""
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Optional[Dict[str, int]] = None
    
    def get_text(self, choice_idx: int = 0) -> str:
        """Extract text from completion response."""
        if not self.choices or choice_idx >= len(self.choices):
            return ""
        
        choice = self.choices[choice_idx]
        # Handle both chat and completion formats
        if "message" in choice:
            return choice["message"].get("content", "")
        elif "text" in choice:
            return choice["text"]
        return ""
    
    def get_all_texts(self) -> List[str]:
        """Extract all generated texts."""
        texts = []
        for i in range(len(self.choices)):
            text = self.get_text(i)
            if text:
                texts.append(text)
        return texts


class VLLMClient:
    """
    Client for interacting with locally hosted vLLM servers.
    
    Features:
    - Async/await support for all operations
    - Automatic retry with exponential backoff
    - Circuit breaker for fault tolerance
    - Streaming response support
    - Multiple completion modes (chat, completion, streaming)
    - Comprehensive error handling and logging
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 120,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: int = 60
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server base URL (defaults to config)
            model_name: Default model name (defaults to config)
            api_key: API key if required (defaults to config)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            circuit_breaker_threshold: Failures before circuit opens
            circuit_breaker_timeout: Circuit breaker recovery timeout
        """
        # Load from config with fallbacks
        config = settings.rag.reranking.vllm
        self.base_url = base_url or config.base_url.rstrip("/")
        self.model_name = model_name or config.model_name
        self.api_key = api_key or config.api_key
        
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        
        # Initialize circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=circuit_breaker_timeout,
            expected_exception=aiohttp.ClientError
        )
        
        # Session management
        self._session: Optional[aiohttp.ClientSession] = None
        
        logger.info(
            "Initialized vLLM client",
            base_url=self.base_url,
            model=self.model_name,
            timeout=timeout
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            headers = {"Content-Type": "application/json"}
            if self.api_key and self.api_key != "dummy-key":
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=headers
            )
    
    async def close(self):
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def _make_request(
        self,
        endpoint: str,
        data: Dict[str, Any],
        stream: bool = False
    ) -> Union[Dict[str, Any], AsyncIterator[str]]:
        """
        Make HTTP request to vLLM server with retry logic.
        
        Args:
            endpoint: API endpoint
            data: Request data
            stream: Whether to stream response
            
        Returns:
            Response data or streaming iterator
        """
        await self._ensure_session()
        
        url = f"{self.base_url}/{endpoint}"
        
        async with self.circuit_breaker:
            try:
                if stream:
                    return self._stream_request(url, data)
                else:
                    async with self._session.post(url, json=data) as response:
                        response.raise_for_status()
                        return await response.json()
                        
            except aiohttp.ClientError as e:
                logger.error(
                    "vLLM request failed",
                    url=url,
                    error=str(e),
                    status=getattr(e, "status", None)
                )
                raise
    
    async def _stream_request(
        self,
        url: str,
        data: Dict[str, Any]
    ) -> AsyncIterator[str]:
        """
        Stream response from vLLM server.
        
        Args:
            url: Request URL
            data: Request data
            
        Yields:
            Response chunks
        """
        data["stream"] = True
        
        async with self._session.post(url, json=data) as response:
            response.raise_for_status()
            
            async for line in response.content:
                line = line.decode("utf-8").strip()
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    
                    try:
                        chunk_data = json.loads(chunk)
                        if chunk_data.get("choices"):
                            choice = chunk_data["choices"][0]
                            if "delta" in choice and "content" in choice["delta"]:
                                yield choice["delta"]["content"]
                            elif "text" in choice:
                                yield choice["text"]
                    except json.JSONDecodeError:
                        logger.warning("Failed to parse streaming chunk", chunk=chunk)
    
    async def complete(
        self,
        prompt: Union[str, List[ChatMessage]],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        stop: Optional[List[str]] = None,
        stream: bool = False,
        model: Optional[str] = None,
        **kwargs
    ) -> Union[CompletionResponse, AsyncIterator[str]]:
        """
        Generate completion from vLLM.
        
        Args:
            prompt: Text prompt or chat messages
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            stop: Stop sequences
            stream: Whether to stream response
            model: Model to use (defaults to configured)
            **kwargs: Additional parameters
            
        Returns:
            Completion response or streaming iterator
        """
        # Prepare request
        request = CompletionRequest(
            model=model or self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop=stop,
            stream=stream,
            **kwargs
        )
        
        # Handle prompt types
        if isinstance(prompt, str):
            request.prompt = prompt
            endpoint = "completions"
        elif isinstance(prompt, list):
            request.messages = [msg.to_dict() for msg in prompt]
            endpoint = "chat/completions"
        else:
            raise ValueError(f"Invalid prompt type: {type(prompt)}")
        
        # Make request
        result = await self._make_request(
            endpoint=endpoint,
            data=request.to_dict(),
            stream=stream
        )
        
        if stream:
            return result
        else:
            return CompletionResponse(**result)
    
    async def chat(
        self,
        messages: List[ChatMessage],
        **kwargs
    ) -> CompletionResponse:
        """
        Chat completion convenience method.
        
        Args:
            messages: Chat messages
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        return await self.complete(prompt=messages, **kwargs)
    
    async def embed(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None
    ) -> List[List[float]]:
        """
        Generate embeddings using vLLM.
        
        Args:
            texts: Text(s) to embed
            model: Model to use
            
        Returns:
            Embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        data = {
            "model": model or self.model_name,
            "input": texts
        }
        
        result = await self._make_request("embeddings", data)
        
        # Extract embeddings
        embeddings = []
        for item in result.get("data", []):
            embeddings.append(item["embedding"])
        
        return embeddings
    
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = True
    ) -> List[Tuple[int, float, str]]:
        """
        Rerank documents using vLLM.
        
        Args:
            query: Search query
            documents: Documents to rerank
            top_k: Number of top documents to return
            return_scores: Whether to return scores
            
        Returns:
            List of (index, score, document) tuples
        """
        # Create reranking prompt
        system_prompt = """You are a relevance scoring expert. For each document, 
        provide a relevance score from 0 to 100 based on how well it answers the query.
        Return only the numeric scores, one per line."""
        
        user_prompt = f"Query: {query}\n\nDocuments:\n"
        for i, doc in enumerate(documents):
            user_prompt += f"\n{i+1}. {doc[:500]}..."  # Truncate long docs
        
        messages = [
            ChatMessage(Role.SYSTEM, system_prompt),
            ChatMessage(Role.USER, user_prompt)
        ]
        
        # Get scores
        response = await self.chat(messages, temperature=0.1, max_tokens=100)
        scores_text = response.get_text()
        
        # Parse scores
        scores = []
        for line in scores_text.strip().split("\n"):
            try:
                score = float(line.strip())
                scores.append(score)
            except ValueError:
                scores.append(0.0)
        
        # Ensure we have scores for all documents
        while len(scores) < len(documents):
            scores.append(0.0)
        
        # Create ranked results
        results = [(i, score/100.0, doc) for i, (score, doc) in enumerate(zip(scores, documents))]
        results.sort(key=lambda x: x[1], reverse=True)
        
        if top_k:
            results = results[:top_k]
        
        return results
    
    async def health_check(self) -> bool:
        """
        Check if vLLM server is healthy.
        
        Returns:
            True if server is healthy
        """
        try:
            await self._ensure_session()
            async with self._session.get(f"{self.base_url}/health") as response:
                return response.status == 200
        except Exception as e:
            logger.warning("vLLM health check failed", error=str(e))
            return False
    
    async def list_models(self) -> List[str]:
        """
        List available models on vLLM server.
        
        Returns:
            List of model names
        """
        try:
            result = await self._make_request("models", {})
            return [model["id"] for model in result.get("data", [])]
        except Exception as e:
            logger.error("Failed to list vLLM models", error=str(e))
            return []
    
    # Pydantic AI compatibility methods
    
    async def __call__(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Pydantic AI compatibility - direct call interface.
        
        Args:
            messages: Chat messages in dict format
            **kwargs: Additional parameters
            
        Returns:
            Generated text
        """
        chat_messages = [
            ChatMessage(Role(msg["role"]), msg["content"]) 
            for msg in messages
        ]
        
        response = await self.chat(chat_messages, **kwargs)
        return response.get_text()
    
    def create_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for Pydantic AI compatibility.
        
        Args:
            messages: Chat messages
            **kwargs: Additional parameters
            
        Returns:
            Completion response
        """
        # Run async method in sync context
        loop = asyncio.new_event_loop()
        try:
            chat_messages = [
                ChatMessage(Role(msg["role"]), msg["content"]) 
                for msg in messages
            ]
            response = loop.run_until_complete(self.chat(chat_messages, **kwargs))
            return response.model_dump()
        finally:
            loop.close()


# Convenience function for creating client
@asynccontextmanager
async def create_vllm_client(**kwargs) -> VLLMClient:
    """
    Create vLLM client as async context manager.
    
    Args:
        **kwargs: Client configuration
        
    Yields:
        VLLMClient instance
    """
    client = VLLMClient(**kwargs)
    try:
        await client._ensure_session()
        yield client
    finally:
        await client.close()


# Example usage
async def example_usage():
    """Example of using VLLMClient."""
    async with create_vllm_client() as client:
        # Simple completion
        response = await client.complete(
            "Summarize the key points of machine learning",
            max_tokens=200
        )
        print(f"Summary: {response.get_text()}")
        
        # Chat completion
        messages = [
            ChatMessage(Role.SYSTEM, "You are a helpful assistant."),
            ChatMessage(Role.USER, "What is the capital of France?")
        ]
        chat_response = await client.chat(messages)
        print(f"Answer: {chat_response.get_text()}")
        
        # Streaming
        print("Streaming response:")
        async for chunk in await client.complete(
            "Write a short story about AI",
            stream=True,
            max_tokens=500
        ):
            print(chunk, end="", flush=True)
        print()
        
        # Reranking
        documents = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "Berlin is the capital of Germany."
        ]
        ranked = await client.rerank("What is the capital of France?", documents)
        for idx, score, doc in ranked:
            print(f"  [{idx}] Score: {score:.2f} - {doc}")


if __name__ == "__main__":
    asyncio.run(example_usage())