"""
Navigation LLM Model - Llama 4 based travel assistant.

This module provides the core ML model for context-aware navigation:
- Llama 4 (Scout/Maverick) as the base LLM via llama-cpp-python
- Tool/function calling for maps, POI, routes
- Multilingual support (EN/KO/VI)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, AsyncGenerator
from uuid import uuid4

logger = logging.getLogger(__name__)

# Check for llama-cpp-python availability
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    Llama = None
    LLAMA_CPP_AVAILABLE = False

# Check for httpx (for remote API)
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False


class SupportedLanguage(str, Enum):
    """Supported languages for the navigator."""
    ENGLISH = "en"
    KOREAN = "ko"
    VIETNAMESE = "vi"


@dataclass
class NavigationLLMConfig:
    """Configuration for Navigation LLM model."""
    
    # Model path (for local) or API URL (for remote)
    model_path: Optional[str] = None
    api_base_url: str = "http://localhost:8080/v1"
    api_key: Optional[str] = None
    
    # Model settings
    model_name: str = "llama-4-scout-instruct"
    chat_format: str = "chatml-function-calling"
    n_ctx: int = 8192
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    
    # Timeouts (for remote API)
    request_timeout: float = 60.0
    connect_timeout: float = 10.0
    
    # Tool calling
    enable_tools: bool = True
    max_tool_calls: int = 5


# =============================================================================
# Tool Definitions
# =============================================================================

NAVIGATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_nearby_places",
            "description": "Search for nearby POIs (restaurants, attractions, transit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {"type": "number", "description": "Latitude"},
                    "lon": {"type": "number", "description": "Longitude"},
                    "radius_m": {"type": "integer", "default": 1000},
                    "categories": {"type": "array", "items": {"type": "string"}},
                    "language": {"type": "string", "enum": ["en", "ko", "vi"]}
                },
                "required": ["lat", "lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_route",
            "description": "Get directions between two locations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_lat": {"type": "number"},
                    "start_lon": {"type": "number"},
                    "end_lat": {"type": "number"},
                    "end_lon": {"type": "number"},
                    "mode": {"type": "string", "enum": ["walking", "transit"], "default": "walking"}
                },
                "required": ["start_lat", "start_lon", "end_lat", "end_lon"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_place_details",
            "description": "Get details about a place (hours, contact, tips).",
            "parameters": {
                "type": "object",
                "properties": {
                    "place_id": {"type": "string"},
                    "language": {"type": "string", "enum": ["en", "ko", "vi"]}
                },
                "required": ["place_id"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "get_menu_item_info",
            "description": "Get dish info (ingredients, dietary info).",
            "parameters": {
                "type": "object",
                "properties": {
                    "dish_name": {"type": "string"},
                    "language": {"type": "string", "enum": ["en", "ko", "vi"]}
                },
                "required": ["dish_name"]
            }
        }
    }
]


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPTS = {
    "en": """You are a helpful travel assistant. You help travelers navigate, \
find places, understand local food, and learn cultural etiquette.

Use tools to get accurate info. Never make up addresses or hours. Be concise.""",

    "ko": """당신은 여행 도우미입니다. 여행자들이 길을 찾고, 장소를 발견하고, \
현지 음식을 이해하도록 돕습니다.

도구를 사용해 정확한 정보를 얻으세요. 주소나 영업시간을 지어내지 마세요.""",

    "vi": """Bạn là trợ lý du lịch. Bạn giúp du khách tìm đường, khám phá địa điểm, \
hiểu ẩm thực địa phương.

Sử dụng công cụ để có thông tin chính xác. Không bịa địa chỉ hoặc giờ mở cửa."""
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ToolCall:
    """A tool/function call from the LLM."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Result from executing a tool."""
    tool_call_id: str
    name: str
    result: Any
    success: bool = True
    error: Optional[str] = None


@dataclass
class LLMResponse:
    """Response from the LLM."""
    content: str
    tool_calls: list[ToolCall]
    finish_reason: str
    usage: dict[str, int]
    model: str
    response_id: str
    
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0
    
    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)


@dataclass
class ChatMessage:
    """A message in the conversation."""
    role: str
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMError(Exception):
    """Exception raised by the LLM."""
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


# =============================================================================
# Navigation LLM Model
# =============================================================================

class NavigationLLM:
    """
    Llama 4-based Navigation LLM with tool calling.
    
    Supports both local (llama-cpp-python) and remote (OpenAI-compatible API) modes.
    """
    
    def __init__(self, config: Optional[NavigationLLMConfig] = None):
        self.config = config or NavigationLLMConfig()
        self._llm: Optional[Llama] = None
        self._client: Optional[Any] = None  # httpx.AsyncClient
        self._is_local = self.config.model_path is not None
        self._is_ready = False
    
    def load(self) -> None:
        """Load the model (for local mode)."""
        if not self._is_local:
            logger.info("Remote mode - no model to load")
            self._is_ready = True
            return
        
        if not LLAMA_CPP_AVAILABLE:
            raise LLMError("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        
        logger.info(f"Loading model from {self.config.model_path}")
        self._llm = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_gpu_layers=self.config.n_gpu_layers,
            chat_format=self.config.chat_format,
            verbose=False
        )
        self._is_ready = True
        logger.info("Model loaded successfully")
    
    async def initialize(self) -> None:
        """Initialize for remote API mode."""
        if self._is_local:
            self.load()
            return
        
        if not HTTPX_AVAILABLE:
            raise LLMError("httpx not installed. Run: pip install httpx")
        
        timeout = httpx.Timeout(
            connect=self.config.connect_timeout,
            read=self.config.request_timeout,
            write=self.config.request_timeout,
            pool=10.0
        )
        
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=self.config.api_base_url,
            timeout=timeout,
            headers=headers
        )
        self._is_ready = True
        logger.info(f"Initialized remote client: {self.config.api_base_url}")
    
    async def close(self) -> None:
        """Close resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._llm = None
        self._is_ready = False
    
    def get_system_prompt(self, language: SupportedLanguage) -> str:
        """Get system prompt for language."""
        return SYSTEM_PROMPTS.get(language.value, SYSTEM_PROMPTS["en"])
    
    def chat_sync(
        self,
        user_message: str,
        language: SupportedLanguage = SupportedLanguage.ENGLISH,
        history: Optional[list[ChatMessage]] = None,
        location: Optional[tuple[float, float]] = None,
    ) -> LLMResponse:
        """
        Synchronous chat (for local model).
        """
        if not self._is_ready:
            self.load()
        
        if not self._is_local or self._llm is None:
            raise LLMError("Use chat() for remote API mode")
        
        messages = self._build_messages(user_message, language, history, location)
        
        response = self._llm.create_chat_completion(
            messages=messages,
            tools=NAVIGATION_TOOLS if self.config.enable_tools else None,
            tool_choice="auto" if self.config.enable_tools else None,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )
        
        return self._parse_response(response)
    
    async def chat(
        self,
        user_message: str,
        language: SupportedLanguage = SupportedLanguage.ENGLISH,
        history: Optional[list[ChatMessage]] = None,
        location: Optional[tuple[float, float]] = None,
    ) -> LLMResponse:
        """
        Async chat (for remote API or wrapped local).
        """
        if not self._is_ready:
            await self.initialize()
        
        # If local, run sync in thread pool
        if self._is_local:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.chat_sync(user_message, language, history, location)
            )
        
        # Remote API
        messages = self._build_messages(user_message, language, history, location)
        
        request_body = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        
        if self.config.enable_tools:
            request_body["tools"] = NAVIGATION_TOOLS
            request_body["tool_choice"] = "auto"
        
        response = await self._client.post("/chat/completions", json=request_body)
        response.raise_for_status()
        return self._parse_response(response.json())
    
    def _build_messages(
        self,
        user_message: str,
        language: SupportedLanguage,
        history: Optional[list[ChatMessage]],
        location: Optional[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Build messages array."""
        system = self.get_system_prompt(language)
        if location:
            system += f"\n\nUser location: ({location[0]:.6f}, {location[1]:.6f})"
        
        messages = [{"role": "system", "content": system}]
        
        if history:
            for msg in history:
                messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": user_message})
        return messages
    
    def _parse_response(self, response: dict[str, Any]) -> LLMResponse:
        """Parse LLM response."""
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        
        tool_calls = []
        for tc in message.get("tool_calls", []):
            try:
                args = json.loads(tc.get("function", {}).get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(ToolCall(
                id=tc.get("id", str(uuid4())),
                name=tc.get("function", {}).get("name", ""),
                arguments=args
            ))
        
        return LLMResponse(
            content=message.get("content", "") or "",
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason", "stop"),
            usage=response.get("usage", {}),
            model=response.get("model", self.config.model_name),
            response_id=response.get("id", str(uuid4()))
        )
    
    def health_check(self) -> bool:
        """Check if model is ready."""
        return self._is_ready

    async def chat_with_tools(
        self,
        user_message: str,
        language: SupportedLanguage = SupportedLanguage.ENGLISH,
        history: Optional[list[ChatMessage]] = None,
        location: Optional[tuple[float, float]] = None,
        max_tool_rounds: int = 3,
    ) -> LLMResponse:
        """
        Chat with automatic tool execution.
        
        This method will:
        1. Send the user message to the LLM
        2. If the LLM requests tool calls, execute them
        3. Feed results back to the LLM
        4. Repeat until the LLM provides a final response
        
        Args:
            user_message: User's query
            language: Response language
            history: Previous conversation
            location: User's current location (lat, lon)
            max_tool_rounds: Max iterations of tool calling
        
        Returns:
            Final LLM response after tool execution
        """
        # Import tools module
        from . import tools
        
        messages = self._build_messages(user_message, language, history, location)
        
        for round_num in range(max_tool_rounds):
            # Get LLM response
            response = await self.chat(
                user_message=user_message if round_num == 0 else "",
                language=language,
                history=history,
                location=location,
            )
            
            # If no tool calls, we're done
            if not response.has_tool_calls:
                return response
            
            # Execute each tool call
            tool_results = []
            for tc in response.tool_calls:
                logger.info(f"Executing tool: {tc.name}({tc.arguments})")
                
                result = await tools.execute_tool(tc.name, tc.arguments)
                
                tool_results.append(ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    result=result,
                    success="error" not in result,
                    error=result.get("error"),
                ))
            
            # Add tool call and results to history for next round
            if history is None:
                history = []
            
            # Add assistant's tool call message
            history.append(ChatMessage(
                role="assistant",
                content="",
                tool_calls=response.tool_calls,
            ))
            
            # Add tool results
            for tr in tool_results:
                history.append(ChatMessage(
                    role="tool",
                    content=json.dumps(tr.result),
                    tool_call_id=tr.tool_call_id,
                ))
        
        # Max rounds reached, return last response
        return response

