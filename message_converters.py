
"""
Streamlined message conversion utilities for MLflow ResponsesAgent integration.

This module provides minimal, efficient message normalization focusing on:
1. Checkpoint deserialization handling (PostgreSQL state management)
2. Content structure normalization (list/dict to string conversion)
3. LangChain <-> ResponsesAgent interoperability

NOTE: Modern databricks-langchain and MLflow 3.x handle most conversions natively.
This utility focuses only on edge cases and checkpoint-specific issues.
"""

import logging
from typing import Any, Dict, List, Union

 

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)


class MessageNormalizer:
    """
    Lightweight message normalization for checkpoint deserialization.
    
    Primary purpose: Handle complex content structures that emerge from
    PostgreSQL checkpoint serialization/deserialization cycles.
    """
    
    @staticmethod
    def normalize_content(content: Any) -> str:
        """
        Normalize message content to plain string format.
        
        PostgreSQL checkpoints can serialize content as:
        - Plain strings (ideal case)
        - Lists of dicts: [{"type": "text", "text": "..."}]
        - Complex nested structures from tool outputs
        
        This ensures all content becomes a simple string suitable for LLM input.
        
        Args:
            content: Message content (string, list, dict, or other)
            
        Returns:
            Normalized string content
            
        Examples:
            >>> normalize_content("Hello")
            "Hello"
            
            >>> normalize_content([{"type": "text", "text": "Hello"}])
            "Hello"
            
            >>> normalize_content([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}])
            "A B"
        """
        # Fast path: already a string
        if isinstance(content, str):
            return content
        
        # Handle None/empty
        if not content:
            return ""
        
        # Handle list content (common from checkpoints)
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    # Priority: 'text' field (most common)
                    if "text" in item:
                        parts.append(str(item["text"]))
                    # Fallback: 'content' field
                    elif "content" in item:
                        parts.append(str(item["content"]))
                    # Last resort: stringify entire dict
                    else:
                        parts.append(str(item))
                elif isinstance(item, str):
                    parts.append(item)
                else:
                    parts.append(str(item))
            
            return " ".join(parts).strip()
        
        # Handle dict content
        if isinstance(content, dict):
            if "text" in content:
                return str(content["text"])
            elif "content" in content:
                return str(content["content"])
            else:
                return str(content)
        
        # Ultimate fallback
        return str(content)
    
    @staticmethod
    def normalize_message(msg: Union[BaseMessage, Dict]) -> BaseMessage:
        """
        Normalize a single message to proper LangChain BaseMessage format.
        
        Handles:
        - BaseMessage objects with complex content structures
        - Dict representations of messages
        - Content normalization for checkpoint compatibility
        
        Args:
            msg: Message as BaseMessage or dict
            
        Returns:
            Properly formatted BaseMessage with normalized content
        """
        # Handle BaseMessage objects
        if isinstance(msg, BaseMessage):
            # Normalize content structure
            normalized_content = MessageNormalizer.normalize_content(msg.content)
            
            # Recreate message with normalized content
            if msg.type == "human":
                return HumanMessage(content=normalized_content)
            elif msg.type == "ai":
                return AIMessage(
                    content=normalized_content,
                    tool_calls=getattr(msg, 'tool_calls', [])
                )
            elif msg.type == "system":
                return SystemMessage(content=normalized_content)
            elif msg.type == "tool":
                return ToolMessage(
                    content=normalized_content,
                    tool_call_id=getattr(msg, 'tool_call_id', '')
                )
            else:
                # Unknown type, default to AIMessage
                logger.warning(f"Unknown message type: {msg.type}, defaulting to AIMessage")
                return AIMessage(content=normalized_content)
        
        # Handle dict representations
        elif isinstance(msg, dict):
            role = msg.get("role", msg.get("type", "user"))
            content = MessageNormalizer.normalize_content(msg.get("content", ""))
            
            if role in ("user", "human"):
                return HumanMessage(content=content)
            elif role in ("assistant", "ai"):
                return AIMessage(
                    content=content,
                    tool_calls=msg.get("tool_calls", [])
                )
            elif role == "system":
                return SystemMessage(content=content)
            elif role == "tool":
                return ToolMessage(
                    content=content,
                    tool_call_id=msg.get("tool_call_id", "")
                )
            else:
                logger.warning(f"Unknown role: {role}, defaulting to HumanMessage")
                return HumanMessage(content=content)
        
        # Fallback for unexpected types
        else:
            logger.warning(f"Unexpected message type: {type(msg)}, converting to string")
            return HumanMessage(content=str(msg))
    
    @staticmethod
    def normalize_messages(messages: List[Union[BaseMessage, Dict]]) -> List[BaseMessage]:
        """
        Normalize a list of messages for LLM consumption.
        
        Critical for:
        - Loading conversation history from PostgreSQL checkpoints
        - Ensuring content compatibility with LLM providers (OpenAI, etc.)
        - Handling mixed message formats
        
        Args:
            messages: List of messages (BaseMessage objects or dicts)
            
        Returns:
            List of properly formatted BaseMessage objects
        """
        return [MessageNormalizer.normalize_message(msg) for msg in messages]
    
    @staticmethod
    def filter_empty_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Remove messages with empty content (except those with tool_calls or tool responses).
        
        Prevents errors like: "Invalid 'messages[x].content': string expected, got empty list"
        
        Args:
            messages: List of BaseMessage objects
            
        Returns:
            Filtered list with only valid messages
        """
        filtered = []
        for msg in messages:
            # Always include system messages
            if isinstance(msg, SystemMessage):
                filtered.append(msg)
            # AI messages are valid if they have content OR tool_calls
            elif isinstance(msg, AIMessage):
                if msg.content or msg.tool_calls:
                    filtered.append(msg)
            # Tool messages are valid if they have content
            elif isinstance(msg, ToolMessage):
                if msg.content:
                    filtered.append(msg)
            # Other message types need non-empty content
            elif hasattr(msg, 'content') and msg.content:
                filtered.append(msg)
        
        return filtered


class ResponsesConverter:
    """
    Utilities for converting between MLflow ResponsesAgent format and LangChain.
    
    NOTE: MLflow 3.x and databricks-langchain handle most conversions automatically.
    These are helper methods for explicit conversion when needed.
    """
    
    @staticmethod
    def responses_to_langchain(input_items: List[Any]) -> List[BaseMessage]:
        """
        Convert MLflow Responses API input items to LangChain messages.
        
        Args:
            input_items: List of ResponsesAgentRequestItem objects
            
        Returns:
            List of LangChain BaseMessage objects
        """
        messages = []
        
        for item in input_items:
            try:
                # Convert to dict
                if hasattr(item, 'model_dump'):
                    item_dict = item.model_dump()
                elif hasattr(item, 'dict'):
                    item_dict = item.dict()
                elif isinstance(item, dict):
                    item_dict = item
                else:
                    # Fallback: treat as user message
                    messages.append(HumanMessage(content=str(item)))
                    continue
                
                role = item_dict.get("role", "user")
                content = item_dict.get("content", "")
                
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(
                        content=content,
                        tool_calls=item_dict.get("tool_calls", [])
                    ))
                elif role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "tool":
                    messages.append(ToolMessage(
                        content=content,
                        tool_call_id=item_dict.get("tool_call_id", "")
                    ))
                else:
                    # Unknown role, default to user message
                    messages.append(HumanMessage(content=content))
                    
            except Exception as e:
                logger.error(f"Error converting input item: {e}")
                # Fallback: create user message
                messages.append(HumanMessage(content=str(item)))
        
        return messages
    
    @staticmethod
    def langchain_to_responses_dict(message: BaseMessage) -> Dict[str, Any]:
        """
        Convert a LangChain message to ResponsesAgent dict format.
        
        Useful for manual response construction or debugging.
        
        Args:
            message: LangChain BaseMessage
            
        Returns:
            Dict in ResponsesAgent format
        """
        message_dict = message.model_dump()
        role = message_dict["type"]
        
        result = {
            "role": role,
            "content": MessageNormalizer.normalize_content(message_dict.get("content", ""))
        }
        
        # Add tool-specific fields
        if role == "ai" and message_dict.get("tool_calls"):
            result["tool_calls"] = message_dict["tool_calls"]
        elif role == "tool":
            result["tool_call_id"] = message_dict.get("tool_call_id", "")
        
        return result



def normalize_checkpoint_messages(messages: List[Union[BaseMessage, Dict]]) -> List[BaseMessage]:
    """
    One-stop normalization for messages loaded from checkpoints.
    
    Applies both content normalization and empty message filtering.
    Use this when loading conversation history from PostgreSQL state.
    
    Args:
        messages: Raw messages from checkpoint
        
    Returns:
        Clean, normalized messages ready for LLM
    """
    normalized = MessageNormalizer.normalize_messages(messages)
    return MessageNormalizer.filter_empty_messages(normalized)
