"""
Vapi Python SDK Type Definitions

This module defines the message types and roles used in Vapi conversations,
aligned with the latest OpenAI API specification.
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Union
import warnings


class MessageRole(str, Enum):
    """
    Supported message roles for conversation messages.

    Aligned with OpenAI API specification (May 2025).

    Attributes:
        SYSTEM: System-level instructions for the assistant.
        USER: Messages from the user/human participant.
        ASSISTANT: Messages from the AI assistant.
        DEVELOPER: Developer-level instructions with elevated priority.
            Required for GPT-5.x and o-series models. Takes precedence
            over system messages when both are present.
        TOOL: Results from tool/function calls.
        FUNCTION: DEPRECATED. Results from function calls.
            Use TOOL role instead for new implementations.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"
    TOOL = "tool"
    FUNCTION = "function"  # Deprecated - use TOOL instead


# Valid roles for add_message method
VALID_MESSAGE_ROLES = frozenset({
    MessageRole.SYSTEM.value,
    MessageRole.USER.value,
    MessageRole.ASSISTANT.value,
    MessageRole.DEVELOPER.value,
    MessageRole.TOOL.value,
    MessageRole.FUNCTION.value,
})


def validate_role(role: str) -> str:
    """
    Validate and normalize a message role.

    Args:
        role: The role string to validate.

    Returns:
        The normalized role string (lowercase).

    Raises:
        ValueError: If the role is not a valid message role.

    Warns:
        DeprecationWarning: If the 'function' role is used.

    Example:
        >>> validate_role("user")
        'user'
        >>> validate_role("DEVELOPER")
        'developer'
        >>> validate_role("function")  # Warns about deprecation
        'function'
    """
    normalized_role = role.lower()

    if normalized_role not in VALID_MESSAGE_ROLES:
        raise ValueError(
            f"Invalid role '{role}'. Valid roles are: "
            f"{', '.join(sorted(VALID_MESSAGE_ROLES))}"
        )

    if normalized_role == MessageRole.FUNCTION.value:
        warnings.warn(
            "The 'function' role is deprecated. Use 'tool' role instead. "
            "The 'function' role may be removed in a future version.",
            DeprecationWarning,
            stacklevel=3
        )

    return normalized_role


class MessageType(str, Enum):
    """
    Supported message types for Vapi communication.

    These types are used in the 'type' field of messages sent
    via the send() method.
    """
    ADD_MESSAGE = "add-message"
    SAY = "say"
    END_CALL = "end-call"
    TRANSFER_CALL = "transfer-call"


class Message:
    """
    Represents a conversation message.

    This class provides a structured way to create messages
    for Vapi conversations with proper type validation.

    Attributes:
        role: The role of the message sender (see MessageRole).
        content: The text content of the message.
        name: Optional name for tool/function messages.
        tool_call_id: Optional ID for tool response messages.

    Example:
        >>> msg = Message(role="user", content="Hello!")
        >>> msg.to_dict()
        {'role': 'user', 'content': 'Hello!'}

        >>> msg = Message(role="developer", content="Be concise.")
        >>> msg.to_dict()
        {'role': 'developer', 'content': 'Be concise.'}
    """

    def __init__(
        self,
        role: Union[str, MessageRole],
        content: str,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ):
        """
        Initialize a Message instance.

        Args:
            role: The role of the message sender. Can be a string or
                MessageRole enum value. Valid roles: 'system', 'user',
                'assistant', 'developer', 'tool', 'function'.
            content: The text content of the message.
            name: Optional name identifier for tool/function messages.
            tool_call_id: Optional ID linking to a specific tool call.
                Required when role is 'tool'.

        Raises:
            ValueError: If role is invalid or required fields are missing.
        """
        if isinstance(role, MessageRole):
            role = role.value

        self.role = validate_role(role)
        self.content = content
        self.name = name
        self.tool_call_id = tool_call_id

        # Validate tool messages have required fields
        if self.role == MessageRole.TOOL.value and not tool_call_id:
            raise ValueError(
                "Messages with 'tool' role require 'tool_call_id' parameter."
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the message to a dictionary format.

        Returns:
            Dictionary representation of the message suitable
            for sending via the Vapi API.
        """
        result: Dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }

        if self.name is not None:
            result["name"] = self.name

        if self.tool_call_id is not None:
            result["tool_call_id"] = self.tool_call_id

        return result


# OpenAI Model Constants
# These are the latest models available as of May 2025

class OpenAIModel(str, Enum):
    """
    OpenAI model identifiers.

    These constants represent available OpenAI models that can be
    used with Vapi assistants.

    Note:
        Models prefixed with 'gpt-5' and 'o' series models support
        the 'developer' role for enhanced instruction following.
    """
    # GPT-4 Series
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4O = "gpt-4o"
    GPT_4O_MINI = "gpt-4o-mini"

    # GPT-4.1 Series (2024-2025)
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"

    # GPT-5 Series (Latest - 2025)
    GPT_5_2 = "gpt-5.2"
    GPT_5_2_CHAT = "gpt-5.2-chat"
    GPT_5_2_CHAT_LATEST = "gpt-5.2-chat-latest"
    GPT_5_2_CODEX = "gpt-5.2-codex"

    # O-Series Reasoning Models
    O1 = "o1"
    O1_MINI = "o1-mini"
    O1_PREVIEW = "o1-preview"
    O3 = "o3"
    O3_MINI = "o3-mini"
    O4_MINI = "o4-mini"

    # Legacy Models (for backwards compatibility)
    GPT_3_5_TURBO = "gpt-3.5-turbo"


# Models that support the developer role
DEVELOPER_ROLE_MODELS = frozenset({
    OpenAIModel.GPT_5_2.value,
    OpenAIModel.GPT_5_2_CHAT.value,
    OpenAIModel.GPT_5_2_CHAT_LATEST.value,
    OpenAIModel.GPT_5_2_CODEX.value,
    OpenAIModel.O1.value,
    OpenAIModel.O1_MINI.value,
    OpenAIModel.O1_PREVIEW.value,
    OpenAIModel.O3.value,
    OpenAIModel.O3_MINI.value,
    OpenAIModel.O4_MINI.value,
})


def supports_developer_role(model: str) -> bool:
    """
    Check if a model supports the developer role.

    The developer role is supported by GPT-5.x and o-series models.
    When using these models, the developer role provides elevated
    instruction priority over the system role.

    Args:
        model: The model identifier string.

    Returns:
        True if the model supports the developer role.

    Example:
        >>> supports_developer_role("gpt-5.2")
        True
        >>> supports_developer_role("o3-mini")
        True
        >>> supports_developer_role("gpt-4o")
        False
    """
    return model in DEVELOPER_ROLE_MODELS
