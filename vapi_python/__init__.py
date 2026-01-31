"""
Vapi Python SDK

A Python SDK for integrating Vapi AI voice assistants into your applications.

This package provides a simple interface to start voice calls with Vapi
AI assistants, send messages, and manage call state.

Basic Usage:
    >>> from vapi_python import Vapi
    >>> vapi = Vapi(api_key='your-public-key')
    >>> vapi.start(assistant_id='your-assistant-id')
    >>> vapi.stop()

For GPT-5.x and o-series models, you can use the developer role:
    >>> vapi.add_message('developer', 'Be concise in your responses.')

Type Definitions:
    - MessageRole: Enum of valid message roles
    - Message: Class for creating structured messages
    - OpenAIModel: Enum of supported OpenAI models
"""

from .vapi_python import Vapi
from .types import (
    MessageRole,
    Message,
    MessageType,
    OpenAIModel,
    VALID_MESSAGE_ROLES,
    DEVELOPER_ROLE_MODELS,
    validate_role,
    supports_developer_role,
)

__author__ = """Vapi AI"""
__email__ = 'team@vapi.ai'
__version__ = '0.2.0'

__all__ = [
    # Main class
    'Vapi',
    # Type definitions
    'MessageRole',
    'Message',
    'MessageType',
    'OpenAIModel',
    # Constants
    'VALID_MESSAGE_ROLES',
    'DEVELOPER_ROLE_MODELS',
    # Utility functions
    'validate_role',
    'supports_developer_role',
]
