"""
Vapi Python SDK - Main Module

This module provides the Vapi class for initiating and managing
voice calls with Vapi AI assistants.

Example:
    >>> from vapi_python import Vapi
    >>> vapi = Vapi(api_key='your-public-key')
    >>> vapi.start(assistant_id='your-assistant-id')
    >>> # ... interact with the assistant ...
    >>> vapi.stop()
"""

from daily import *
import requests
from typing import Optional, Dict, Any, Union
from .daily_call import DailyCall
from .types import (
    MessageRole,
    Message,
    MessageType,
    validate_role,
    VALID_MESSAGE_ROLES,
    supports_developer_role,
)

SAMPLE_RATE = 16000
CHANNELS = 1


def create_web_call(api_url: str, api_key: str, payload: Dict[str, Any]) -> tuple:
    """
    Create a web call via the Vapi API.

    Args:
        api_url: The base URL of the Vapi API.
        api_key: The API key for authentication.
        payload: The request payload containing call configuration.

    Returns:
        A tuple of (call_id, web_call_url).

    Raises:
        Exception: If the API call fails.
    """
    url = f"{api_url}/call/web"
    headers = {
        'Authorization': 'Bearer ' + api_key,
        'Content-Type': 'application/json'
    }
    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    if response.status_code == 201:
        call_id = data.get('id')
        web_call_url = data.get('webCallUrl')
        return call_id, web_call_url
    else:
        raise Exception(f"Error: {data['message']}")


class Vapi:
    """
    Vapi client for managing voice AI calls.

    This class provides methods to start, manage, and stop voice calls
    with Vapi AI assistants. It supports both predefined assistants
    (via assistant_id) and inline assistant configurations.

    Attributes:
        api_key: The Vapi API key for authentication.
        api_url: The base URL of the Vapi API.

    Example:
        Basic usage with an assistant ID:

        >>> vapi = Vapi(api_key='your-public-key')
        >>> vapi.start(assistant_id='your-assistant-id')
        >>> vapi.stop()

        Using an inline assistant configuration:

        >>> vapi = Vapi(api_key='your-public-key')
        >>> assistant = {
        ...     'firstMessage': 'Hello! How can I help you today?',
        ...     'model': 'gpt-5.2',
        ...     'voice': 'jennifer-playht'
        ... }
        >>> vapi.start(assistant=assistant)

        Sending messages during a call:

        >>> vapi.add_message('user', 'What is the weather?')
        >>> # For GPT-5.x models, you can use the developer role:
        >>> vapi.add_message('developer', 'Respond briefly.')
    """

    def __init__(self, *, api_key: str, api_url: str = "https://api.vapi.ai"):
        """
        Initialize the Vapi client.

        Args:
            api_key: Your Vapi public API key.
            api_url: The base URL for the Vapi API.
                Defaults to 'https://api.vapi.ai'.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.__client: Optional[DailyCall] = None

    def start(
        self,
        *,
        assistant_id: Optional[str] = None,
        assistant: Optional[Dict[str, Any]] = None,
        assistant_overrides: Optional[Dict[str, Any]] = None,
        squad_id: Optional[str] = None,
        squad: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Start a new voice call with a Vapi assistant.

        You must provide exactly one of: assistant_id, assistant,
        squad_id, or squad.

        Args:
            assistant_id: The ID of a predefined assistant to use.
            assistant: An inline assistant configuration dictionary.
                See https://docs.vapi.ai/api-reference/assistants/create-assistant
                for available options.
            assistant_overrides: Optional dictionary of assistant
                parameters to override. Can include 'variableValues'
                for template variables.
            squad_id: The ID of a predefined squad to use.
            squad: An inline squad configuration dictionary.

        Raises:
            Exception: If no assistant is specified or if the call
                cannot be created.

        Example:
            Using assistant_id with overrides:

            >>> vapi.start(
            ...     assistant_id='your-assistant-id',
            ...     assistant_overrides={
            ...         'variableValues': {'name': 'John'}
            ...     }
            ... )
        """
        # Start a new call
        if assistant_id:
            payload = {
                'assistantId': assistant_id,
                'assistantOverrides': assistant_overrides
            }
        elif assistant:
            payload = {
                'assistant': assistant,
                'assistantOverrides': assistant_overrides
            }
        elif squad_id:
            payload = {'squadId': squad_id}
        elif squad:
            payload = {'squad': squad}
        else:
            raise Exception("Error: No assistant specified.")

        call_id, web_call_url = create_web_call(
            self.api_url, self.api_key, payload)

        if not web_call_url:
            raise Exception("Error: Unable to create call.")

        print('Joining call... ' + call_id)

        self.__client = DailyCall()
        self.__client.join(web_call_url)

    def stop(self) -> None:
        """
        Stop the current call and clean up resources.

        This method leaves the call and releases all associated
        resources. After calling stop(), you can start a new call
        with the start() method.
        """
        if self.__client:
            self.__client.leave()
            self.__client = None

    def send(self, message: Dict[str, Any]) -> None:
        """
        Send a generic message to the assistant.

        This is a low-level method for sending arbitrary messages.
        For common operations, consider using the higher-level methods
        like add_message() instead.

        Args:
            message: A dictionary containing the message type and content.
                Must include a 'type' key.

        Raises:
            Exception: If the call has not been started.
            ValueError: If the message format is invalid.

        Example:
            >>> vapi.send({
            ...     'type': 'add-message',
            ...     'message': {'role': 'user', 'content': 'Hello!'}
            ... })
        """
        if not self.__client:
            raise Exception("Call not started. Please start the call first.")

        # Check message format here instead of serialization
        if not isinstance(message, dict) or 'type' not in message:
            raise ValueError("Invalid message format.")

        try:
            self.__client.send_app_message(message)  # Send dictionary directly
        except Exception as e:
            print(f"Failed to send message: {e}")

    def add_message(
        self,
        role: Union[str, MessageRole],
        content: str,
        *,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
    ) -> None:
        """
        Send a message to the assistant during the call.

        This method adds a message to the conversation. The role
        determines how the message is interpreted by the model.

        Args:
            role: The role of the message sender. Valid values:
                - 'system': System-level instructions
                - 'user': User messages
                - 'assistant': Assistant responses (for context)
                - 'developer': Developer instructions (GPT-5.x, o-series)
                - 'tool': Tool/function call results
                - 'function': DEPRECATED - use 'tool' instead

            content: The text content of the message.

            name: Optional name for tool/function messages.

            tool_call_id: Required for 'tool' role messages.
                The ID of the tool call this message responds to.

        Raises:
            Exception: If the call has not been started.
            ValueError: If the role is invalid or required fields missing.

        Note:
            The 'developer' role is supported by GPT-5.x and o-series
            models. It provides elevated instruction priority over the
            'system' role. When using older models, use 'system' instead.

        Warning:
            The 'function' role is deprecated. Use 'tool' role instead.
            The 'function' role may be removed in a future version.

        Example:
            Basic user message:

            >>> vapi.add_message('user', 'What is 2 + 2?')

            Developer instruction (GPT-5.x/o-series models):

            >>> vapi.add_message(
            ...     'developer',
            ...     'Respond in a formal tone.'
            ... )

            Tool response:

            >>> vapi.add_message(
            ...     'tool',
            ...     '{"temperature": 72, "unit": "F"}',
            ...     tool_call_id='call_abc123'
            ... )
        """
        # Validate role (this also handles deprecation warning for 'function')
        if isinstance(role, MessageRole):
            validated_role = role.value
        else:
            validated_role = validate_role(role)

        # Build the message
        message_content: Dict[str, Any] = {
            'role': validated_role,
            'content': content
        }

        if name is not None:
            message_content['name'] = name

        if tool_call_id is not None:
            message_content['tool_call_id'] = tool_call_id

        # Validate tool messages have required fields
        if validated_role == MessageRole.TOOL.value and not tool_call_id:
            raise ValueError(
                "Messages with 'tool' role require 'tool_call_id' parameter."
            )

        message = {
            'type': MessageType.ADD_MESSAGE.value,
            'message': message_content
        }
        self.send(message)

    def say(self, text: str, *, end_call_after: bool = False) -> None:
        """
        Make the assistant say a specific message.

        This method instructs the assistant to speak the provided
        text immediately, interrupting any current speech.

        Args:
            text: The text for the assistant to speak.
            end_call_after: If True, the call will end after the
                assistant finishes speaking. Defaults to False.

        Raises:
            Exception: If the call has not been started.

        Example:
            >>> vapi.say("Please hold while I look that up.")
            >>> vapi.say("Goodbye!", end_call_after=True)
        """
        message: Dict[str, Any] = {
            'type': MessageType.SAY.value,
            'content': text,
        }
        if end_call_after:
            message['endCallAfter'] = True

        self.send(message)

    def end_call(self) -> None:
        """
        Request the assistant to end the call gracefully.

        This sends an end-call request to the assistant, which
        will typically result in the assistant saying a farewell
        message before ending the call.

        Raises:
            Exception: If the call has not been started.
        """
        message = {'type': MessageType.END_CALL.value}
        self.send(message)
