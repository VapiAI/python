"""
Tests for Vapi Python SDK main module.

This module tests the Vapi class and its methods for proper
message handling and OpenAI API specification compliance.
"""

import pytest
import warnings
from unittest.mock import Mock, patch, MagicMock
from vapi_python.vapi_python import Vapi, create_web_call
from vapi_python.types import MessageRole, MessageType


class TestVapiInit:
    """Tests for Vapi initialization."""

    def test_init_with_api_key(self):
        """Test Vapi initialization with API key."""
        vapi = Vapi(api_key='test-key')
        assert vapi.api_key == 'test-key'
        assert vapi.api_url == 'https://api.vapi.ai'

    def test_init_with_custom_url(self):
        """Test Vapi initialization with custom API URL."""
        vapi = Vapi(api_key='test-key', api_url='https://custom.api.com')
        assert vapi.api_url == 'https://custom.api.com'


class TestVapiAddMessage:
    """Tests for Vapi.add_message method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.vapi = Vapi(api_key='test-key')
        # Mock the client
        self.mock_client = Mock()
        self.vapi._Vapi__client = self.mock_client

    def test_add_message_user_role(self):
        """Test adding a user message."""
        self.vapi.add_message('user', 'Hello!')

        self.mock_client.send_app_message.assert_called_once()
        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['type'] == 'add-message'
        assert call_args['message']['role'] == 'user'
        assert call_args['message']['content'] == 'Hello!'

    def test_add_message_developer_role(self):
        """Test adding a developer message (OpenAI spec compliance)."""
        self.vapi.add_message('developer', 'Be concise.')

        self.mock_client.send_app_message.assert_called_once()
        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['type'] == 'add-message'
        assert call_args['message']['role'] == 'developer'
        assert call_args['message']['content'] == 'Be concise.'

    def test_add_message_with_enum(self):
        """Test adding a message using MessageRole enum."""
        self.vapi.add_message(MessageRole.DEVELOPER, 'Instructions here.')

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['message']['role'] == 'developer'

    def test_add_message_system_role(self):
        """Test adding a system message."""
        self.vapi.add_message('system', 'You are a helpful assistant.')

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['message']['role'] == 'system'

    def test_add_message_assistant_role(self):
        """Test adding an assistant message."""
        self.vapi.add_message('assistant', 'How can I help?')

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['message']['role'] == 'assistant'

    def test_add_message_tool_role_with_id(self):
        """Test adding a tool message with tool_call_id."""
        self.vapi.add_message(
            'tool',
            '{"result": "success"}',
            tool_call_id='call_123'
        )

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['message']['role'] == 'tool'
        assert call_args['message']['tool_call_id'] == 'call_123'

    def test_add_message_tool_role_without_id_raises(self):
        """Test that tool messages without tool_call_id raise error."""
        with pytest.raises(ValueError) as exc_info:
            self.vapi.add_message('tool', 'result')
        assert 'tool_call_id' in str(exc_info.value)

    def test_add_message_with_name(self):
        """Test adding a message with name parameter."""
        self.vapi.add_message(
            'tool',
            '{"temp": 72}',
            name='get_weather',
            tool_call_id='call_abc'
        )

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['message']['name'] == 'get_weather'

    def test_add_message_function_role_deprecated(self):
        """Test that function role emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.vapi.add_message('function', 'result')
            # Find deprecation warnings
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1

    def test_add_message_invalid_role_raises(self):
        """Test that invalid roles raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self.vapi.add_message('invalid_role', 'content')
        assert 'Invalid role' in str(exc_info.value)

    def test_add_message_case_insensitive(self):
        """Test that role validation is case-insensitive."""
        self.vapi.add_message('DEVELOPER', 'Test')

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['message']['role'] == 'developer'

    def test_add_message_no_client_raises(self):
        """Test that add_message raises when call not started."""
        vapi = Vapi(api_key='test-key')  # No client set
        with pytest.raises(Exception) as exc_info:
            vapi.add_message('user', 'Hello')
        assert 'Call not started' in str(exc_info.value)


class TestVapiSend:
    """Tests for Vapi.send method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.vapi = Vapi(api_key='test-key')
        self.mock_client = Mock()
        self.vapi._Vapi__client = self.mock_client

    def test_send_valid_message(self):
        """Test sending a valid message."""
        message = {'type': 'add-message', 'message': {'role': 'user', 'content': 'Hi'}}
        self.vapi.send(message)
        self.mock_client.send_app_message.assert_called_once_with(message)

    def test_send_missing_type_raises(self):
        """Test that messages without 'type' raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self.vapi.send({'message': 'test'})
        assert 'Invalid message format' in str(exc_info.value)

    def test_send_non_dict_raises(self):
        """Test that non-dict messages raise ValueError."""
        with pytest.raises(ValueError):
            self.vapi.send("string message")

    def test_send_no_client_raises(self):
        """Test that send raises when call not started."""
        vapi = Vapi(api_key='test-key')
        with pytest.raises(Exception) as exc_info:
            vapi.send({'type': 'test'})
        assert 'Call not started' in str(exc_info.value)


class TestVapiSay:
    """Tests for Vapi.say method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.vapi = Vapi(api_key='test-key')
        self.mock_client = Mock()
        self.vapi._Vapi__client = self.mock_client

    def test_say_basic(self):
        """Test basic say functionality."""
        self.vapi.say("Hello there!")

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['type'] == 'say'
        assert call_args['content'] == 'Hello there!'
        assert 'endCallAfter' not in call_args

    def test_say_with_end_call(self):
        """Test say with end_call_after flag."""
        self.vapi.say("Goodbye!", end_call_after=True)

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['type'] == 'say'
        assert call_args['endCallAfter'] is True


class TestVapiEndCall:
    """Tests for Vapi.end_call method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.vapi = Vapi(api_key='test-key')
        self.mock_client = Mock()
        self.vapi._Vapi__client = self.mock_client

    def test_end_call(self):
        """Test end_call sends correct message type."""
        self.vapi.end_call()

        call_args = self.mock_client.send_app_message.call_args[0][0]
        assert call_args['type'] == 'end-call'


class TestVapiStop:
    """Tests for Vapi.stop method."""

    def test_stop_clears_client(self):
        """Test that stop clears the client."""
        vapi = Vapi(api_key='test-key')
        mock_client = Mock()
        vapi._Vapi__client = mock_client

        vapi.stop()

        mock_client.leave.assert_called_once()
        assert vapi._Vapi__client is None

    def test_stop_when_no_client(self):
        """Test that stop is safe when no client exists."""
        vapi = Vapi(api_key='test-key')
        vapi.stop()  # Should not raise


class TestCreateWebCall:
    """Tests for the create_web_call function."""

    @patch('vapi_python.vapi_python.requests.post')
    def test_create_web_call_success(self, mock_post):
        """Test successful web call creation."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            'id': 'call_123',
            'webCallUrl': 'https://daily.co/room123'
        }
        mock_post.return_value = mock_response

        call_id, url = create_web_call(
            'https://api.vapi.ai',
            'test-key',
            {'assistantId': 'asst_123'}
        )

        assert call_id == 'call_123'
        assert url == 'https://daily.co/room123'

    @patch('vapi_python.vapi_python.requests.post')
    def test_create_web_call_failure(self, mock_post):
        """Test web call creation failure."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {'message': 'Invalid assistant ID'}
        mock_post.return_value = mock_response

        with pytest.raises(Exception) as exc_info:
            create_web_call(
                'https://api.vapi.ai',
                'test-key',
                {'assistantId': 'invalid'}
            )
        assert 'Invalid assistant ID' in str(exc_info.value)


class TestVapiStart:
    """Tests for Vapi.start method."""

    @patch('vapi_python.vapi_python.create_web_call')
    @patch('vapi_python.vapi_python.DailyCall')
    def test_start_with_assistant_id(self, mock_daily_call, mock_create_web_call):
        """Test starting call with assistant_id."""
        mock_create_web_call.return_value = ('call_123', 'https://daily.co/room')
        mock_client = Mock()
        mock_daily_call.return_value = mock_client

        vapi = Vapi(api_key='test-key')
        vapi.start(assistant_id='asst_123')

        mock_create_web_call.assert_called_once()
        call_args = mock_create_web_call.call_args[0][2]
        assert call_args['assistantId'] == 'asst_123'

    @patch('vapi_python.vapi_python.create_web_call')
    @patch('vapi_python.vapi_python.DailyCall')
    def test_start_with_assistant_config(self, mock_daily_call, mock_create_web_call):
        """Test starting call with inline assistant config."""
        mock_create_web_call.return_value = ('call_123', 'https://daily.co/room')
        mock_client = Mock()
        mock_daily_call.return_value = mock_client

        vapi = Vapi(api_key='test-key')
        assistant_config = {'model': 'gpt-5.2', 'voice': 'jennifer'}
        vapi.start(assistant=assistant_config)

        call_args = mock_create_web_call.call_args[0][2]
        assert call_args['assistant'] == assistant_config

    def test_start_no_assistant_raises(self):
        """Test that start without assistant raises error."""
        vapi = Vapi(api_key='test-key')
        with pytest.raises(Exception) as exc_info:
            vapi.start()
        assert 'No assistant specified' in str(exc_info.value)
