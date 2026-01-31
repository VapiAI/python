"""
Tests for Vapi Python SDK type definitions.

This module tests the MessageRole, Message, and related type utilities
for OpenAI API specification compliance.
"""

import pytest
import warnings
from vapi_python.types import (
    MessageRole,
    Message,
    MessageType,
    OpenAIModel,
    VALID_MESSAGE_ROLES,
    DEVELOPER_ROLE_MODELS,
    validate_role,
    supports_developer_role,
)


class TestMessageRole:
    """Tests for the MessageRole enum."""

    def test_all_roles_defined(self):
        """Verify all expected roles are defined."""
        expected_roles = {'system', 'user', 'assistant', 'developer', 'tool', 'function'}
        actual_roles = {role.value for role in MessageRole}
        assert actual_roles == expected_roles

    def test_role_values(self):
        """Verify role string values."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.DEVELOPER.value == "developer"
        assert MessageRole.TOOL.value == "tool"
        assert MessageRole.FUNCTION.value == "function"

    def test_developer_role_exists(self):
        """Test that developer role is available (OpenAI spec compliance)."""
        assert hasattr(MessageRole, 'DEVELOPER')
        assert MessageRole.DEVELOPER.value == "developer"


class TestValidateRole:
    """Tests for the validate_role function."""

    def test_valid_roles(self):
        """Test validation of all valid roles."""
        for role in ['system', 'user', 'assistant', 'developer', 'tool', 'function']:
            result = validate_role(role)
            assert result == role

    def test_case_insensitive(self):
        """Test that role validation is case-insensitive."""
        assert validate_role('USER') == 'user'
        assert validate_role('Developer') == 'developer'
        assert validate_role('SYSTEM') == 'system'

    def test_invalid_role_raises(self):
        """Test that invalid roles raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            validate_role('invalid_role')
        assert 'Invalid role' in str(exc_info.value)

    def test_function_role_deprecation_warning(self):
        """Test that 'function' role emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_role('function')
            assert result == 'function'
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert 'deprecated' in str(w[0].message).lower()
            assert 'tool' in str(w[0].message).lower()

    def test_developer_role_no_warning(self):
        """Test that 'developer' role does not emit warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = validate_role('developer')
            assert result == 'developer'
            # Filter out any unrelated warnings
            deprecation_warnings = [
                warning for warning in w
                if issubclass(warning.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0


class TestMessage:
    """Tests for the Message class."""

    def test_basic_message(self):
        """Test creating a basic message."""
        msg = Message(role='user', content='Hello!')
        assert msg.role == 'user'
        assert msg.content == 'Hello!'
        assert msg.name is None
        assert msg.tool_call_id is None

    def test_message_with_enum_role(self):
        """Test creating a message with MessageRole enum."""
        msg = Message(role=MessageRole.DEVELOPER, content='Be concise.')
        assert msg.role == 'developer'
        assert msg.content == 'Be concise.'

    def test_developer_role_message(self):
        """Test creating a message with developer role."""
        msg = Message(role='developer', content='Follow these instructions.')
        assert msg.role == 'developer'
        assert msg.content == 'Follow these instructions.'

    def test_tool_message_requires_tool_call_id(self):
        """Test that tool messages require tool_call_id."""
        with pytest.raises(ValueError) as exc_info:
            Message(role='tool', content='result')
        assert 'tool_call_id' in str(exc_info.value)

    def test_tool_message_with_tool_call_id(self):
        """Test creating a valid tool message."""
        msg = Message(
            role='tool',
            content='{"result": "success"}',
            tool_call_id='call_123'
        )
        assert msg.role == 'tool'
        assert msg.tool_call_id == 'call_123'

    def test_to_dict_basic(self):
        """Test to_dict for a basic message."""
        msg = Message(role='user', content='Test')
        result = msg.to_dict()
        assert result == {'role': 'user', 'content': 'Test'}

    def test_to_dict_with_all_fields(self):
        """Test to_dict with all optional fields."""
        msg = Message(
            role='tool',
            content='result',
            name='get_weather',
            tool_call_id='call_abc'
        )
        result = msg.to_dict()
        assert result == {
            'role': 'tool',
            'content': 'result',
            'name': 'get_weather',
            'tool_call_id': 'call_abc'
        }

    def test_invalid_role_raises(self):
        """Test that invalid roles raise ValueError."""
        with pytest.raises(ValueError):
            Message(role='invalid', content='test')


class TestMessageType:
    """Tests for the MessageType enum."""

    def test_message_types_defined(self):
        """Verify all expected message types are defined."""
        assert MessageType.ADD_MESSAGE.value == "add-message"
        assert MessageType.SAY.value == "say"
        assert MessageType.END_CALL.value == "end-call"
        assert MessageType.TRANSFER_CALL.value == "transfer-call"


class TestOpenAIModel:
    """Tests for the OpenAIModel enum."""

    def test_gpt5_models_defined(self):
        """Test that GPT-5 series models are defined."""
        assert OpenAIModel.GPT_5_2.value == "gpt-5.2"
        assert OpenAIModel.GPT_5_2_CHAT.value == "gpt-5.2-chat"
        assert OpenAIModel.GPT_5_2_CHAT_LATEST.value == "gpt-5.2-chat-latest"
        assert OpenAIModel.GPT_5_2_CODEX.value == "gpt-5.2-codex"

    def test_o_series_models_defined(self):
        """Test that o-series models are defined."""
        assert OpenAIModel.O1.value == "o1"
        assert OpenAIModel.O1_MINI.value == "o1-mini"
        assert OpenAIModel.O3.value == "o3"
        assert OpenAIModel.O3_MINI.value == "o3-mini"
        assert OpenAIModel.O4_MINI.value == "o4-mini"

    def test_legacy_models_defined(self):
        """Test that legacy models are still available."""
        assert OpenAIModel.GPT_4.value == "gpt-4"
        assert OpenAIModel.GPT_4O.value == "gpt-4o"
        assert OpenAIModel.GPT_3_5_TURBO.value == "gpt-3.5-turbo"


class TestSupportsDeveloperRole:
    """Tests for the supports_developer_role function."""

    def test_gpt5_models_support_developer(self):
        """Test that GPT-5 models support developer role."""
        assert supports_developer_role("gpt-5.2") is True
        assert supports_developer_role("gpt-5.2-chat") is True
        assert supports_developer_role("gpt-5.2-chat-latest") is True
        assert supports_developer_role("gpt-5.2-codex") is True

    def test_o_series_support_developer(self):
        """Test that o-series models support developer role."""
        assert supports_developer_role("o1") is True
        assert supports_developer_role("o1-mini") is True
        assert supports_developer_role("o3") is True
        assert supports_developer_role("o3-mini") is True
        assert supports_developer_role("o4-mini") is True

    def test_older_models_no_developer(self):
        """Test that older models do not support developer role."""
        assert supports_developer_role("gpt-4") is False
        assert supports_developer_role("gpt-4o") is False
        assert supports_developer_role("gpt-3.5-turbo") is False

    def test_unknown_model(self):
        """Test behavior with unknown model."""
        assert supports_developer_role("unknown-model") is False


class TestValidMessageRolesConstant:
    """Tests for the VALID_MESSAGE_ROLES constant."""

    def test_contains_all_roles(self):
        """Test that constant contains all valid roles."""
        expected = {'system', 'user', 'assistant', 'developer', 'tool', 'function'}
        assert VALID_MESSAGE_ROLES == expected

    def test_developer_in_valid_roles(self):
        """Test that developer role is in valid roles."""
        assert 'developer' in VALID_MESSAGE_ROLES


class TestDeveloperRoleModelsConstant:
    """Tests for the DEVELOPER_ROLE_MODELS constant."""

    def test_contains_gpt5_models(self):
        """Test that GPT-5 models are in the set."""
        assert "gpt-5.2" in DEVELOPER_ROLE_MODELS
        assert "gpt-5.2-chat" in DEVELOPER_ROLE_MODELS

    def test_contains_o_series(self):
        """Test that o-series models are in the set."""
        assert "o1" in DEVELOPER_ROLE_MODELS
        assert "o3" in DEVELOPER_ROLE_MODELS

    def test_does_not_contain_older_models(self):
        """Test that older models are not in the set."""
        assert "gpt-4" not in DEVELOPER_ROLE_MODELS
        assert "gpt-3.5-turbo" not in DEVELOPER_ROLE_MODELS
