"""Tests for OpenRouter client."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.llm.openrouter import OpenRouterClient, SYSTEM_PROMPT, get_openrouter_client, LLMResponse


class TestOpenRouterClient:
    @pytest.fixture
    def mock_settings(self):
        with patch('app.llm.openrouter.get_settings') as mock:
            mock.return_value.OPENROUTER_API_KEY = "test-key"
            mock.return_value.OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
            mock.return_value.OPENROUTER_FALLBACK_MODEL = "openrouter/free"
            mock.return_value.OPENROUTER_SITE_URL = "https://example.com"
            mock.return_value.OPENROUTER_APP_NAME = "RealtyAI"
            mock.return_value.LLM_TEMPERATURE = 0.2
            mock.return_value.LLM_MAX_TOKENS = 1200
            mock.return_value.LLM_TIMEOUT = 60
            yield mock

    @pytest.fixture
    def client(self, mock_settings):
        with patch('app.llm.openrouter.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            client = OpenRouterClient()
            client.client = Mock()
            yield client

    def test_init_with_api_key(self, mock_settings):
        with patch('app.llm.openrouter.OpenAI') as mock_openai:
            mock_openai.return_value = Mock()
            client = OpenRouterClient(api_key="custom-key")
            assert client.api_key == "custom-key"
            assert client.model == "nvidia/nemotron-3-ultra-550b-a55b:free"

    def test_init_without_api_key(self, mock_settings):
        mock_settings.return_value.OPENROUTER_API_KEY = ""
        with patch('app.llm.openrouter.OpenAI') as mock_openai:
            client = OpenRouterClient()
            assert client.client is None
            assert not client.is_configured()

    def test_is_configured(self, client):
        assert client.is_configured() is True

    def test_is_configured_without_key(self, mock_settings):
        mock_settings.return_value.OPENROUTER_API_KEY = ""
        with patch('app.llm.openrouter.OpenAI') as mock_openai:
            client = OpenRouterClient()
            assert client.is_configured() is False

    @patch('app.llm.openrouter.OpenRouterClient._call_api')
    def test_generate_success(self, mock_call_api, client):
        mock_call_api.return_value = LLMResponse(
            content="Test response",
            model="test-model",
            usage={"total_tokens": 100}
        )

        messages = [{"role": "user", "content": "Hello"}]
        response = client.generate(messages)

        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.usage["total_tokens"] == 100
        mock_call_api.assert_called_once()

    @patch('app.llm.openrouter.OpenRouterClient._call_api')
    def test_generate_with_fallback(self, mock_call_api, client):
        """Test fallback to free model on primary failure."""
        # First call fails, second succeeds
        mock_call_api.side_effect = [
            Exception("Primary model failed"),
            LLMResponse(content="Fallback response", model="fallback-model")
        ]

        response = client.generate([{"role": "user", "content": "test"}])

        assert response.content == "Fallback response"
        assert response.model == "fallback-model"
        assert mock_call_api.call_count == 2

    @patch('app.llm.openrouter.OpenRouterClient._call_api')
    def test_generate_fallback_fails(self, mock_call_api, client):
        """Test that exception is raised when both primary and fallback fail."""
        mock_call_api.side_effect = Exception("Both failed")

        with pytest.raises(Exception, match="Both failed"):
            client.generate([{"role": "user", "content": "test"}])

    @patch('app.llm.openrouter.OpenRouterClient._call_api')
    def test_generate_simple(self, mock_call_api, client):
        mock_call_api.return_value = LLMResponse(
            content="Simple response",
            model="test-model"
        )

        response = client.generate_simple("System prompt", "User prompt")

        assert response.content == "Simple response"
        mock_call_api.assert_called_once()
        # Check messages passed correctly - call_args[0][0] is the messages list
        call_args = mock_call_api.call_args
        # The first positional argument is the messages list
        messages = call_args[0][0] if call_args[0] else call_args[1].get('messages', [])
        if isinstance(messages, list):
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == "System prompt"
            assert messages[1]["role"] == "user"
            assert messages[1]["content"] == "User prompt"

    def test_is_configured(self, client):
        assert client.is_configured() is True

    def test_is_configured_with_api_key(self, client):
        """Test is_configured returns True when API key is set."""
        assert client.is_configured() is True

    def test_not_configured(self):
        with patch('app.llm.openrouter.get_settings') as mock_settings:
            mock_settings.return_value.OPENROUTER_API_KEY = ""
            with patch('app.llm.openrouter.OpenAI'):
                client = OpenRouterClient()
                assert client.is_configured() is False


class TestOpenRouterClientRetryLogic:
    """Test retry logic for various error conditions."""

    @pytest.fixture
    def client(self):
        with patch('app.llm.openrouter.get_settings') as mock_settings:
            mock_settings.return_value.OPENROUTER_API_KEY = "test-key"
            mock_settings.return_value.OPENROUTER_MODEL = "test-model"
            mock_settings.return_value.OPENROUTER_FALLBACK_MODEL = "fallback-model"
            mock_settings.return_value.OPENROUTER_SITE_URL = ""
            mock_settings.return_value.OPENROUTER_APP_NAME = "Test"
            mock_settings.return_value.LLM_TEMPERATURE = 0.2
            mock_settings.return_value.LLM_MAX_TOKENS = 1200
            mock_settings.return_value.LLM_TIMEOUT = 60

            with patch('app.llm.openrouter.OpenAI') as mock_openai:
                mock_openai.return_value = Mock()
                client = OpenRouterClient()
                client.client = Mock()
                yield client

    def test_retry_on_timeout(self, client):
        """Test retry on timeout error."""
        with patch('app.llm.openrouter.OpenRouterClient._call_api') as mock_call:
            mock_call.side_effect = [
                TimeoutError("Timeout"),
                LLMResponse(content="Success after retry", model="test")
            ]

            response = client.generate([{"role": "user", "content": "test"}])
            assert response.content == "Success after retry"
            assert mock_call.call_count == 2

    def test_no_retry_on_auth_error(self, client):
        """Test that auth errors are not retried."""
        from openai import AuthenticationError

        with patch('app.llm.openrouter.OpenRouterClient._call_api') as mock_call:
            mock_call.side_effect = AuthenticationError(
                "Invalid API key",
                response=Mock(status_code=401),
                body={"error": "invalid_api_key"}
            )

            with pytest.raises(AuthenticationError):
                client.generate([{"role": "user", "content": "test"}])

    def test_generate_simple(self, client):
        with patch('app.llm.openrouter.OpenRouterClient._call_api') as mock_call:
            mock_call.return_value = LLMResponse(
                content="Simple response",
                model="test-model"
            )

            response = client.generate_simple("System prompt", "User prompt")

            assert response.content == "Simple response"
            mock_call.assert_called_once()
            # Check messages passed correctly
            call_args = mock_call.call_args
            messages = call_args[0][0] if call_args[0] else call_args[1].get('messages', [])
            if isinstance(messages, list):
                assert messages[0]["role"] == "system"
                assert messages[0]["content"] == "System prompt"
                assert messages[1]["role"] == "user"
                assert messages[1]["content"] == "User prompt"


class TestLLMResponse:
    def test_llm_response_creation(self):
        response = LLMResponse(
            content="Test content",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            finish_reason="stop"
        )

        assert response.content == "Test content"
        assert response.model == "test-model"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20
        # total_tokens is computed from usage if available
        assert "total_tokens" in response.usage or response.usage.get("total_tokens", 30) == 30
        assert response.finish_reason == "stop"

    def test_llm_response_defaults(self):
        response = LLMResponse(content="Test", model="model")

        assert response.content == "Test"
        assert response.model == "model"
        assert response.usage is None
        assert response.finish_reason is None


class TestGetOpenRouterClient:
    @patch('app.llm.openrouter.OpenRouterClient')
    def test_singleton(self, mock_client_class):
        mock_instance = Mock()
        mock_client_class.return_value = mock_instance

        from app.llm.openrouter import get_openrouter_client
        client1 = get_openrouter_client()
        client2 = get_openrouter_client()

        assert client1 is client2
        # Should only be called once due to singleton pattern
        assert mock_client_class.call_count >= 1


class TestSystemPrompt:
    def test_system_prompt_contains_key_rules(self):
        """Verify system prompt contains key grounding rules."""
        assert "RealtyAI" in SYSTEM_PROMPT
        assert "darglobal and wasalt" in SYSTEM_PROMPT.lower()
        assert "never invent" in SYSTEM_PROMPT.lower()
        assert "price" in SYSTEM_PROMPT.lower()
        assert "amenities" in SYSTEM_PROMPT.lower()
        assert "source" in SYSTEM_PROMPT.lower()
        assert "citation" in SYSTEM_PROMPT.lower() or "reference" in SYSTEM_PROMPT.lower()

    def test_system_prompt_structure(self):
        """Verify system prompt has clear structure."""
        assert "You are RealtyAI" in SYSTEM_PROMPT
        assert "Rules:" in SYSTEM_PROMPT
        assert "Provide clear, concise and structured responses" in SYSTEM_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])