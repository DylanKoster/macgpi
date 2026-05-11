from unittest.mock import patch, MagicMock
from macgpi.engine.vllm import vllm_health


class TestVllmHealth:
    """Tests for vllm_health function."""

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_success(self, mock_get):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = vllm_health("localhost", 8000)

        assert result is True
        mock_get.assert_called_once_with(
            "http://localhost:8000/health",
            timeout=2
        )

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_custom_host_port(self, mock_get):
        """Test health check with custom host and port."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        result = vllm_health("example.com", 9000)

        assert result is True
        mock_get.assert_called_once_with(
            "http://example.com:9000/health",
            timeout=2
        )

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_status_codes_200_299(self, mock_get):
        """Test that status codes 200-299 are successful."""
        for status_code in [200, 201, 250, 299]:
            mock_response = MagicMock()
            mock_response.status_code = status_code
            mock_get.return_value = mock_response

            result = vllm_health("localhost", 8000)
            assert result is True

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_status_code_300(self, mock_get):
        """Test that status code 300 is unsuccessful."""
        mock_response = MagicMock()
        mock_response.status_code = 300
        mock_get.return_value = mock_response

        result = vllm_health("localhost", 8000)

        assert result is False

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_status_code_404(self, mock_get):
        """Test that 404 error is handled."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        result = vllm_health("localhost", 8000)

        assert result is False

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_status_code_500(self, mock_get):
        """Test that 500 error is handled."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        result = vllm_health("localhost", 8000)

        assert result is False

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_connection_timeout(self, mock_get):
        """Test timeout exception is handled."""
        mock_get.side_effect = TimeoutError()

        result = vllm_health("localhost", 8000)

        assert result is False

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_connection_error(self, mock_get):
        """Test connection error is handled."""
        mock_get.side_effect = ConnectionError()

        result = vllm_health("localhost", 8000)

        assert result is False

    @patch("macgpi.engine.vllm.requests.get")
    def test_health_check_generic_exception(self, mock_get):
        """Test generic exception is handled."""
        mock_get.side_effect = Exception("Generic error")

        result = vllm_health("localhost", 8000)

        assert result is False
