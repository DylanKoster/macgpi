import pytest
from macgpi.engine.vllm import vllm_health


@pytest.mark.unit
class TestVllmHealth:
    """Tests for vllm_health function."""

    def test_health_check_success(self, mocker):
        """Test successful health check."""
        mock_get = mocker.patch("macgpi.engine.vllm.requests.get")
        mock_get.return_value.status_code = 200

        result = vllm_health("localhost", 8000)

        assert result is True
        mock_get.assert_called_once_with(
            "http://localhost:8000/health",
            timeout=2
        )

    def test_health_check_custom_host_port(self, mocker):
        """Test health check with custom host and port."""
        mock_get = mocker.patch("macgpi.engine.vllm.requests.get")
        mock_get.return_value.status_code = 200

        result = vllm_health("example.com", 9000)

        assert result is True
        mock_get.assert_called_once_with(
            "http://example.com:9000/health",
            timeout=2
        )

    @pytest.mark.parametrize("status_code", [200, 201, 250, 299])
    def test_health_check_2xx_status_codes_pass(self, mocker, status_code):
        """Test that status codes 200-299 are successful."""
        mock_get = mocker.patch("macgpi.engine.vllm.requests.get")
        mock_get.return_value.status_code = status_code

        assert vllm_health("localhost", 8000) is True

    @pytest.mark.parametrize("status_code", [300, 404, 500])
    def test_health_check_non_2xx_status_codes_fail(self, mocker, status_code):
        """Test that status codes outside 2xx range are unsuccessful."""
        mock_get = mocker.patch("macgpi.engine.vllm.requests.get")
        mock_get.return_value.status_code = status_code

        assert vllm_health("localhost", 8000) is False

    @pytest.mark.parametrize("exc", [TimeoutError, ConnectionError, Exception])
    def test_health_check_exceptions_return_false(self, mocker, exc):
        """Test that connection exceptions are handled and return False."""
        mock_get = mocker.patch("macgpi.engine.vllm.requests.get")
        mock_get.side_effect = exc()

        assert vllm_health("localhost", 8000) is False
