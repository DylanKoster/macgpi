import pytest
from unittest.mock import patch
from macgpi.__main__ import cli


class TestCliArgumentParsing:
    """Tests for CLI argument parsing."""

    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", ["macgpi", "test input", "test-model", "/output/path"])
    def test_cli_required_arguments(self, mock_macgpi):
        """Test CLI with only required arguments."""
        cli()

        # Verify macgpi was called with required arguments
        mock_macgpi.assert_called_once()
        call_kwargs = mock_macgpi.call_args.kwargs
        assert call_kwargs["input"] == "test input"
        assert call_kwargs["model_name"] == "test-model"
        assert call_kwargs["output_dir"] == "/output/path"

    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", [
        "macgpi", "test input", "test-model", "/output/path",
        "--model-host", "custom.host",
        "--model-port", "9000",
        "--prompt-dir", "/custom/prompts",
        "--model-config", "model.yaml",
        "--agent-config", "agent.yaml",
        "--phases-config", "phases.json"

    ])
    def test_cli_with_kwargs(self, mock_macgpi):
        """Test CLI with custom host and port."""
        cli()

        call_kwargs = mock_macgpi.call_args.kwargs
        assert call_kwargs["model_host"] == "custom.host"
        assert call_kwargs["model_port"] == 9000
        assert call_kwargs["prompt_dir"] == "/custom/prompts"
        assert call_kwargs["model_config"] == "model.yaml"
        assert call_kwargs["agent_config"] == "agent.yaml"
        assert call_kwargs["phases_config"] == "phases.json"

    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", [
        "macgpi", "test input", "test-model", "/output/path",
        "--log-level", "DEBUG"
    ])
    def test_cli_with_log_level(self, mock_macgpi):
        """Test CLI with log level."""
        cli()

        # Log level should be processed but not passed to macgpi
        call_kwargs = mock_macgpi.call_args.kwargs
        assert "log_level" not in call_kwargs

    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", ["macgpi", "test input", "test-model", "/output/path"])
    def test_cli_defaults(self, mock_macgpi):
        """Test that default host is localhost."""
        cli()

        call_kwargs = mock_macgpi.call_args.kwargs
        assert call_kwargs["model_host"] == "localhost"
        assert call_kwargs["model_port"] == 8000
        assert call_kwargs["prompt_dir"] is None
        assert call_kwargs["model_config"] is None
        assert call_kwargs["agent_config"] is None

    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", [
        "macgpi", "test input", "test-model", "/output/path",
        "--model-port", "9999"
    ])
    def test_cli_port_converted_to_int(self, mock_macgpi):
        """Test that port is converted to integer."""
        cli()

        call_kwargs = mock_macgpi.call_args.kwargs
        assert isinstance(call_kwargs["model_port"], int)
        assert call_kwargs["model_port"] == 9999

    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", [
        "macgpi", "test input", "test-model", "/output/path",
        "--model-port", "invalid"
    ])
    def test_cli_invalid_port_raises_error(self, mock_macgpi):
        """Test that invalid port raises error."""
        with pytest.raises(SystemExit):
            cli()


class TestCliLogging:
    """Tests for CLI logging configuration."""

    @patch("macgpi.__main__.logging.basicConfig")
    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", [
        "macgpi", "test input", "test-model", "/output/path",
        "--log-level", "DEBUG"
    ])
    def test_cli_debug_logging(self, mock_macgpi, mock_logging):
        """Test that DEBUG log level is configured."""
        cli()

        mock_logging.assert_called_once()
        call_kwargs = mock_logging.call_args.kwargs
        assert call_kwargs["level"] == "DEBUG"

    @patch("macgpi.__main__.logging.basicConfig")
    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", [
        "macgpi", "test input", "test-model", "/output/path",
        "--log-level", "WARNING"
    ])
    def test_cli_info_logging(self, mock_macgpi, mock_logging):
        """Test that WARNING log level is configured."""
        cli()

        call_kwargs = mock_logging.call_args.kwargs
        assert call_kwargs["level"] == "WARNING"

    @patch("macgpi.__main__.logging.basicConfig")
    @patch("macgpi.__main__.macgpi")
    @patch("sys.argv", ["macgpi", "test input", "test-model", "/output/path"])
    def test_cli_default_logging(self, mock_macgpi, mock_logging):
        """Test that default log level is INFO."""
        cli()

        call_kwargs = mock_logging.call_args.kwargs
        assert call_kwargs["level"] == "INFO"
