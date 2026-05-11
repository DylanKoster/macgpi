import os
import pytest
import yaml
from unittest.mock import patch, MagicMock
from macgpi.engine.agent_manager import AgentManager


@pytest.fixture
def mock_model() -> MagicMock:
    """Create a mock model object."""
    return MagicMock()


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent object."""
    agent: MagicMock = MagicMock()
    agent.run.return_value = "Agent output"
    return agent


@pytest.fixture
def sample_model_config(temp_dir) -> str:
    """Create a sample model configuration file."""
    config: dict = {
        "observation_template": "template",
        "model_kwargs": {
            "drop_params": True
        }
    }
    config_file: str = os.path.join(temp_dir, "model.config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def sample_agent_config(temp_dir) -> str:
    """Create a sample agent configuration file."""
    config: dict = {
        "system_template": "You are a helpful assistant",
        "instance_template": "Solve this task: {{ task }}"
    }
    config_file: str = os.path.join(temp_dir, "agent.config.yaml")
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return config_file


class TestAgentManager:
    """Tests for AgentManager."""

    @patch("macgpi.engine.agent_manager.DefaultAgent")
    @patch("macgpi.engine.agent_manager.get_model")
    def test_init_with_model_and_agent_configs(
        self,
        mock_get_model: MagicMock,
        mock_agent_class: MagicMock,
        sample_model_config: str,
        sample_agent_config: str,
        mock_model: MagicMock,
        mock_agent: MagicMock
    ):
        """Test initialization with both model and agent configs."""
        mock_get_model.return_value = mock_model
        mock_agent_class.return_value = mock_agent

        manager: AgentManager = AgentManager(
            "test-model",
            model_host="localhost",
            model_port=8000,
            model_config_file=sample_model_config,
            agent_config_file=sample_agent_config
        )

        assert manager.agent == mock_agent
        assert manager.model_config is not None
        assert "observation_template" in manager.model_config.keys()
        assert manager.agent_config is not None
        assert "system_template" in manager.agent_config.keys()

    @patch("macgpi.engine.agent_manager.DefaultAgent")
    @patch("macgpi.engine.agent_manager.get_model")
    def test_init_with_custom_host_port(
        self,
        mock_get_model: MagicMock,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_agent: MagicMock
    ):
        """Test initialization with custom host and port."""
        mock_get_model.return_value = mock_model
        mock_agent_class.return_value = mock_agent

        manager: AgentManager = AgentManager(
            "test-model",
            model_host="example.com",
            model_port=9000,
            model_config_file=None,
            agent_config_file=None
        )

        # Check that model config contains correct API base
        assert "example.com:9000" in manager.model_config["model_kwargs"]["api_base"]

    @patch("macgpi.engine.agent_manager.DefaultAgent")
    @patch("macgpi.engine.agent_manager.get_model")
    def test_init_without_configs_uses_defaults(
        self,
        mock_get_model: MagicMock,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_agent: MagicMock
    ):
        """Test initialization without configs uses default paths."""
        mock_get_model.return_value = mock_model
        mock_agent_class.return_value = mock_agent

        manager: AgentManager = AgentManager(
            "test-model",
            model_host="localhost",
            model_port=8000
        )

        assert manager.agent == mock_agent
        print(manager.model_config)
        assert manager.model_config is not None
        assert "observation_template" in manager.model_config.keys()
        assert "format_error_template" in manager.model_config.keys()

        assert manager.agent_config is not None
        assert "system_template" in manager.agent_config.keys()
        assert manager.agent_config["step_limit"] == 0
        assert manager.agent_config["cost_limit"] == 0

    def test_load_model_config_updates_required_fields(self, sample_model_config: str):
        """Test that required model config fields are updated."""
        manager: AgentManager = AgentManager.__new__(AgentManager)
        manager.load_model_config(
            sample_model_config,
            "gpt-4",
            "remote.server",
            9000
        )

        # Check required fields
        assert manager.model_config["model_name"] == "gpt-4"
        assert manager.model_config["api_key"] == "EMPTY"
        assert manager.model_config["cost_tracking"] == "ignore_errors"
        # custom_llm_provider might be nested in model_kwargs
        if "custom_llm_provider" in manager.model_config:
            assert manager.model_config["custom_llm_provider"] == "hosted_vllm"
        elif "custom_llm_provider" in manager.model_config.get("model_kwargs", {}):
            assert manager.model_config["model_kwargs"]["custom_llm_provider"] == "hosted_vllm"
        # API base should be constructed with the right host and port
        assert "remote.server:9000" in manager.model_config["model_kwargs"]["api_base"]

    @patch("macgpi.engine.agent_manager.DefaultAgent")
    @patch("macgpi.engine.agent_manager.get_model")
    def test_run_with_prompt(
        self,
        mock_get_model: MagicMock,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_agent: MagicMock
    ):
        """Test running agent with a prompt."""
        expected_output: str = "Agent response"
        mock_agent.run.return_value = expected_output
        mock_get_model.return_value = mock_model
        mock_agent_class.return_value = mock_agent

        manager: AgentManager = AgentManager(
            "test-model",
            model_host="localhost",
            model_port=8000
        )

        result = manager.run("Test prompt")

        assert result == expected_output
        mock_agent.run.assert_called_once_with("Test prompt")

    @patch("macgpi.engine.agent_manager.DefaultAgent")
    @patch("macgpi.engine.agent_manager.get_model")
    def test_run_multiple_prompts(
        self,
        mock_get_model: MagicMock,
        mock_agent_class: MagicMock,
        mock_model: MagicMock,
        mock_agent: MagicMock
    ):
        """Test running agent multiple times."""
        mock_get_model.return_value = mock_model
        mock_agent_class.return_value = mock_agent
        mock_agent.run.side_effect = ["output1", "output2", "output3"]

        manager: AgentManager = AgentManager(
            "test-model",
            model_host="localhost",
            model_port=8000
        )

        result1 = manager.run("Prompt 1")
        result2 = manager.run("Prompt 2")
        result3 = manager.run("Prompt 3")

        assert result1 == "output1"
        assert result2 == "output2"
        assert result3 == "output3"
        assert mock_agent.run.call_count == 3
