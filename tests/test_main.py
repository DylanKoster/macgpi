import os
import pytest
from macgpi.engine.main import MACGPi


@pytest.mark.unit
class TestMacgpiMain:
    """Basic tests for macgpi function."""

    def test_macgpi_unhealthy_vllm_returns_early(self, mocker, temp_dir):
        """Test that macgpi returns early if vLLM is unhealthy."""
        mock_health = mocker.patch("macgpi.engine.main.vllm_health", return_value=False)

        macgpi: MACGPi = MACGPi(
            input_description="test input",
            model_name="test-model",
            output_dir=temp_dir
        )
        result: bool = macgpi.run()

        assert result is False
        mock_health.assert_called_once()

    def test_macgpi_creates_output_dir(self, mocker, temp_dir):
        """Test that macgpi creates output directory structure."""
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)
        mocker.patch("macgpi.engine.main.is_phase_dir", return_value=True)

        mock_tm = mocker.MagicMock()
        mock_tm.prompt_dir = temp_dir
        mocker.patch("macgpi.engine.main.TemplateManager", return_value=mock_tm)
        mocker.patch("macgpi.engine.main.AgentManager", return_value=mocker.MagicMock())

        output_dir = os.path.join(temp_dir, "output")

        macgpi: MACGPi = MACGPi(
            input_description="test input",
            model_name="test-model",
            output_dir=output_dir,
            prompt_dir=None,
            model_config_file=None,
            agent_config_file=None,
            phases_config_file=None
        )
        macgpi.run()

        # Check that docs directory was created
        assert os.path.exists(os.path.join(output_dir, "docs"))

    def test_macgpi_writes_project_description(self, mocker, temp_dir):
        """Test that project description is written to output."""
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)
        mocker.patch("macgpi.engine.main.is_phase_dir", return_value=True)

        mock_tm = mocker.MagicMock()
        mock_tm.prompt_dir = temp_dir
        mocker.patch("macgpi.engine.main.TemplateManager", return_value=mock_tm)
        mocker.patch("macgpi.engine.main.AgentManager", return_value=mocker.MagicMock())

        output_dir = os.path.join(temp_dir, "output")
        test_input = "Build a web application"

        macgpi: MACGPi = MACGPi(
            input_description=test_input,
            model_name="test-model",
            output_dir=output_dir,
            prompt_dir=None,
            model_config_file=None,
            agent_config_file=None,
            phases_config_file=None
        )
        macgpi.run()

        # Check that project description was written
        prd_path = os.path.join(output_dir, "docs", "project_description.md")
        assert os.path.exists(prd_path)
        with open(prd_path, "r") as f:
            content = f.read()
        assert content == test_input

    def test_macgpi_fails_on_invalid_phase_directories(
        self, mocker, temp_prompts_dir, sample_output_dir
    ):
        """Test that macgpi fails on invalid phase directories."""
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)

        from macgpi.engine.template_manager import TemplateManager
        actual_template_mgr = TemplateManager(prompt_dir=temp_prompts_dir)
        mocker.patch("macgpi.engine.main.TemplateManager", return_value=actual_template_mgr)
        mocker.patch("macgpi.engine.main.AgentManager", return_value=mocker.MagicMock())

        # Create config with invalid phase
        config = {
            "phases": {
                "invalid": {
                    "inputs": {
                        "system_prd": "docs/project_description.md"
                    },
                    "schema": False,
                    "path": "04_invalid/",
                    "output_path": "docs/plan.json",
                    "next": "implement"
                }
            }
        }
        mocker.patch("macgpi.engine.main.parse_phase_config", return_value=config)

        # Should return False due to invalid phase
        macgpi: MACGPi = MACGPi(
            input_description="test",
            model_name="test-model",
            output_dir=sample_output_dir,
            prompt_dir=temp_prompts_dir,
            phases_config_file=None
        )
        result = macgpi.run()

        assert result is False


@pytest.mark.integration
class TestMacgpiIntegration:
    """Integration tests using real TemplateManager and phase directories."""

    def test_macgpi_validates_phase_directories(
        self, mocker, temp_prompts_dir, sample_phase_config, sample_output_dir
    ):
        """Test that macgpi validates phase directories."""
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)

        from macgpi.engine.template_manager import TemplateManager
        actual_template_mgr = TemplateManager(prompt_dir=temp_prompts_dir)
        mocker.patch("macgpi.engine.main.TemplateManager", return_value=actual_template_mgr)

        mock_agent = mocker.MagicMock()
        mocker.patch("macgpi.engine.main.AgentManager", return_value=mock_agent)

        macgpi: MACGPi = MACGPi(
            input_description="test",
            model_name="test-model",
            output_dir=sample_output_dir,
            prompt_dir=temp_prompts_dir,
            phases_config_file=sample_phase_config
        )
        result = macgpi.run()

        assert result is True

    def test_macgpi_executes_phases_in_order(
        self, mocker, temp_prompts_dir, sample_output_dir, sample_phase_config
    ):
        """Test that phases are executed in correct order."""
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)

        # Create actual template manager
        from macgpi.engine.template_manager import TemplateManager
        actual_template_mgr = TemplateManager(prompt_dir=temp_prompts_dir)
        mocker.patch("macgpi.engine.main.TemplateManager", return_value=actual_template_mgr)

        mock_agent = mocker.MagicMock()
        mocker.patch("macgpi.engine.main.AgentManager", return_value=mock_agent)

        macgpi: MACGPi = MACGPi(
            input_description="test",
            model_name="test-model",
            output_dir=sample_output_dir,
            prompt_dir=temp_prompts_dir,
            phases_config_file=sample_phase_config
        )
        result = macgpi.run()

        # Verify agent was called
        assert result is True
        mock_agent.run.assert_called()

        phase_visits = macgpi.get_phase_visits()
        assert phase_visits["plan"] == 1
        assert phase_visits["implement"] == 1
        assert phase_visits["evaluate"] == 3
        assert phase_visits["revise"] == 3
        assert phase_visits["finally"] == 1


@pytest.mark.unit
class TestMacgpiErrorHandling:
    """Tests for macgpi error handling."""

    def test_macgpi_handles_missing_config_gracefully(self, mocker, temp_dir):
        """Test that macgpi handles missing config gracefully."""
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)

        macgpi: MACGPi = MACGPi(
            input_description="test",
            model_name="test-model",
            output_dir=temp_dir,
            phases_config_file="/nonexistent/path/config.json"
        )
        result = macgpi.run()

        assert result is False
