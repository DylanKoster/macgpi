import os
from unittest.mock import patch, MagicMock
from macgpi.engine.main import macgpi


class TestMacgpiMain:
    """Basic tests for macgpi function."""

    @patch("macgpi.engine.main.vllm_health")
    def test_macgpi_unhealthy_vllm_returns_early(self, mock_health, temp_dir):
        """Test that macgpi returns early if vLLM is unhealthy."""
        mock_health.return_value = False

        result = macgpi(
            input="test input",
            model_name="test-model",
            output_dir=temp_dir
        )

        assert result is False
        mock_health.assert_called_once()

    @patch("macgpi.engine.main.is_phase_dir")
    @patch("macgpi.engine.main.AgentManager")
    @patch("macgpi.engine.main.TemplateManager")
    @patch("macgpi.engine.main.vllm_health")
    def test_macgpi_creates_output_dir(
        self,
        mock_health,
        mock_template_mgr,
        mock_agent_mgr,
        mock_is_phase_dir,
        temp_dir
    ):
        """Test that macgpi creates output directory structure."""
        mock_health.return_value = True
        mock_is_phase_dir.return_value = True

        mock_tm = MagicMock()
        mock_tm.prompt_dir = temp_dir
        mock_template_mgr.return_value = mock_tm
        mock_agent_mgr.return_value = MagicMock()

        output_dir = os.path.join(temp_dir, "output")

        macgpi(
            input="test input",
            model_name="test-model",
            output_dir=output_dir,
            prompt_dir=None,
            model_config=None,
            agent_config=None,
            phases_config=None
        )

        # Check that docs directory was created
        assert os.path.exists(os.path.join(output_dir, "docs"))

    @patch("macgpi.engine.main.is_phase_dir")
    @patch("macgpi.engine.main.AgentManager")
    @patch("macgpi.engine.main.TemplateManager")
    @patch("macgpi.engine.main.vllm_health")
    def test_macgpi_writes_project_description(
        self,
        mock_health,
        mock_template_mgr,
        mock_agent_mgr,
        mock_is_phase_dir,
        temp_dir
    ):
        """Test that project description is written to output."""
        mock_health.return_value = True
        mock_is_phase_dir.return_value = True

        mock_tm = MagicMock()
        mock_tm.prompt_dir = temp_dir
        mock_template_mgr.return_value = mock_tm
        mock_agent_mgr.return_value = MagicMock()

        output_dir = os.path.join(temp_dir, "output")
        test_input = "Build a web application"

        macgpi(
            input=test_input,
            model_name="test-model",
            output_dir=output_dir,
            prompt_dir=None,
            model_config=None,
            agent_config=None,
            phases_config=None
        )

        # Check that project description was written
        prd_path = os.path.join(output_dir, "docs", "project_description.md")
        assert os.path.exists(prd_path)
        with open(prd_path, "r") as f:
            content = f.read()
        assert content == test_input

    @patch("macgpi.engine.main.parse_phase_config")
    @patch("macgpi.engine.main.AgentManager")
    @patch("macgpi.engine.main.TemplateManager")
    @patch("macgpi.engine.main.vllm_health")
    def test_macgpi_validates_phase_directories(
        self,
        mock_health,
        mock_template_mgr,
        mock_agent_mgr,
        mock_parse_config,
        temp_prompts_dir,
        temp_dir
    ):
        """Test that macgpi validates phase directories."""
        mock_health.return_value = True

        # Create template manager with actual prompts
        from macgpi.engine.template_manager import TemplateManager
        actual_template_mgr = TemplateManager(prompt_dir=temp_prompts_dir)
        mock_template_mgr.return_value = actual_template_mgr

        mock_agent = MagicMock()
        mock_agent_mgr.return_value = mock_agent

        # Create valid phase config that only has one phase
        config = {
            "phases": {
                "plan": {
                    "inputs": {
                        "system_prd": "docs/project_description.md"
                    },
                    "schema": True,
                    "path": "01_plan/",
                    "output_path": "docs/plan.json",
                    "next": "finish"
                }
            }
        }
        mock_parse_config.return_value = config

        output_dir = os.path.join(temp_dir, "output")

        # Should complete validation successfully
        result = macgpi(
            input="test",
            model_name="test-model",
            output_dir=output_dir,
            prompt_dir=temp_prompts_dir,
            phases_config=None
        )
        assert result is True

    @patch("macgpi.engine.main.parse_phase_config")
    @patch("macgpi.engine.main.AgentManager")
    @patch("macgpi.engine.main.TemplateManager")
    @patch("macgpi.engine.main.vllm_health")
    def test_macgpi_fails_on_invalid_phase_directories(
        self,
        mock_health,
        mock_template_mgr,
        mock_agent_mgr,
        mock_parse_config,
        temp_prompts_dir,
        temp_dir
    ):
        """Test that macgpi fails on invalid phase directories."""
        mock_health.return_value = True

        # Create template manager with actual prompts
        from macgpi.engine.template_manager import TemplateManager
        actual_template_mgr = TemplateManager(prompt_dir=temp_prompts_dir)
        mock_template_mgr.return_value = actual_template_mgr

        mock_agent_mgr.return_value = MagicMock()

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
        mock_parse_config.return_value = config

        output_dir = os.path.join(temp_dir, "output")

        # Should return False due to invalid phase
        result = macgpi(
            input="test",
            model_name="test-model",
            output_dir=output_dir,
            prompt_dir=temp_prompts_dir,
            phases_config=None
        )
        assert result is False

    @patch("macgpi.engine.main.AgentManager")
    @patch("macgpi.engine.main.TemplateManager")
    @patch("macgpi.engine.main.vllm_health")
    def test_macgpi_executes_phases_in_order(
        self,
        mock_health,
        mock_template_mgr,
        mock_agent_mgr,
        temp_prompts_dir,
        sample_output_dir,
        sample_phase_config
    ):
        """Test that phases are executed in correct order."""
        mock_health.return_value = True

        # Create actual template manager
        from macgpi.engine.template_manager import TemplateManager
        actual_template_mgr = TemplateManager(prompt_dir=temp_prompts_dir)
        mock_template_mgr.return_value = actual_template_mgr

        mock_agent = MagicMock()
        mock_agent_mgr.return_value = mock_agent

        result = macgpi(
            input="test",
            model_name="test-model",
            output_dir=sample_output_dir,
            prompt_dir=temp_prompts_dir,
            phases_config=sample_phase_config
        )

        # Verify agent was called
        assert result is True
        mock_agent.run.assert_called()


class TestMacgpiErrorHandling:
    """Tests for macgpi error handling."""

    @patch("macgpi.engine.main.vllm_health")
    def test_macgpi_handles_missing_config_gracefully(self, mock_health, temp_dir):
        """Test that macgpi handles missing config gracefully."""
        mock_health.return_value = True

        # Should not raise exception, but return False
        result = macgpi(
            input="test",
            model_name="test-model",
            output_dir=temp_dir,
            phases_config="/nonexistent/path/config.json"
        )

        assert result is False
