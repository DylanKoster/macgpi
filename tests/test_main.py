import json
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

    def test_macgpi_fails_on_invalid_entry_phase(
        self, mocker, temp_prompts_dir, sample_output_dir
    ):
        """Test that macgpi fails on invalid entry phase."""
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)

        from macgpi.engine.template_manager import TemplateManager
        actual_template_mgr = TemplateManager(prompt_dir=temp_prompts_dir)
        mocker.patch("macgpi.engine.main.TemplateManager", return_value=actual_template_mgr)
        mocker.patch("macgpi.engine.main.AgentManager", return_value=mocker.MagicMock())

        # Create config with invalid entry
        config = {
            "entry": "non_existent",
            "phases": {
                "phase_1": {
                    "inputs": {
                        "system_prd": "docs/project_description.md"
                    },
                    "schema": False,
                    "path": "01_plan/",
                    "output_path": "docs/plan.json",
                    "next": "implement"
                }
            }
        }
        mocker.patch("macgpi.engine.main.parse_phase_config", return_value=config)

        # Should return False due to invalid entry
        macgpi: MACGPi = MACGPi(
            input_description="test",
            model_name="test-model",
            output_dir=sample_output_dir,
            prompt_dir=temp_prompts_dir,
            phases_config_file=None
        )
        result = macgpi.run()

        assert result is False


# ---------------------------------------------------------------------------
# Minimal JSON payloads that satisfy the real plan / evaluate schemas
# ---------------------------------------------------------------------------
_VALID_PLAN: dict = {
    "objectives": [{"id": "OBJ-001", "description": "Build a script", "priority": "high"}],
    "architecture": {"pattern": "script", "rationale": "Simplest structure"},
    "components": [{"name": "main", "responsibility": "Entry point"}],
    "dependencies": [],
    "implementation_tasks": [
        {"id": "TASK-001", "title": "Write script", "component": "main", "description": "Write it"}
    ],
    "quality_standards": {},
}

_VALID_EVAL_FINISH: dict = {
    "overall_rating": "pass",
    "summary": "Implementation meets all requirements.",
    "compliance": [{"requirement_id": "OBJ-001", "status": "met"}],
    "findings": [],
    "recommendations": [],
    "next": "finish",
}


def _write_json(path: str, content: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f)


def _mock_agent(mocker, output_dir: str):
    """Patch AgentManager with a mock whose run() writes required output files."""
    mock_am = mocker.MagicMock()
    mocker.patch("macgpi.engine.main.AgentManager", return_value=mock_am)

    call_index = {"n": 0}

    def agent_side_effect(prompt: str) -> None:
        n = call_index["n"]
        call_index["n"] += 1
        if n == 0:  # plan phase
            _write_json(
                os.path.join(output_dir, "docs", "implementation_plan.json"),
                _VALID_PLAN,
            )
        elif n == 3:  # evaluate phase (after 2 implement templates)
            _write_json(
                os.path.join(output_dir, "docs", "evaluation_report.json"),
                _VALID_EVAL_FINISH,
            )

    mock_am.run.side_effect = agent_side_effect
    return mock_am


@pytest.mark.integration
@pytest.mark.slow
class TestMacgpiIntegration:
    """
    Integration tests using the real prompt files, TemplateManager, and phase config.

    Only the network boundary (vllm_health, AgentManager) is mocked.
    The pipeline exercises real Jinja2 template rendering and real JSON schema validation.
    """

    def test_full_pipeline_completes_successfully(self, mocker, tmp_path):
        """Full pipeline returns True using real prompt files and real schema validation."""
        output_dir = str(tmp_path)
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)
        _mock_agent(mocker, output_dir)

        result = MACGPi(
            input_description="Build a hello world Python script.",
            model_name="test-model",
            output_dir=output_dir,
        ).run()

        assert result is True

    def test_agent_called_once_per_template(self, mocker, tmp_path):
        """Agent is called once per template: plan(1) + implement(2) + evaluate(1) = 4."""
        output_dir = str(tmp_path)
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)
        mock_am = _mock_agent(mocker, output_dir)

        MACGPi(
            input_description="Build a hello world Python script.",
            model_name="test-model",
            output_dir=output_dir,
        ).run()

        assert mock_am.run.call_count == 4

    def test_rendered_prompt_contains_input_description(self, mocker, tmp_path):
        """Plan prompt rendered by the real TemplateManager contains the input description."""
        output_dir = str(tmp_path)
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)
        mock_am = _mock_agent(mocker, output_dir)

        description = "Build a hello world Python script."
        MACGPi(
            input_description=description,
            model_name="test-model",
            output_dir=output_dir,
        ).run()

        plan_prompt: str = mock_am.run.call_args_list[0].args[0]
        assert description in plan_prompt

    def test_schema_validation_retries_phase_on_invalid_output(self, mocker, tmp_path):
        """Invalid plan output causes the phase to retry; valid output on the second attempt continues."""
        output_dir = str(tmp_path)
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)

        mock_am = mocker.MagicMock()
        mocker.patch("macgpi.engine.main.AgentManager", return_value=mock_am)

        call_index = {"n": 0}

        def agent_side_effect(prompt: str) -> None:
            n = call_index["n"]
            call_index["n"] += 1
            if n == 0:  # plan — first attempt: invalid JSON
                _write_json(
                    os.path.join(output_dir, "docs", "implementation_plan.json"),
                    {"invalid": True},
                )
            elif n == 1:  # plan — retry: valid JSON
                _write_json(
                    os.path.join(output_dir, "docs", "implementation_plan.json"),
                    _VALID_PLAN,
                )
            elif n == 4:  # evaluate (after retry + 2 implement templates)
                _write_json(
                    os.path.join(output_dir, "docs", "evaluation_report.json"),
                    _VALID_EVAL_FINISH,
                )

        mock_am.run.side_effect = agent_side_effect

        result = MACGPi(
            input_description="Build a hello world Python script.",
            model_name="test-model",
            output_dir=output_dir,
        ).run()

        assert result is True
        # plan×2 (retry) + implement×2 + evaluate×1 = 5 total agent calls
        assert mock_am.run.call_count == 5

    def test_output_files_written_by_pipeline(self, mocker, tmp_path):
        """Expected output files exist after a successful run."""
        output_dir = str(tmp_path)
        mocker.patch("macgpi.engine.main.vllm_health", return_value=True)
        _mock_agent(mocker, output_dir)

        MACGPi(
            input_description="Build a hello world Python script.",
            model_name="test-model",
            output_dir=output_dir,
        ).run()

        assert os.path.exists(os.path.join(output_dir, "docs", "project_description.md"))
        assert os.path.exists(os.path.join(output_dir, "docs", "implementation_plan.json"))
        assert os.path.exists(os.path.join(output_dir, "docs", "evaluation_report.json"))


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


@pytest.mark.unit
class TestMacgpiConfigValidation:
    """Focused tests for validating MACGPi phase config JSON schema rules."""

    def test_schema_true_requires_output_file(self, temp_dir):
        """A phase with schema=true must provide output_file."""
        config = {
            "entry": "plan",
            "phases": {
                "plan": {
                    "schema": True,
                    "path": "01_plan/",
                    "next": "finish"
                }
            }
        }

        macgpi = MACGPi(
            input_description="test",
            model_name="test-model",
            output_dir=temp_dir,
        )

        assert macgpi.validate_macgpi_config(config) is False

    def test_max_visits_exceeded_next_cannot_be_dynamic(self, temp_dir):
        """max_visits_exceeded_next must not be dynamic."""
        config = {
            "entry": "plan",
            "phases": {
                "plan": {
                    "schema": False,
                    "path": "01_plan/",
                    "next": "finish",
                    "max_visits": 3,
                    "max_visits_exceeded_next": "dynamic"
                }
            }
        }

        macgpi = MACGPi(
            input_description="test",
            model_name="test-model",
            output_dir=temp_dir,
        )

        assert macgpi.validate_macgpi_config(config) is False
