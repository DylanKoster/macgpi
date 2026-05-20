import os
import json
import pytest
from macgpi.engine.phase_utils import (
    parse_phase_config,
    is_phase_dir,
    read_phase_inputs,
    is_finished_phase,
    get_next_phase,
    get_phase_prompts,
    validate_output_file
)


class TestParsePhaseConfig:
    """Tests for parse_phase_config function."""

    def test_parse_valid_config(self, sample_phase_config):
        """Test parsing a valid phase configuration file."""
        config: dict = parse_phase_config(sample_phase_config)

        assert "phases" in config
        assert "plan" in config["phases"]
        assert config["phases"]["plan"]["schema"] is True
        assert "implement" in config["phases"]
        assert config["phases"]["implement"]["schema"] is False

    def test_parse_config_structure(self, sample_phase_config):
        """Test that parsed config has expected structure."""
        config: dict = parse_phase_config(sample_phase_config)

        plan_phase: dict = config["phases"]["plan"]
        assert "inputs" in plan_phase
        assert "schema" in plan_phase
        assert "path" in plan_phase
        assert "output_file" in plan_phase
        assert "next" in plan_phase

    def test_parse_config_with_none(self):
        """Test parsing with None uses default config."""
        # This tests the default behavior when config_file is None
        # The function should look for macgpi/configs/macgpi_phases.json
        config: dict = parse_phase_config(None)
        assert isinstance(config, dict)
        assert "phases" in config
        assert "plan" in config["phases"]
        assert "implement" in config["phases"]
        assert "evaluate" in config["phases"]
        assert "revise" in config["phases"]


class TestIsPhaseDdir:
    """Tests for is_phase_dir function."""

    def test_valid_phase_dir_with_schema(self, temp_phase_dir):
        """Test that a valid phase directory with schema is recognized."""
        assert is_phase_dir(temp_phase_dir, schema_required=True)

    def test_valid_phase_dir_without_schema_requirement(self, temp_phase_dir):
        """Test that a phase dir without schema_required check returns True."""
        assert is_phase_dir(temp_phase_dir, schema_required=False)

    def test_invalid_dir_no_template(self, temp_dir):
        """Test that a dir without template.md is invalid."""
        phase_dir: str = os.path.join(temp_dir, "invalid_phase")
        os.makedirs(phase_dir)

        # Create only schema, no template
        with open(os.path.join(phase_dir, "schema.json"), "w") as f:
            json.dump({"type": "object"}, f)

        assert not is_phase_dir(phase_dir)

    def test_invalid_dir_no_schema_when_required(self, temp_dir):
        """Test that a dir without schema fails when schema_required=True."""
        phase_dir: str = os.path.join(temp_dir, "no_schema_phase")
        os.makedirs(phase_dir)

        # Create only template, no schema
        with open(os.path.join(phase_dir, "template.md"), "w") as f:
            f.write("# Template")

        assert not is_phase_dir(phase_dir, schema_required=True)

    def test_nonexistent_dir(self, temp_dir):
        """Test that nonexistent directory returns False."""
        nonexistent: str = os.path.join(temp_dir, "nonexistent")
        assert not is_phase_dir(nonexistent)


class TestReadPhaseInputs:
    """Tests for read_phase_inputs function."""

    def test_read_inputs(self, sample_output_dir):
        """Test reading multiple phase inputs."""
        inputs: dict = {
            "system_prd": "docs/project_description.md",
            "implementation_plan": "docs/plan.json"
        }

        result: dict = read_phase_inputs(inputs, sample_output_dir)

        assert "system_prd" in result
        assert "implementation_plan" in result
        assert len(result) == 3  # includes output_dir

    def test_read_inputs_includes_output_dir(self, sample_output_dir):
        """Test that output_dir is included in results."""
        inputs: dict = {}
        result: dict = read_phase_inputs(inputs, sample_output_dir)

        assert "output_dir" in result
        assert result["output_dir"] == sample_output_dir

    def test_read_nonexistent_input_raises_error(self, sample_output_dir):
        """Test that reading nonexistent input raises FileNotFoundError."""
        inputs: dict = {
            "missing": "docs/nonexistent.md"
        }

        with pytest.raises(FileNotFoundError):
            read_phase_inputs(inputs, sample_output_dir)


class TestIsFinishedPhase:
    """Tests for is_finished_phase function."""

    def test_finish_string(self):
        """Test that 'finish' string is recognized."""
        assert is_finished_phase("finish")

    def test_none_value(self):
        """Test that None is recognized as finished."""
        assert is_finished_phase(None)

    def test_regular_phase_not_finished(self):
        """Test that regular phase names are not finished."""
        assert not is_finished_phase("plan")
        assert not is_finished_phase("implement")
        assert not is_finished_phase("evaluate")

    def test_empty_string_not_finished(self):
        """Test that empty string is not finished."""
        assert not is_finished_phase("")


class TestGetNextPhase:
    """Tests for get_next_phase function."""

    def test_static_next_phase(self, sample_output_dir):
        """Test getting a statically defined next phase."""
        phase_config: dict = {
            "next": "implement"
        }

        result: str = get_next_phase(phase_config, sample_output_dir)
        assert result == "implement"

    def test_none_next_phase(self, sample_output_dir):
        """Test when next phase is not defined."""
        phase_config: dict = {}

        result: str = get_next_phase(phase_config, sample_output_dir)
        assert result is None

    def test_dynamic_next_phase(self, sample_output_dir):
        """Test getting dynamically determined next phase."""
        # Update the plan.json file to have "next" field
        plan_path: str = os.path.join(sample_output_dir, "docs", "plan.json")
        with open(plan_path, "w") as f:
            json.dump({"plan": "test plan", "next": "plan"}, f)

        phase_config: dict = {
            "next": "dynamic",
            "output_path": "docs/plan.json"
        }

        result: str = get_next_phase(phase_config, sample_output_dir)
        assert result == "plan"

    def test_dynamic_without_output_path(self, sample_output_dir):
        """Test dynamic next phase without output_path returns None."""
        phase_config: dict = {
            "next": "dynamic"
        }

        result: str = get_next_phase(phase_config, sample_output_dir)
        assert result is None


class TestGetPhasePrompts:
    """Tests for get_phase_prompts function."""

    def test_single_template_file(self, temp_phase_dir):
        """Test retrieving single template file."""
        prompts: list[str] = get_phase_prompts(temp_phase_dir)

        assert len(prompts) == 1
        assert prompts[0] == "template.md"

    def test_multiple_template_files(self, temp_dir):
        """Test retrieving multiple template files in sorted order."""
        phase_dir: str = os.path.join(temp_dir, "multi_template")
        os.makedirs(phase_dir)

        # Create multiple templates
        for i in [1, 3, 2]:
            with open(os.path.join(phase_dir, f"template_{i:02d}.md"), "w") as f:
                f.write(f"# Template {i}")

        prompts: list[str] = get_phase_prompts(phase_dir)

        assert len(prompts) == 3
        # Should be sorted
        assert prompts[0] == "template_01.md"
        assert prompts[1] == "template_02.md"
        assert prompts[2] == "template_03.md"

    def test_ignores_non_template_files(self, temp_dir):
        """Test that non-template files are ignored."""
        phase_dir: str = os.path.join(temp_dir, "mixed_files")
        os.makedirs(phase_dir)

        # Create templates and other files
        with open(os.path.join(phase_dir, "template.md"), "w") as f:
            f.write("# Template")
        with open(os.path.join(phase_dir, "schema.json"), "w") as f:
            json.dump({}, f)
        with open(os.path.join(phase_dir, "readme.txt"), "w") as f:
            f.write("readme")

        prompts: list[str] = get_phase_prompts(phase_dir)

        assert len(prompts) == 1
        assert prompts[0] == "template.md"

    def test_empty_phase_dir(self, temp_dir):
        """Test that empty directory returns empty list."""
        phase_dir: str = os.path.join(temp_dir, "empty")
        os.makedirs(phase_dir)

        prompts: list[str] = get_phase_prompts(phase_dir)
        assert prompts == []


class TestPhaseOutputValidation:
    """Tests for validate_output_file function."""

    def test_valid_output(self):
        """Test that valid output passes validation."""
        with open(os.path.join(os.path.dirname(__file__), "fixtures", "schema_1.json"), "r") as schema:
            schema_dict: dict = json.load(schema)

            for i in range(1, 3):
                path: str = os.path.join(os.path.dirname(__file__), "fixtures", f"schema_1_pass_{i}.json")

                with open(path, "r") as output:
                    output_dict: dict = json.load(output)
                    assert validate_output_file(output_dict, schema_dict)

    def test_invalid_output(self):
        """Test that invalid output fails validation."""
        with open(os.path.join(os.path.dirname(__file__), "fixtures", "schema_1.json"), "r") as schema:
            schema_dict: dict = json.load(schema)

            for i in range(1, 5):
                path: str = os.path.join(os.path.dirname(__file__), "fixtures", f"schema_1_fail_{i}.json")
                with open(path, "r") as output:
                    output_dict: dict = json.load(output)
                    assert not validate_output_file(output_dict, schema_dict)
