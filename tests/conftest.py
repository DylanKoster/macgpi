import os
import json
import tempfile
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def temp_phase_dir(temp_dir):
    """Create a temporary phase directory with template and schema files."""
    phase_dir = os.path.join(temp_dir, "test_phase")
    os.makedirs(phase_dir)

    # Create template file
    template_content = """# Test Template
This is a test template.
{{ test_input }}
"""
    with open(os.path.join(phase_dir, "template.md"), "w") as f:
        f.write(template_content)

    # Create schema file
    schema_content = {
        "type": "object",
        "properties": {
            "output": {"type": "string"}
        }
    }
    with open(os.path.join(phase_dir, "schema.json"), "w") as f:
        json.dump(schema_content, f)

    return phase_dir


@pytest.fixture
def temp_prompts_dir(temp_dir):
    """Create a temporary prompts directory with multiple phases."""
    prompts_dir = os.path.join(temp_dir, "prompts")
    os.makedirs(prompts_dir)

    # Create plan phase
    plan_phase = os.path.join(prompts_dir, "01_plan")
    os.makedirs(plan_phase)
    with open(os.path.join(plan_phase, "template.md"), "w") as f:
        f.write("# Plan phase\n{{ system_prd }}\n{{ schema_format }}")
    with open(os.path.join(plan_phase, "schema.json"), "w") as f:
        json.dump({"type": "object"}, f)

    # Create implement phase
    impl_phase = os.path.join(prompts_dir, "02_implement")
    os.makedirs(impl_phase)
    with open(os.path.join(impl_phase, "template_01.md"), "w") as f:
        f.write(
            "# Implement phase\n{{ system_prd }}\n{{ implementation_plan }}")
    with open(os.path.join(impl_phase, "template_02.md"), "w") as f:
        f.write(
            "# Implement phase\n{{ system_prd }}\n{{ implementation_plan }}")

    # Create evaluate phase
    eval_phase = os.path.join(prompts_dir, "03_evaluate")
    os.makedirs(eval_phase)
    with open(os.path.join(eval_phase, "template.md"), "w") as f:
        f.write("# Evaluate phase\n{{ schema_format }}")
    with open(os.path.join(eval_phase, "schema.json"), "w") as f:
        json.dump({"type": "object"}, f)

    # Create invalid phase 1
    eval_phase = os.path.join(prompts_dir, "04_invalid")
    os.makedirs(eval_phase)

    # Create  invalid phase 2
    eval_phase = os.path.join(prompts_dir, "05_invalid")
    os.makedirs(eval_phase)
    with open(os.path.join(eval_phase, "template.md"), "w") as f:
        f.write("# Evaluate phase")

    return prompts_dir


@pytest.fixture
def sample_phase_config(temp_dir):
    """Create a sample phase configuration file."""
    config = {
        "phases": {
            "plan": {
                "inputs": {
                    "system_prd": "docs/project_description.md"
                },
                "schema": True,
                "path": "01_plan/",
                "output_path": "docs/plan.json",
                "next": "implement"
            },
            "implement": {
                "inputs": {
                    "system_prd": "docs/project_description.md",
                    "implementation_plan": "docs/plan.json"
                },
                "schema": False,
                "path": "02_implement/",
                "next": "finish"
            }
        }
    }

    config_file = os.path.join(temp_dir, "phases.json")
    with open(config_file, "w") as f:
        json.dump(config, f)

    return config_file


@pytest.fixture
def sample_output_dir(temp_dir):
    """Create a sample output directory with test files."""
    output_dir = os.path.join(temp_dir, "output")
    docs_dir = os.path.join(output_dir, "docs")
    os.makedirs(docs_dir)

    # Create project description
    with open(os.path.join(docs_dir, "project_description.md"), "w") as f:
        f.write("# Test Project\nThis is a test project.")

    # Create plan output
    with open(os.path.join(docs_dir, "plan.json"), "w") as f:
        json.dump({"plan": "test plan"}, f)

    return output_dir
