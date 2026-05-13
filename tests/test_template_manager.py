import pytest
from jinja2 import UndefinedError
from macgpi.engine.template_manager import TemplateManager


class TestTemplateManagerInit:
    """Tests for TemplateManager initialization."""

    def test_init_with_custom_dir(self, temp_prompts_dir):
        """Test initializing with custom prompt directory."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        assert manager.prompt_dir == temp_prompts_dir
        assert manager.env is not None

    def test_init_with_none_uses_default(self):
        """Test that None prompt_dir uses default directory."""
        manager = TemplateManager(prompt_dir=None)

        # Should resolve to macgpi/prompts
        assert manager.prompt_dir.endswith("prompts")
        assert manager.env is not None

    def test_jinja_environment_created(self, temp_prompts_dir):
        """Test that Jinja2 environment is properly created."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        assert manager.env is not None
        assert manager.env.undefined.__name__ == "StrictUndefined"
        assert manager.env.loader.searchpath == [temp_prompts_dir]


class TestTemplateManagerRender:
    """Tests for TemplateManager.render method."""

    def test_render_simple_template(self, temp_prompts_dir):
        """Test rendering a simple template."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        result = manager.render(
            "01_plan/",
            system_prd="Test project description",
            template_file="template.md",
        )

        assert "Test project description" in result
        assert "# Plan phase" in result

    def test_render_with_schema(self, temp_prompts_dir):
        """Test rendering template with schema available."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        # Provide required system_prd variable
        result = manager.render("01_plan/",
                                template_file="template.md", system_prd="Test project")

        # Verify template renders with schema available
        assert "# Plan phase" in result
        assert "Test project" in result
        assert '{"type": "object"}' in result

    def test_render_without_schema(self, temp_prompts_dir):
        """Test rendering template without schema file."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        # 02_implement doesn't have schema
        result = manager.render(
            "02_implement/",
            system_prd="Project description",
            template_file="template_01.md",
            implementation_plan="test plan"
        )

        assert "test plan" in result
        assert "# Implement phase" in result

    def test_render_with_multiple_variables(self, temp_prompts_dir):
        """Test rendering with multiple template variables."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        result = manager.render(
            "02_implement/",
            system_prd="Project description",
            template_file="template_02.md",
            implementation_plan="Implementation details"
        )

        assert "Project description" in result
        assert "Implementation details" in result

    def test_render_undefined_variable_raises_error(self, temp_prompts_dir):
        """Test that undefined variables raise error with StrictUndefined."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        # 01_plan expects system_prd to be provided
        with pytest.raises(UndefinedError):
            manager.render("01_plan/", template_file="template.md")

    def test_render_with_nonexistent_template(self, temp_prompts_dir):
        """Test rendering nonexistent template raises error."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        with pytest.raises(Exception):  # Jinja2 will raise TemplateNotFound
            manager.render("nonexistent/", template_file="template.md")

    def test_render_with_no_template_provided(self, temp_prompts_dir):
        """Test rendering nonexistent template raises error."""
        manager = TemplateManager(prompt_dir=temp_prompts_dir)

        with pytest.raises(TypeError):
            manager.render("nonexistent/")
