import os
import logging

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

logger = logging.getLogger(__name__)


class TemplateManager:
    '''
    Manages the loading and rendering of templates for different phases of the pipeline. Templates are expected to be
    organized in subdirectories under a main prompts directory, which is provided in the constructor. Each
    subdirectory should be named after the phase it corresponds to. Each phase's subdirectory should contain a
    "template.md" file for the template and a "schema.json" file for the output schema.

    Parameters:
        prompt_dir (str, optional): The directory where the prompt templates are located. If not provided, it defaults
            to a "prompts" directory located in the parent directory of this module.
    '''

    def __init__(self, prompt_dir: str | None = None):
        # Resolve prompts directory relative to this module if not provided
        if prompt_dir is None:
            prompt_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "prompts"))

        self.prompt_dir = prompt_dir

        self.env: Environment = Environment(
            loader=FileSystemLoader(prompt_dir),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, phase_dir: str, template_file: str, **kwargs: object) -> str:
        '''
        Render the template for the given phase with the provided keyword arguments.
        The template is searched in the prompts directory under a subdirectory named after the phase,
        and is expected to be named "template.md". The output schema for the phase is expected to be in the same
        subdirectory and named "schema.json".

        Parameters:
            phase_dir (str): The path to the phase directory whose template should be rendered.
            template_file (str): The name of the template file to render.
            **kwargs: Additional keyword arguments to pass to the template for rendering.
        '''
        # Use template name relative to the loader root
        template_path = os.path.join(phase_dir, template_file)
        schema_path = os.path.join(self.prompt_dir, phase_dir, "schema.json")

        schema_format: str | None = None
        if os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                schema_format = f.read()

        logger.debug(
            f"Rendering template for phase {phase_dir} at path {template_path} "
            + (f"with schema at path {schema_path}" if schema_format is not None else ""))

        template: Template = self.env.get_template(template_path)

        if (schema_format is not None):
            kwargs.update({"schema_format": schema_format})

        render: str = template.render(**kwargs)

        logger.debug("Done rendering template!")
        return render
