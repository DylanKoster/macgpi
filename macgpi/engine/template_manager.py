import os

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template


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

    def render(self, phase: str, **kwargs) -> str:
        '''
        Render the template for the given phase with the provided keyword arguments.
        The template is searched in the prompts directory under a subdirectory named after the phase,
        and is expected to be named "template.md". The output schema for the phase is expected to be in the same
        subdirectory and named "schema.json".

        Parameters:
            phase (str): The name of the phase whose template should be rendered.
            **kwargs: Additional keyword arguments to pass to the template for rendering.
        '''
        # Use template name relative to the loader root
        template_path = f"{phase}/template.md"
        schema_path = os.path.join(self.prompt_dir, phase, f"schema.json")
        
        schema_format: str | None = None
        with open(schema_path, "r") as f:
            schema_format = f.read()
        
        if (schema_format == None):
            raise Exception(f"Failed to read schema for phase {phase} at path {schema_path}")

        template: Template = self.env.get_template(template_path)
        return template.render(schema_format=schema_format, **kwargs)