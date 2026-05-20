
import json
import logging
import os

import jsonschema

logger = logging.getLogger(__name__)


def parse_phase_config(config_file: str | None) -> dict | None:
    '''
    Parses the given phase configuration file and returns a dictionary mapping phase names to their configurations.
    The configuration file is expected to be in JSON format, with the following structure:
    {
        "phases": {
            "phase_name_1": {
                "path": "path/to/phase",
                "inputs": {
                    "input1": "path/to/input1",
                    "input2": "path/to/input2"
                },
                "output_path": "output_path",
                "schema:" true
            },
            "phase_name_2": {
                ...
            },
            ...
        }
    }
    '''
    if config_file is None:
        config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "macgpi_phases.json")

    try:
        with open(config_file, "r") as f:
            config: dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error occurred while parsing phase configuration file at {config_file}: {e}")
        return None

    return config


def is_phase_dir(path: str, schema_required: bool = True) -> bool:
    '''
    Tests whether the given path is a valid phase directory. A directory is a valid phase directory if it contains at
    least one file that starts with "template" and ends with ".md", which is used as the prompt template for the phase.
    If schema_required is true, the directory must also contain a "schema.json" file, which is used to validate the
    output of the phase. If schema_required is false, the presence of the "schema.json" file is not required for the
    directory to be considered a valid phase directory.
    '''
    is_phase_dir: bool = os.path.isdir(path)
    if not is_phase_dir:
        return False

    # Test if there is at least 1 template file.
    files: list[str] = os.listdir(path)
    prompt_files: list[str] = [file for file in files if is_template_file(file)]
    is_phase_dir = is_phase_dir and len(prompt_files) > 0

    if schema_required:
        is_phase_dir = is_phase_dir and os.path.isfile(os.path.join(path, "schema.json"))

    return is_phase_dir


def read_phase_inputs(inputs: dict, output_dir: str) -> dict:
    '''
    Reads the inputs for a phase from the given paths and returns a dictionary mapping input names to their contents.
    The input paths are expected to be relative to the output directory of the pipeline, which is provided in the
    constructor of the MACGPi class. The input paths are specified in the phase configuration file.

    Parameters:
        inputs (dict): A dictionary mapping input names to their relative paths.
        output_dir (str): The output directory of the pipeline, which is used as the base path for the input paths.
    Returns:
        dict: A dictionary mapping input names to their contents.
    '''
    input_contents: dict = {}
    input_contents["output_dir"] = output_dir

    for input_name, input_path in inputs.items():
        with open(os.path.join(output_dir, input_path), "r") as f:
            input_contents[input_name] = f.read()
    return input_contents


def is_finished_phase(phase: str | None) -> bool:
    '''
    Tests whether the given phase is the "finish" phase or None, which indicates that the pipeline has finished
    executing all phases.
    '''
    return phase is None or phase == "finish"


def get_next_phase(phase_config: dict, output_dir: str) -> str | None:
    '''
    Gets the next phase to execute from the given phase configuration. The next phase can either be specified statically
    in the "next" field of the phase configuration, or it can be specified dynamically in the output of the phase, in
    which case the "next" field should be set to "dynamic".
    '''
    next_phase: str | None = phase_config.get("next", None)

    if next_phase is None:
        return None

    if next_phase != "dynamic":
        return next_phase

    logger.debug(f"Next phase is dynamic. Attempting to read next phase from output directory at {output_dir}...")

    output_path: str | None = phase_config.get("output_path", None)
    if output_path is None:
        return None

    with open(os.path.join(output_dir, output_path), "r") as f:
        output_content: dict = json.load(f)

    logger.debug(
        f"Inferred phase: {output_content['next']} from output content at {output_path}: \n{json.dumps(output_content,
                                                                                                       indent=4)}")

    return str(output_content["next"])


def get_phase_prompts(phase_path: str) -> list[str]:
    '''
    Gets the prompts for a phase from the given phase path. The prompts are expected to be in a file called
    "template.md" in the phase directory. If the "template.md" file contains multiple prompts, their names should all
    start with template. The function returns a list of prompt file names, sorted in alphabetical order.
    '''
    files: list[str] = os.listdir(phase_path)

    prompt_files: list[str] = [file for file in files if is_template_file(file)]
    prompt_files.sort()
    return prompt_files


def validate_output_file(output_content: dict, schema: dict) -> bool:
    '''
    Validates the output of a phase against the given schema. The function returns true if the output is valid
    according to the schema, and false otherwise.
    '''
    try:
        jsonschema.validate(instance=output_content, schema=schema)
        return True
    except jsonschema.ValidationError as e:
        logger.error(f"Output validation error: {e}")
        return False


def is_template_file(file_name: str) -> bool:
    '''
    Tests whether the given file name is a valid template file name. A valid template file name starts with "template"
    and ends with ".md".
    '''
    return file_name.startswith("template") and file_name.endswith(".md")
