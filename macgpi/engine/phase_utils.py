
import json
import os


import os


def parse_phase_config(config_file: str | None) -> dict:
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
    
    with open(config_file, "r") as f:
        config: dict = json.load(f)
    
    return config["phases"]

def is_phase_dir(path: str, schema_required: bool = True) -> bool:
    '''
    Tests whether the given path is a valid phase directory. A directory is a valid phase directory iff it contains both
    a "template.md" file. If schema_required is true, it must also contain a "schema.json" file.
    '''
    is_phase_dir: bool = (os.path.isdir(path) 
        and os.path.isfile(os.path.join(path, "template.md")))
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

def is_finished_phase(phase: str) -> bool:
    '''
    Tests whether the given phase is the "finish" phase or None, which indicates that the pipeline has finished
    executing all phases.
    '''
    return phase is None or phase == "finish"