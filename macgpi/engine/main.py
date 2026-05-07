from enum import Enum
import json
import logging
import os

from macgpi.engine.agent_manager import AgentManager
from macgpi.engine.vllm import vllm_health
from macgpi.engine.template_manager import TemplateManager

logger = logging.getLogger(__name__)

def macgpi(
    input: str,
    model_name: str,
    output_dir: str,
    model_host: str = "localhost",
    phases: list[str] | None = None,
    model_port: int = 8000,
    prompt_dir: str | None = None,
    model_config: dict | None = None,
    agent_config: dict | None = None,
    phases_config: dict | None = None,
):
    '''
    Entry point for the MACGPi pipeline. This function orchestrates the execution of the pipeline executing the
    specified MACGPi phases on the (vLLM) hosted LLM.

    Parameters:
        input (str): The project/issue description to be processed by the pipeline.
        model_name (str): The name of the language model to use, as understood by mini-swe-agent.
        output_dir (str): The path to the target project directory where the output will be generated.
        model_host (str, optional): The host address of the vLLM server. Defaults to "localhost".
        model_port (int, optional): The port number of the vLLM server. Defaults to 8000.
        phases (list[str], optional): The phases of the pipeline to execute, in order. If not provided, all phases will
            be executed in the order they are found in the prompts directory.
        prompt_dir (str, optional): The directory under which the valid phase prompts and schemas are located. If None,
            it defaults to a "prompts" directory located in the parent directory of this module.
        model_config (dict, optional): Configuration file for the model. If None, the default mini-swe-agent
            configuration will be used.
        agent_config (dict, optional): Configuration file for the agent. If None, the default mini-swe-agent
            configuration will be used.
        phases_config (dict, optional): Configuration file for the MACGPi phases. If None, the default MACGPi configuration
    '''
    try:
        if not vllm_health(model_host, model_port):
            logger.error(f"Cannot reach vLLM server. Start a server on {model_host}:{model_port} or update the host " +
                         "and port parameters accordingly.")
            return

        # Attempting vLLM host connection
        logger.debug(f"Attempting to connect to model server at {model_host}:{model_port}...")
        
        # Manager instantiations
        logger.debug("Instantiating managers.")
        
        templateManager: TemplateManager = TemplateManager(prompt_dir=prompt_dir)
        agentManager: AgentManager = AgentManager(model_name, model_host=model_host, model_port=model_port,
                                                  model_config_file=model_config, agent_config_file=agent_config)
        
        logger.debug("Done instantiating managers!")
        
        phases_config: dict = parse_phase_config(phases_config)
        
        # If no phases are specified, default to all phases in the config
        if phases is None:
            phases = list(phases_config.keys())

        # Pre-execution validation check
        logger.debug("Executing pre-execution validation checks")

        # Test whether all phases contain valid phase directories
        for phase in phases:
            logger.debug(f"Checking phase validity of phase \"{phase}\"")

            phase_config: dict = phases_config[phase]
            phase_path: str = os.path.join(templateManager.prompt_dir, phase_config["path"])
            schema_required: bool = phase_config["schema"]

            if (not is_phase_dir(phase_path, schema_required=schema_required)):
                logger.error(f"Phase {phase} is not a valid phase directory in {templateManager.prompt_dir}. " +
                             f"Please ensure that it contains both a template.md file" + 
                             f"{" and a schema.json file" if schema_required else ''}.")
                return
            
        logger.debug("Pre-validation checks OK")

        # Phase execution
        logger.info("Starting MACGPi execution")
        
        logger.debug(f"Writing project description to output directory at {output_dir}...")
        # Write PRD to output dir
        project_description_path: str = os.path.join(output_dir, "docs", "project_description.md")
        
        if not os.path.exists(project_description_path):
            os.makedirs(os.path.dirname(project_description_path), exist_ok=True)

        with open(project_description_path, "w") as f:
            f.write(input)

        logger.debug("Done!")

        for phase in phases:
            logger.info(f"Starting phase {phase}...")
            phase_config: dict = phases_config[phase]

            output_path: str | None = output_dir
            if "output_path" in phase_config.keys():
                output_path = output_dir + "/" + phase_config["output_path"]

            inputs: dict = read_phase_inputs(phase_config["inputs"], output_dir)
            
            template_output: str = templateManager.render(phase_config["path"], output_path=output_path, **inputs)
            agentManager.run(template_output)

            logger.info(f"Finished phase {phase}.")

        logger.info(f"MACGPi execution finished, result copied to {output_dir}")
    except Exception as e:
        logger.error(f"An error occured while executing MACGPi:\nError {e}")

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