import json
import logging
import os
import traceback

from macgpi.engine.agent_manager import AgentManager
from macgpi.engine.vllm import vllm_health
from macgpi.engine.template_manager import TemplateManager
from macgpi.engine.phase_utils import (get_next_phase, get_phase_prompts, is_finished_phase, is_phase_dir,
                                       parse_phase_config, read_phase_inputs, validate_output_file)

logger = logging.getLogger(__name__)


class MACGPi:
    def __init__(self,
                 input_description: str,
                 model_name: str,
                 output_dir: str,
                 model_host: str = "localhost",
                 model_port: int = 8000,
                 prompt_dir: str | None = None,
                 model_config_file: str | None = None,
                 agent_config_file: str | None = None,
                 phases_config_file: str | None = None,
                 ):
        '''
        Initializer for the MACGPi object.

        Parameters:
            input_description (str): The project/issue description to be processed by the pipeline.
            model_name (str): The name of the language model to use, as understood by mini-swe-agent.
            output_dir (str): The path to the target project directory where the output will be generated.
            model_host (str, optional): The host address of the vLLM server. Defaults to "localhost".
            model_port (int, optional): The port number of the vLLM server. Defaults to 8000.
            prompt_dir (str, optional): The directory under which the valid phase prompts and schemas are located. If
            None, it defaults to a "prompts" directory located in the parent directory of this module.
            model_config_file (str, optional): Path to the configuration file for the model. If None, the default
                mini-swe-agent configuration will be used.
            agent_config_file (str, optional): Path to the configuration file for the agent. If None, the default
                mini-swe-agent configuration will be used.
            phases_config_file (str, optional): Path to the configuration file for the MACGPi phases. If None, the
                default MACGPi configuration will be used.
        '''
        self.input_description = input_description
        self.model_name = model_name
        self.output_dir = output_dir
        self.model_host = model_host
        self.model_port = model_port
        self.prompt_dir = prompt_dir
        self.phases_config_file = phases_config_file

        self.phases_visits = {}

        # Instantiating managers
        try:
            self.templateManager: TemplateManager = TemplateManager(
                prompt_dir=self.prompt_dir)
            self.agentManager: AgentManager = AgentManager(self.model_name, model_host=model_host,
                                                           model_port=model_port, model_config_file=model_config_file,
                                                           agent_config_file=agent_config_file)
        except Exception:
            logger.error(
                f"An error occurred while initializing MACGPi:\nError {traceback.format_exc()}")

    def run(self) -> bool:
        '''
        This function orchestrates the execution of the pipeline executing the specified MACGPi phases on the (vLLM)
        hosted LLM.
        '''
        try:
            # -------------------------------------
            # vLLM host connection health check
            # -------------------------------------
            if not self.test_vllm_connection():
                return False

            # -------------------------------------
            # Phase configuration parsing
            # -------------------------------------
            phases_config = self.load_phases_config()

            # -------------------------------------
            # Pre-execution validation check
            # -------------------------------------
            if not self.pre_execution_validation(phases_config):
                return False

            # -------------------------------------
            # MACGPi phase execution
            # -------------------------------------
            logger.info("Starting MACGPi execution")

            # Write PRD to output dir
            self.write_input_to_output_dir()

            phase: str = list(phases_config.keys())[0]
            while not is_finished_phase(phase):
                phase_config: dict = phases_config[phase]

                # Test whether max_visits has been exceeded for this phase, and if so, move to the
                # max_visits_exceeded_next
                max_visits: int = phase_config.get("max_visits", 1)
                if (self.phase_visits[phase] >= max_visits):
                    next_phase = phase_config.get(
                        "max_visits_exceeded_next", "finish")
                    logger.info(
                        f"Max visits exceeded for phase {phase}, moving to max_visits_exceeded_next phase "
                        + f"{next_phase}...")
                    phase = next_phase
                    continue

                logger.info(f"Starting phase {phase}...")

                self.phase_visits[phase] += 1

                output_file = None
                if "output_file" in phase_config.keys():
                    output_file = self.output_dir + \
                        "/" + phase_config["output_file"]

                inputs: dict = read_phase_inputs(
                    phase_config["inputs"], self.output_dir)

                prompts: list[str] = get_phase_prompts(os.path.join(
                    self.templateManager.prompt_dir, phase_config["path"]))
                for prompt_file in prompts:
                    logger.info(
                        f"Running prompt {prompt_file} for phase {phase}...")
                    template_output: str = self.templateManager.render(phase_config["path"], template_file=prompt_file,
                                                                       output_file=output_file, **inputs)
                    self.agentManager.run(template_output)

                # If the phase requires schema validation of the output, validate the output and re-run the phase if it
                # fails
                if phase_config.get("schema", False) and output_file is not None:
                    schema_path: str = os.path.join(self.templateManager.prompt_dir,
                                                    phase_config["path"], "schema.json")

                    if not self.validate_output(output_file, schema_path, phase):
                        self.phase_visits[phase] -= 1
                        continue

                logger.info(f"Finished phase {phase}.")
                phase = get_next_phase(
                    phase_config, output_dir=self.output_dir)

            logger.info(
                f"MACGPi execution finished, result copied to {self.output_dir}")
            return True
        except Exception:
            logger.error(
                f"An error occured while executing MACGPi:\nError {traceback.format_exc()}")
            return False

    def get_phase_visits(self) -> dict:
        '''
        Returns the number of visits for each phase.
        '''
        return self.phase_visits

    def test_vllm_connection(self) -> bool:
        '''
        Tests the connection to the vLLM server specified by the model_host and model_port parameters. Returns True if
        the connection test succeeds, and False if it fails.
        '''
        logger.debug(
            f"Attempting to connect to model server at {self.model_host}:{self.model_port}...")

        if not vllm_health(self.model_host, self.model_port):
            logger.error(f"Cannot reach vLLM server. Start a server on {self.model_host}:{self.model_port} or "
                         + "update the host "
                         + "and port parameters accordingly.")
            return False

        return True

    def validate_phases(self, phases_config: dict) -> bool:
        '''
        Validates that all phases specified in the phase configuration contain valid phase directories. A valid phase
        directory is a directory that contains at least a template.md file, and if the phase requires a schema, also
        contains a schema.json file. Returns True if all phases are valid, and False otherwise.
        '''
        for phase in phases_config.keys():
            logger.debug(f"Checking phase validity of phase \"{phase}\"")

            phase_config: dict = phases_config[phase]
            phase_path: str = os.path.join(
                self.templateManager.prompt_dir, phase_config["path"])
            schema_required: bool = phase_config["schema"]

            if (not is_phase_dir(phase_path, schema_required=schema_required)):
                logger.error(f"Phase {phase} is not a valid phase directory in {self.templateManager.prompt_dir}. "
                             + "Please ensure that it contains both a template.md file"
                             + f"{" and a schema.json file" if schema_required else ''}.")
                return False

        return True

    def validate_output(self, output_path: str, schema_path: str, phase: str) -> bool:
        '''
        Validates the output of a phase against the specified schema. Returns True if the output is valid, and False
        if it is not.
        '''
        with open(schema_path, "r") as schema_file, open(output_path, "r") as output_file:
            schema_dict: dict = json.load(schema_file)
            output_dict: dict = json.load(output_file)
            valid: bool = validate_output_file(output_dict, schema_dict)

            # Output not correct according to schema, re-run phase
            if not valid:
                logger.warning(
                    f"Output for phase {phase} did not validate against the schema. Re-running phase...")
                return False

        return True

    def load_phases_config(self) -> dict:
        '''
        Loads the phase configuration from the specified configuration file, or from the default configuration if no
        file is specified. Returns the loaded phase configuration as a dictionary.
        '''
        logger.debug("Parsing phase configuration...")

        macgpi_config: dict = parse_phase_config(self.phases_config_file)
        phases_config: dict = macgpi_config["phases"]
        self.phase_visits: dict = {
            phase_name: 0 for phase_name in phases_config.keys()}

        logger.debug("Phase configuration parsed successfully.")

        return phases_config

    def pre_execution_validation(self, phases_config: dict) -> bool:
        '''
        Executes validation checks before running the pipeline, including validating the phase configuration and
        testing if the template and agent managers were initialized successfully. Returns True if all validation checks
        pass, and False if any check fails.
        '''
        logger.debug("Executing pre-execution validation checks")

        if (self.templateManager is None or self.agentManager is None):
            logger.error(
                "MACGPi was not initialized successfully, cannot run pipeline.")
            return False

        # Test whether all phases contain valid phase directories
        if not self.validate_phases(phases_config):
            return False

        logger.debug("Pre-execution validation checks OK")
        return True

    def write_input_to_output_dir(self) -> None:
        '''
        Writes the project description to the output directory. This is useful for providing the project description as
        a reference for the prompts and agents during execution.
        '''
        logger.debug(
            f"Writing project description to output directory at {self.output_dir}...")

        project_description_path: str = os.path.join(
            self.output_dir, "docs", "project_description.md")
        os.makedirs(os.path.dirname(project_description_path), exist_ok=True)
        with open(project_description_path, "w") as f:
            f.write(self.input_description)

        logger.debug("Done!")
