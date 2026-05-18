import json
import logging
import os
import traceback

from macgpi.engine.agent_manager import AgentManager
from macgpi.engine.vllm import vllm_health
from macgpi.engine.template_manager import TemplateManager
from macgpi.engine.phase_utils import (get_next_phase, get_phase_prompts, is_finished_phase, is_phase_dir,
                                       parse_phase_config, read_phase_inputs)

logger = logging.getLogger(__name__)


class MACGPi:
    def __init__(self,
                 input: str,
                 model_name: str,
                 output_dir: str,
                 model_host: str = "localhost",
                 model_port: int = 8000,
                 prompt_dir: str | None = None,
                 model_config: dict | None = None,
                 agent_config: dict | None = None,
                 phases_config: dict | None = None,
                 ):
        '''
        Initializer for the MACGPi object.

        Parameters:
            input (str): The project/issue description to be processed by the pipeline.
            model_name (str): The name of the language model to use, as understood by mini-swe-agent.
            output_dir (str): The path to the target project directory where the output will be generated.
            model_host (str, optional): The host address of the vLLM server. Defaults to "localhost".
            model_port (int, optional): The port number of the vLLM server. Defaults to 8000.
            prompt_dir (str, optional): The directory under which the valid phase prompts and schemas are located. If
            None, it defaults to a "prompts" directory located in the parent directory of this module.
            model_config (dict, optional): Configuration file for the model. If None, the default mini-swe-agent
                configuration will be used.
            agent_config (dict, optional): Configuration file for the agent. If None, the default mini-swe-agent
                configuration will be used.
            phases_config (dict, optional): Configuration file for the MACGPi phases. If None, the default MACGPi
                configuration will be used.
        '''
        self.input = input
        self.model_name = model_name
        self.output_dir = output_dir
        self.model_host = model_host
        self.model_port = model_port
        self.prompt_dir = prompt_dir
        self.phases_config = phases_config

        self.phases_visits = {}

        # Instantiating managers
        try:
            self.templateManager: TemplateManager = TemplateManager(
                prompt_dir=self.prompt_dir)
            self.agentManager: AgentManager = AgentManager(self.model_name, model_host=model_host,
                                                           model_port=model_port, model_config_file=model_config,
                                                           agent_config_file=agent_config)
        except Exception:
            logger.error(f"An error occurred while initializing MACGPi:\nError {traceback.format_exc()}")

    def run(self) -> bool:
        '''
        This function orchestrates the execution of the pipeline executing the specified MACGPi phases on the (vLLM)
        hosted LLM.
        '''
        try:
            # -------------------------------------
            # vLLM host connection health check
            # -------------------------------------
            logger.debug(
                f"Attempting to connect to model server at {self.model_host}:{self.model_port}...")
            if not vllm_health(self.model_host, self.model_port):
                logger.error(f"Cannot reach vLLM server. Start a server on {self.model_host}:{self.model_port} or "
                             + "update the host "
                             + "and port parameters accordingly.")
                return False

            # -------------------------------------
            # Phase configuration parsing
            # -------------------------------------
            logger.debug("Parsing phase configuration...")

            macgpi_config: dict = parse_phase_config(self.phases_config)
            phases_config: dict = macgpi_config["phases"]
            self.phase_visits: dict = {
                phase_name: 0 for phase_name in phases_config.keys()}

            logger.debug("Phase configuration parsed successfully.")

            # -------------------------------------
            # Pre-execution validation check
            # -------------------------------------
            logger.debug("Executing pre-execution validation checks")

            if (self.templateManager is None or self.agentManager is None):
                logger.error("MACGPi was not initialized successfully, cannot run pipeline.")
                return False

            # Test whether all phases contain valid phase directories
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

            logger.debug("Pre-validation checks OK")

            # -------------------------------------
            # MACGPi phase execution
            # -------------------------------------
            logger.info("Starting MACGPi execution")

            # Write PRD to output dir
            logger.debug(
                f"Writing project description to output directory at {self.output_dir}...")
            project_description_path: str = os.path.join(
                self.output_dir, "docs", "project_description.md")

            if not os.path.exists(project_description_path):
                os.makedirs(os.path.dirname(
                    project_description_path), exist_ok=True)

            with open(project_description_path, "w") as f:
                f.write(self.input)

            logger.debug("Done!")

            phase: str = list(phases_config.keys())[0]
            while not is_finished_phase(phase):
                phase_config: dict = phases_config[phase]

                # Test whether max_visits has been exceeded for this phase, and if so, move to the
                # max_visits_exceeded_next
                max_visits: int = phase_config.get("max_visits", 1)
                if (self.phase_visits[phase] >= max_visits):
                    next_phase = phase_config.get("max_visits_exceeded_next", "finish")
                    logger.info(
                        f"Max visits exceeded for phase {phase}, moving to max_visits_exceeded_next phase "
                        + f"{next_phase}...")
                    phase = next_phase
                    continue

                logger.info(f"Starting phase {phase}...")

                self.phase_visits[phase] += 1

                logger.debug(
                    f"Phase config for phase {phase}:\n{json.dumps(phase_config, indent=4)}")

                output_path: str | None = self.output_dir
                if "output_path" in phase_config.keys():
                    output_path = self.output_dir + "/" + phase_config["output_path"]

                inputs: dict = read_phase_inputs(
                    phase_config["inputs"], self.output_dir)

                prompts: list[str] = get_phase_prompts(os.path.join(
                    self.templateManager.prompt_dir, phase_config["path"]))
                for prompt_file in prompts:
                    logger.info(
                        f"Running prompt {prompt_file} for phase {phase}...")
                    template_output: str = self.templateManager.render(phase_config["path"], template_file=prompt_file,
                                                                       output_path=output_path, **inputs)
                    self.agentManager.run(template_output)

                logger.info(f"Finished phase {phase}.")
                phase = get_next_phase(phase_config, output_dir=self.output_dir)

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
