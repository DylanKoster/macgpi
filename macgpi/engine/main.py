import logging
import os

from macgpi.engine.agent_manager import AgentManager
from macgpi.engine.vllm import VLLMServer
from macgpi.engine.template_manager import TemplateManager

logger = logging.getLogger(__name__)

def macgpi(
    input: str,
    model_name: str,
    output_dir: str,
    host_vllm: bool = True,
    model_host: str = "localhost",
    phases: list[str] | None = None,
    model_port: int = 8000,
    model_toolset: str | None = None,
    max_model_len: int | None = None,
    tensor_parallel_size: int = 1,
    prompt_dir: str | None = None,
    model_config: dict | None = None,
    agent_config: dict | None = None
):
    '''
    Entry point for the MACGPi pipeline. This function orchestrates the execution of the pipeline by managing the vLLM
    server and executing the specified phase MACGPi phases.

    Parameters:
        input (str): The project/issue description to be processed by the pipeline.
        model_name (str): The name of the language model to use, as understood by mini-swe-agent.
        output_dir (str): The path to the target project directory where the output will be generated.
        host_vllm (bool, optional): If True, MACGPi will host the vLLM server itself; if False, it is assumed that a
            vLLM server is already running and accessible at the specified model_host and model_port. Defaults to True.
        model_host (str, optional): The host address of the vLLM server. Defaults to "localhost".
        model_port (int, optional): The port number of the vLLM server. Defaults to 8000.
        model_toolset (str, optional  ): The tool call parser for the model. If None, the tool call parser will tried to
            be implied, if unsuccesfull, an error will occur. See 
            https://docs.vllm.ai/en/latest/features/tool_calling/#automatic-function-calling
        max_model_len (int, optional): The maximum context length for the model. If None, the default context length of
            the model will be used.
        tensor_parallel_size (int, optional): The amount of multithreading to use for the vLLM server. Defaults to 1.
        phases (list[str], optional): The phases of the pipeline to execute, in order. If not provided, all phases will
            be executed in the order they are found in the prompts directory.
        prompt_dir (str, optional): The directory under which the valid phase prompts and schemas are located. If None,
            it defaults to a "prompts" directory located in the parent directory of this module.
        model_config (dict, optional): Configuration file for the model. If None, the default mini-swe-agent
            configuration will be used.
        agent_config (dict, optional): Configuration file for the agent. If None, the default mini-swe-agent
            configuration will be used.
    '''
    try:
        vLLMServer: VLLMServer = VLLMServer(model_name, model_toolset=model_toolset, max_model_len=max_model_len)
        if host_vllm:
            if not vLLMServer.start_vllm(model_host, model_port, tensor_parallel_size=tensor_parallel_size):
                logger.error("Failed to start vLLM server. Please ensure that vLLM is installed and properly " + 
                             "configured.")
                return
            
        # Manager instantiations
        logger.debug("Instantiating managers.")
        
        templateManager: TemplateManager = TemplateManager(prompt_dir=prompt_dir)
        agentManager: AgentManager = AgentManager(model_name, model_host=model_host, model_port=model_port,
                                                  model_config_file=model_config, agent_config_file=agent_config)
        
        logger.debug("Done instantiating managers!")

        # If no phases are specified, default to all valid phase directories under the prompts directory
        if phases is None:
            logger.debug(f"No phases specified, inferring from prompt directory {templateManager.prompt_dir}")

            dirs: list[str] = os.listdir(templateManager.prompt_dir)
            phases = [name for name in dirs if is_phase_dir(os.path.join(templateManager.prompt_dir, name))]

            logger.debug(f"Found phases: {phases}")

        # Pre-execution validation check
        logger.debug("Executing pre-execution validation checks")
        
        for phase in phases:
            if (not is_phase_dir(os.path.join(templateManager.prompt_dir, phase))):
                logger.error(f"Phase {phase} is not a valid phase directory in {templateManager.prompt_dir}. " +
                             "Please ensure that it contains both a template.md file and a schema.json file.")
                return
            
        logger.debug("Pre-validation checks OK")

        # Phase execution
        logger.info("Starting MACGPi execution")

        for phase in phases:
            logger.info(f"Starting phase {phase}...")
            template_output: str = templateManager.render(phase, system_prd=input, output_dir=output_dir)
            agent_output: str = agentManager.run(template_output)

            logger.info(f"Finished phase {phase}.")

        logger.info(f"MACGPi execution finished, result copied to {output_dir}")
    except Exception as e:
        logger.error(f"An error occured while executing MACGPi:\nError {e}")
    finally:
        vLLMServer.close()


def is_phase_dir(path: str):
    '''
    Tests whether the given path is a valid phase directory. A directory is a valid phase directory iff it contains both
    a "template.md" file and a "schema.json" file.
    '''
    is_phase_dir: bool = (os.path.isdir(path) 
        and os.path.isfile(os.path.join(path, "template.md"))
        and os.path.isfile(os.path.join(path, "schema.json")))
    return is_phase_dir