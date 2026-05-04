import os

from macgpi.engine.vllm import VLLMServer
from macgpi.engine.template_manager import TemplateManager

def macgpi(
    input: str,
    model_name: str,
    output_dir: str,
    host_vllm: bool = True,
    model_host: str = "localhost",
    phases: list[str] = None,
    model_port: int = 8000,
    model_toolset: str = None,
    max_model_len: int = None,
    tensor_parallel_size: int = 1,
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
    '''
    try:
        vLLMServer: VLLMServer = VLLMServer(model_name, model_toolset=model_toolset, max_model_len=max_model_len)
        if host_vllm:
            vLLMServer.start_vllm(model_host, model_port, tensor_parallel_size=tensor_parallel_size)

        manager: TemplateManager = TemplateManager()

        # If no phases are specified, default to all valid phase directories under the prompts directory
        if phases is None:
            dirs: list[str] = os.listdir(manager.prompt_dir)
            phases = [name for name in dirs if is_phase_dir(os.path.join(manager.prompt_dir, name))]

        # Pre-execution validation check
        for phase in phases:
            if (not is_phase_dir(os.path.join(manager.prompt_dir, phase))):
                raise ValueError(f"Phase {phase} is not a valid phase directory in {manager.prompt_dir}. Please " + 
                                 "ensure that it contains both a template.md file and a schema.json file.")

        # Phase execution
        for phase in phases:
            print(f"Starting phase {phase}...")
            template_output: str = manager.render(phase, system_prd=input, output_dir=output_dir)
            
            print(f"Finished phase {phase}.")
    except Exception as e:
        print(f"An error occured while executing phase {phase}:\nError {e}")
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