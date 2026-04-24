import subprocess
import sys
import os
import time

class VLLMServer:
    def __init__(self, model_name: str, model_toolset: str = None, max_model_len: int=None):
        """
        General vLLM server class, starts and stops the vLLMServer.

        Args:
            model_name (str): The name (local or huggingface) of the model to use. If no local model is available (e.g. in the huggingface cache), the model is downloaded from huggingface.
            model_toolset (str): The tool call parser for the model. If None, the tool call parser will tried to be implied, if unsuccesfull, an error will occur. See https://docs.vllm.ai/en/latest/features/tool_calling/#automatic-function-calling 
            max_model_len (int): The maximum context length for the model. If None, the default context length of the model will be used.
        """
        self.process = None
        self.model_name = model_name
        self.model_toolset = model_toolset
        self.max_model_len = max_model_len
        self.tool_call_parser = None
        if (model_toolset == None):
            self.tool_call_parser = get_tool_call_parser(model_name)
            if (self.tool_call_parser == None):
                raise ValueError(f"Could not imply a tool call parser for model {model_name}. Please provide a tool call parser using the --model-toolset argument.")
        else:
            self.tool_call_parser = model_toolset

    def start_vllm(self, host: str="localhost", port: int=8000, tensor_parallel_size: int=1) -> bool:
        """
        Start the vLLM server.

        Args:
            host (str): The host address to bind the server to.
            port (int): The port number to listen on.
            tensor_parallel_size (int): The amount of multithreading to use.

        Returns:
            bool: True if the server started successfully, False otherwise.
        """

        try:
            curEpoch: int = time.time()
            logFile: str = f"{os.getcwd()}/vllm_logs/vLLMServer_log{curEpoch}.log"
            os.makedirs(os.path.dirname(logFile), exist_ok=True)

            cmd: list[str] = ["vllm", "server", "--tensor-parallel-size", str(tensor_parallel_size), "--host", host, "--port", str(port), "--enable-auto-tool-choice", "--tool-call-parser", self.tool_call_parser]
            if (self.max_model_len != None):
                cmd.extend(["--max-model-len", str(self.max_model_len)])

            # Start the vLLM server as a subprocess
            process = subprocess.Popen(
                cmd,
                stdout=open(logFile, "w"),
                stderr=open(logFile, "w"),
            )
            print(f"vLLM server started on {host}:{port} with PID {process.pid}")
            self.process = process
            return True
        except Exception as e:
            print(f"Failed to start vLLM server: {e}")
            return False


    def close(self):
        """
        Terminate the vLLM server if it is running.
        """
        if self.process:
            self.process.terminate()
            print(f"vLLM server with PID {self.process.pid} terminated.")
            self.process = None


def get_tool_call_parser(model_name: str) -> str | None:
    """
    Return the vLLM --tool-call-parser value for a given model name.

    Returns None when no known parser can be inferred.
    """
    name = model_name.lower()

    parser_rules = [
        # More specific rules first
        ("qwen3-coder", "qwen3_xml"),
        ("deepseek-v3.1", "deepseek_v31"),
        ("deepseek-v3", "deepseek_v3"),
        ("llama-3.1", "llama3_json"),
        ("llama-3.2", "llama3_json"),
        ("llama-4", "llama3_json"),
        ("glm-4.7", "glm47"),
        ("glm-4.5", "glm45"),
        ("kimi-k2", "kimi_k2"),

        # Broader rules
        ("mistral", "mistral"),
        ("mixtral", "mistral"),
        ("nous-hermes", "hermes"),
        ("hermes", "hermes"),
        ("gpt-oss", "openai"),
        ("openai", "openai"),
        ("xlam", "xlam"),
        ("internlm", "internlm"),
        ("jamba", "jamba"),
        ("minimax", "minimax_m1"),
        ("hunyuan", "hunyuan_a13b"),
        ("functiongemma", "functiongemma"),
        ("olmo", "olmo3"),
        ("gigachat", "gigachat3"),
    ]

    for pattern, parser in parser_rules:
        if pattern in name:
            return parser

    return None