import logging
import subprocess
import os
import time
import urllib.request
import urllib.error

from enum import Enum

logger = logging.getLogger(__name__)

class ServerStatus(Enum):
    STARTING=0
    ACTIVE=1
    CLOSED=2
    TIMEOUT=3
    ERR=4


class VLLMServer:
    def __init__(self, model_name: str, model_toolset: str = None, max_model_len: int=None):
        """
        General vLLM server class, starts and stops the vLLMServer.

        Args:
            model_name (str): The name (local or huggingface) of the model to use. If no local model is available (e.g.
                in the huggingface cache), the model is downloaded from huggingface.
            model_toolset (str): The tool call parser for the model. If None, the tool call parser will tried to be
                implied, if unsuccesfull, an error will occur. See 
                https://docs.vllm.ai/en/latest/features/tool_calling/#automatic-function-calling 
            max_model_len (int): The maximum context length for the model. If None, the default context length of the
                model will be used.
        """
        self.process = None
        self.model_name = model_name
        self.model_toolset = model_toolset
        self.max_model_len = max_model_len
        self.tool_call_parser = None
        if (model_toolset == None):
            self.tool_call_parser = get_tool_call_parser(model_name)
            if (self.tool_call_parser == None):
                raise ValueError(f"Could not imply a tool call parser for model {model_name}. Please provide a tool " +
                                 "call parser using the --model-toolset argument.")
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
            logFile: str = f"{os.getcwd()}/vllm_logs/vLLMServer_log_{curEpoch}.log"
            os.makedirs(os.path.dirname(logFile), exist_ok=True)

            cmd: list[str] = ["vllm", "server", "--tensor-parallel-size", str(tensor_parallel_size), "--host", host, 
                              "--port", str(port), "--enable-auto-tool-choice", "--tool-call-parser", 
                              self.tool_call_parser]
            if (self.max_model_len != None):
                cmd.extend(["--max-model-len", str(self.max_model_len)])

            # Start the vLLM server as a subprocess
            process = subprocess.Popen(
                cmd,
                stdout=open(logFile, "w"),
                stderr=open(logFile, "w"),
            )
            logger.info(f"vLLM server starting on {host}:{port} with PID {process.pid}")
            self.process = process

            status = self._wait_until_online(host, port, timeout_seconds=120)
            match status:
                case ServerStatus.ACTIVE:
                    logger.info(f"vLLM server started on {host}:{port} with PID {process.pid}")
                    return True
                case ServerStatus.TIMEOUT:
                    logger.error(f"vLLM server failed to become ready on {host}:{port} within 120 seconds.")
                    self.close()
                    return False
                case ServerStatus.ERR:
                    logger.error(f"vLLM server process exited with code {process.returncode} before becoming ready. "+ 
                                 f"See the vLLM log at {logFile} for more details.")    
                    self.close()
                    return False
                case _:
                    return False
        except Exception as e:
            logger.error(f"Exception occured while starting vLLM server: {e}")
            return False

    def _wait_until_online(self, host: str, port: int, timeout_seconds: int = 120) -> ServerStatus:
        """
        Poll common vLLM endpoints until the server is reachable or timeout is hit.
        """
        urls = [
            f"http://{host}:{port}/health",
            f"http://{host}:{port}/v1/models",
        ]
        logger.debug("Waiting for vLLM server to become active...")
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                return ServerStatus.ERR

            for url in urls:
                try:
                    with urllib.request.urlopen(url, timeout=2) as response:
                        if 200 <= response.status < 300:
                            return ServerStatus.ACTIVE
                except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
                    continue
            logger.debug(f"Failed polling {urls}, retrying in 2 seconds...")

            time.sleep(2)

        return ServerStatus.TIMEOUT


    def close(self):
        """
        Terminate the vLLM server if it is running.
        """
        if self.process:
            self.process.terminate()
            logger.info(f"vLLM server with PID {self.process.pid} terminated.")
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
        ("qwen2.5-coder", "hermes"),
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