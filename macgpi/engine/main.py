import time

from macgpi.engine.vllm import VLLMServer


def macgpi(
    input: str,
    model_name: str,
    output_dir: str,
    host_vllm: bool = True,
    model_host: str = "localhost",
    model_port: int = 8000,
    model_toolset: str = None,
    max_model_len: int = None,
    tensor_parallel_size: int = 1,
):
    try:
        vLLMServer: VLLMServer = VLLMServer(model_name, model_toolset=model_toolset, max_model_len=max_model_len)
        if host_vllm:
            vLLMServer.start_vllm(model_host, model_port, tensor_parallel_size=tensor_parallel_size)

    finally:
        vLLMServer.close()