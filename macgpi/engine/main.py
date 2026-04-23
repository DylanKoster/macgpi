import time

from macgpi.engine.vllm import VLLMServer


def macgpi(
    input: str,
    model_name: str,
    output_dir: str,
    host_vllm: bool = True,
    model_host: str = "localhost",
    model_port: int = 8000,
):
    try:
        vLLMServer: VLLMServer = VLLMServer()
        if host_vllm:
            vLLMServer.start_vllm(model_host, model_port)

    finally:
        vLLMServer.close()