import subprocess
import sys
import os
import time

class VLLMServer:
    def __init__(self):
        self.process = None

    def start_vllm(self, host: str="localhost", port: int=8000) -> bool:
        """
        Start the vLLM server.

        Args:
            host (str): The host address to bind the server to.
            port (int): The port number to listen on.

        Returns:
            bool: True if the server started successfully, False otherwise.
        """

        try:
            curEpoch: int = time.time()
            logFile: str = f"{os.getcwd()}/vllm_logs/vLLMServer_log{curEpoch}.log"
            os.makedirs(os.path.dirname(logFile), exist_ok=True)

            # Start the vLLM server as a subprocess
            process = subprocess.Popen(
                [sys.executable, "-m", "vllm.server", "--host", host, "--port", str(port)],
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