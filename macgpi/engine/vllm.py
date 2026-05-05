import logging

import urllib

logger = logging.getLogger(__name__)

def vllm_health(host: str, port: int) -> bool:
    '''
    Test the health of the vLLM server by sending a request to the models endpoint. Returns True if the server is
    healthy and reachable, False otherwise.
    '''
    url: str = f"http://{host}:{port}/v1/models",
        
    logger.debug("Testing vLLM endpoint...")
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            if 200 <= response.status < 300:
                return True
        
    except Exception:
        pass

    return False