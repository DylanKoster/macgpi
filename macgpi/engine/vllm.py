import logging

import requests

logger = logging.getLogger(__name__)

def vllm_health(host: str, port: int) -> bool:
    '''
    Test the health of the vLLM server by sending a request to the health endpoint. Returns True if the server is
    healthy and reachable, False otherwise.
    '''
    url: str = f"http://{host}:{port}/health"
        
    logger.debug(f"Testing vLLM endpoint ({url})...")
    try:
        resp = requests.get(url, timeout=2)
        logger.debug(f"vLL response: {resp}")
        if 200 <= resp.status_code < 300:
            return True
    except Exception:
        logger.debug("Exception while testing vLLM endpoint", exc_info=True)
        pass

    return False