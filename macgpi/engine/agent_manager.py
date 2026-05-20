import os
import yaml

from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model
from minisweagent.environments.local import LocalEnvironment


class AgentManager:
    '''
    Manages the execution of agents for different phases of the pipeline. The AgentManager is responsible for ownership
    of a singleton MiniSWEAgent instance, and executing the phase prompts.
    '''

    def __init__(self, model_name: str, model_host: str = "localhost", model_port: int = 8000,
                 model_config_file: str | None = None, agent_config_file: str | None = None):
        self.load_model_config(model_config_file, model_name, model_host, model_port)
        self.load_agent_config(agent_config_file)

        self.agent = DefaultAgent(
            get_model(input_model_name=model_name, config=self.model_config),
            LocalEnvironment(),
            **self.agent_config
        )

    def run(self, prompt: str) -> dict:
        '''
        Run the agent on the given prompt and return the output.

        Parameters:
            prompt (str): The prompt to run the agent on.
        Returns:
            str: The output of the agent.
        '''
        return self.agent.run(prompt)

    def load_model_config(self, model_config_file: str | None, model_name: str, model_host: str,
                          model_port: int) -> None:
        '''
        Load a model configuration from the given file and update the agent's model configuration accordingly.

        Parameters:
            model_config_file (str | None): The path to the model configuration file. If None, a default path is used.
            model_name (str): The name of the model to use, which is added to the model configuration.
            model_host (str): The host of the model API, which is added to the model configuration.
            model_port (int): The port of the model API, which is added to the model configuration.
        '''
        model_config_extra: dict = {
            "model_name": model_name,
            "cost_tracking": "ignore_errors",
            "model_kwargs": {
                "api_base": f"http://{model_host}:{model_port}/v1",
                "custom_llm_provider": "hosted_vllm",
            },
            "api_key": "EMPTY",
        }

        if model_config_file is None:
            model_config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "model.config.yaml")

        with open(model_config_file, "r") as f:
            model_config: dict = yaml.safe_load(f)

        model_config.update(model_config_extra)
        self.model_config = model_config

    def load_agent_config(self, agent_config_file: str | None) -> None:
        '''
        Load an agent configuration from the given file and update the agent's configuration accordingly.

        Parameters:
            agent_config_file (str | None): The path to the agent configuration file. If None, a default path is used.
        '''
        if agent_config_file is None:
            agent_config_file = os.path.join(os.path.dirname(__file__), "..", "configs", "agent.config.yaml")

        with open(agent_config_file, "r") as f:
            agent_config: dict = yaml.safe_load(f)

        self.agent_config = agent_config
