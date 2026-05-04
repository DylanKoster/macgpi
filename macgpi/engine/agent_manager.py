from minisweagent.agents.default import DefaultAgent
from minisweagent.models import get_model
from minisweagent.environments.local import LocalEnvironment

class AgentManager:
    '''
    Manages the execution of agents for different phases of the pipeline. The AgentManager is responsible for ownership
    of a singleton MiniSWEAgent instance, and executing the phase prompts.
    '''
    def __init__(self, model_name: str, model_config: dict, agent_config: dict):
        self.agent = DefaultAgent(
            get_model(input_model_name=model_name, config=model_config),
            LocalEnvironment(),
            **agent_config,
        )

    def run(self, prompt: str) -> str:
        '''
        Run the agent on the given prompt and return the output.

        Parameters:
            prompt (str): The prompt to run the agent on.
        Returns:
            str: The output of the agent.
        '''
        return self.agent.run(prompt)