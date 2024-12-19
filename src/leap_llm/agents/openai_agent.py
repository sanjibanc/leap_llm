from jinja2 import Template
from leap_llm.agents.agent import Agent
from leap_llm.utils.openai import generate_from_openai_completion
from typing import Callable, List, Dict, Tuple, Any, Optional


class OpenAIAgent(Agent):
    """
    An agent that uses OpenAI's completion API to predict reasons and actions based on a task,
    observation, and candidate actions. 
    """
    def __init__(self, 
                 model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None) -> None:
        """
        Initializes the OpenAIAgent with a model identifier, a prompt template, verbosity, and optional debug settings.
        
        Args:
            model_id: The identifier for the OpenAI model 
            prompt_template_file: Path to a Jinja2 template file for generating prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            parse_reason_action_fn: A callable function that parses the model's response to extract reason and action.
        """
        super().__init__()
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

    def predict_reason_action(self, 
                              task: str, 
                              observation: Any, 
                              candidate_actions: List[str], 
                              reward: Optional[str] = "",
                              privileged_state: Any = None) -> Tuple[str, str]:
        """
        Predict reason and action given task, observation and candidate_actions. 

        Args:
            task: The task the agent is performing.
            observation: The current observation or input the agent is reacting to.
            candidate_actions: A list of possible actions the agent can take.
            reward: An optional reward signal from prior actions (default is an empty string).
        
        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """        
        observation_action_history = [{'observation': entry['observation'], 'action': entry['action']} for entry in self.observation_reason_action_history]

        input_data = {
            'mode': 'input',
            'task': task,
            'reward': reward,
            'privileged_state': privileged_state,
            'observation_action_history': observation_action_history,
            'observation': observation,
            'candidate_actions': candidate_actions
        }
        input_prompt = self.prompt_template.render(**input_data)

        messages = [
            {"role": "user", "content": input_prompt}
        ]
        response, _ = generate_from_openai_completion(messages=messages, model=self.model_id)

        reason, action = self.parse_reason_action_fn(response)
        if self.verbose > 0:
            if self.verbose > 1:
                print(f"\n OBSERVATION: {observation}")
                print(f"\n RESPONSE: {response}")
            print(f"\n OBSERVATION: {observation}")
            print(f"\n CANDIDATE ACTIONS: {candidate_actions}")
            print(f"\n REASON: {reason}")
            print(f"\n ACTION: {action}")
        
        if self.debug:
            human_input = input()
            if human_input != "c":
                action = human_input
                reason = "None"

        self.update_history(observation, reason, action)
        return reason, action
