from jinja2 import Template
from leap_llm.agents.agent import Agent
from typing import Callable, List, Tuple, Any, Optional

class ChatAgent(Agent):
    """
    A specialized agent class that utilizes a provided chat based generate_fn to predict reason and action. 
    """
    def __init__(self, model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 generate_fn: Callable[[List[dict], str], Tuple[str, Any]] = None, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None) -> None:
        """
        Initializes the ChatAgent with a model identifier, a prompt template, and optional verbosity/debug settings.
        
        Args:
            model_id: The identifier for the language model being used.
            prompt_template_file: The file path to a Jinja2 template for generating prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            generate_fn: A callable function for generating model responses based on messages.
            parse_reason_action_fn: A callable function for parsing the model response to extract reason and action.
        """
        super().__init__()
        self.model_id = model_id
        self.verbose = verbose
        self.debug = debug
        self.generate_fn = generate_fn
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

    def predict_reason_action(self, task: str, 
                              observation: Any, 
                              candidate_actions: List[str], 
                              reward: Optional[str] = "") -> Tuple[str, str]:
        """
        Predict reason and action given task, observation and candidate_actions. 
        
        Args:
            task: The task the agent is performing.
            observation: The current observation or input the agent is reacting to.
            candidate_actions: A list of possible actions that the agent can take.
            reward: An optional reward signal from prior actions, which may influence decision-making (default is an empty string).
        
        Returns:
            A tuple containing the predicted reason (str) and action (str).
        """
        observation_action_history = [{'observation': entry['observation'], 'action': entry['action']} for entry in self.observation_reason_action_history]

        input_data = {
            'mode': 'input',
            'task': task,
            'reward': reward,
            'observation_action_history': observation_action_history,
            'observation': observation,
            'candidate_actions': candidate_actions
        }
        input_prompt = self.prompt_template.render(**input_data)

        messages = [
            {"role": "user", "content": input_prompt}
        ]
        response, _ = self.generate_fn(messages=messages, model=self.model_id)

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
