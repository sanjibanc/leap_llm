from jinja2 import Template
from leap_llm.agents.agent import Agent
import sglang as sgl
from typing import Callable, List, Dict, Tuple, Any, Optional


# Refer: https://github.com/sgl-project/sglang/blob/main/examples/frontend_language/quick_start/local_example_chat.py
@sgl.function
def multi_turn_message(s, message_1):
    """
    A multi-turn message function using sglang to handle interactions between a user and an assistant.
    
    Args:
        s: The sglang state object.
        message_1: The initial message from the user.
    
    Returns:
        The state containing the assistant's response.
    """
    s += sgl.user(message_1)
    s += sgl.assistant(sgl.gen("answer_1", max_tokens=256, temperature=0.3))

class SGLangServerAgent(Agent):
    """
    An agent that uses sglang and interacts with an sglang server to predict reasons and actions
    based on task observations and candidate actions. 
    """

    def __init__(self, 
                 server_url: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None) -> None:
        """
        Initializes the SGLangServerAgent with the server URL, prompt template, verbosity, and optional debug settings.
        
        Args:
            server_url: The URL of the sglang server (e.g., "http://localhost:30000/").
            prompt_template_file: Path to a Jinja2 template file for generating prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            parse_reason_action_fn: A callable function that parses the generated response to extract reason and action.
        """
        super().__init__()
        self.server_url = server_url  # sglang server URL, e.g. http://localhost:30000/
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

        sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

    def predict_reason_action(self, 
                              task: str, 
                              observation: Any, 
                              candidate_actions: List[str], 
                              reward: Optional[str] = "") -> Tuple[str, str]:
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

        observation_action_history = [
            {"observation": entry["observation"], "action": entry["action"]}
            for entry in self.observation_reason_action_history
        ]
        input_data = {
            "mode": "input",
            "task": task,
            "reward": reward,
            "observation_action_history": observation_action_history,
            "observation": observation,
            "candidate_actions": candidate_actions,
        }

        input_prompt = self.prompt_template.render(**input_data)
        state = multi_turn_message.run(message_1=input_prompt)

        generated_text = state["answer_1"]
        reason, action = self.parse_reason_action_fn(generated_text)

        if self.verbose > 0:
            if self.verbose > 1:
                print(f"\n OBSERVATION: {observation}")
                print(f"\n RESPONSE: {generated_text}")
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