import torch
from jinja2 import Template
from leap_llm.agents.agent import Agent
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Callable, List, Tuple, Any, Optional

class HFAgent(Agent):
    """
    A Hugging Face-based agent class that utilizes a pre-trained language model to predict reasons and actions
    from observations and candidate actions. 
    """
    def __init__(self, 
                 model_id: str, 
                 prompt_template_file: str, 
                 verbose: int = 0, 
                 debug: bool = False, 
                 parse_reason_action_fn: Callable[[str], Tuple[str, str]] = None, 
                 max_length: Optional[int] = None) -> None:
        """
        Initializes the HFAgent with a pre-trained language model, tokenizer, and a prompt template.

        Args:
            model_id: The identifier for the Hugging Face model.
            prompt_template_file: Path to a Jinja2 template file used to create input prompts.
            verbose: An optional flag (int) for verbosity level (default is 0).
            debug: A flag for enabling debug mode, where user input can override actions (default is False).
            parse_reason_action_fn: A callable function that parses the model's response to extract reason and action.
            max_length: An optional maximum length for tokenization (default is None).
        """
        super().__init__()

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model = model
        self.verbose = verbose
        self.debug = debug
        self.parse_reason_action_fn = parse_reason_action_fn
        self.max_length = max_length  
        tokenizer = AutoTokenizer.from_pretrained(model_id, truncation=True, padding=True)
        tokenizer.truncation_side = "left"
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        with open(prompt_template_file, "r") as file:
            self.prompt_template = Template(file.read())

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
            candidate_actions: A list of possible actions that the agent can take.
            reward: An optional reward signal from prior actions (default is an empty string).
        
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
        message = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized_inputs = self.tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(self.model.device)
        
        outputs = self.model.generate(
            tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            max_new_tokens=256,
            eos_token_id=[
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            temperature=0.3,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output = outputs[0]
        
        response = self.tokenizer.decode(output[tokenized_inputs["input_ids"].shape[-1] :],skip_special_tokens=True)

        reason, action = self.parse_reason_action_fn(response)
        if self.verbose > 0:
            if self.verbose > 1:
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
