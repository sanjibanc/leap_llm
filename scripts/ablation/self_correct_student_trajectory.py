import os
import json
import yaml
import argparse
import copy
import pandas as pd
import torch
from jinja2 import Template
from tqdm import tqdm
from typing import List, Tuple, Dict
from leap_llm.utils.parser import parse_reason_and_action_alfworld
from transformers import AutoTokenizer, AutoModelForCausalLM


def correct_student_trajectory(
    student_trajectory: List[dict], 
    privileged_state: str, 
    correction_oracle_template: Template, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    task: str
) -> List[dict]:
    """
    Corrects a student's trajectory based on privileged state using a pretrained language model.

    Args:
        student_trajectory (List[dict]): A list of dictionaries representing the student's trajectory.
        privileged_state (str): The privileged state used for corrections.
        correction_oracle_template (Template): A Jinja2 template for creating the input prompt for the model.
        model (AutoModelForCausalLM): A pretrained language model used for generating corrections.
        tokenizer (AutoTokenizer): Tokenizer associated with the pretrained model.
        task (str): The task identifier used in the input prompt.

    Returns:
        List[dict]: The corrected trajectory with additional fields such as corrected reasons, actions, and flags.
    """
    corrected_trajectory = []
    for tidx, traj_step in enumerate(tqdm(student_trajectory)):
        observation_action_history = [
            {
                "observation": tr["observation"],
                "action": tr["action"],
            }
            for tr in student_trajectory[0:tidx]
        ]

        input_data = {
            "mode": "input",
            "task": task,
            "observation_action_history": observation_action_history,
            "observation": traj_step["observation"],
            "candidate_actions": traj_step["candidate_actions"] if ("candidate_actions" in traj_step) else "",
            "privileged_state": privileged_state           
        }
        input_prompt = correction_oracle_template.render(**input_data)

        messages = [
            {"role": "user", "content": input_prompt}
        ]
        message = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tokenized_inputs = tokenizer(message, return_tensors="pt", padding=True, truncation=True, max_length=None).to(model.device)
        
        outputs = model.generate(
            tokenized_inputs["input_ids"],
            attention_mask=tokenized_inputs["attention_mask"],
            max_new_tokens=256,
            eos_token_id=[
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
            temperature=0.3,
            pad_token_id=tokenizer.eos_token_id
        )
        output = outputs[0]
        
        response = tokenizer.decode(output[tokenized_inputs["input_ids"].shape[-1] :],skip_special_tokens=True)
        
        corrected_reason, corrected_action = parse_reason_and_action_alfworld(response)
        
        corrected_traj_step = copy.deepcopy(traj_step)

        corrected_traj_step['original_observation'] = traj_step['observation']
        corrected_traj_step['candidate_actions'] = traj_step['candidate_actions']
        corrected_traj_step['original_reason'] = traj_step['reason']
        corrected_traj_step['original_action'] = traj_step['action']
        corrected_traj_step['corrected_reason'] = corrected_reason
        corrected_traj_step['corrected_action'] = corrected_action

        if (corrected_traj_step['corrected_action'] == corrected_traj_step['original_action']) or (corrected_traj_step['corrected_action'] not in corrected_traj_step['candidate_actions']):
            corrected_traj_step['is_corrected'] = False
        else:
            corrected_traj_step['is_corrected'] = True

        corrected_trajectory.append(corrected_traj_step)
    return corrected_trajectory

def process_logs(
    log_files: List[str], 
    correction_oracle_template: Template, 
    id_to_privileged_state: Dict[str, str], 
    log_dir: str, 
    output_log_dir: str, 
    id_field_name: str, 
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer
) -> None:
    """
    Processes multiple log files, applies trajectory correction, and saves the corrected logs.

    Args:
        log_files (List[str]): List of log filenames to process.
        correction_oracle_template (Template): Jinja2 template for the correction oracle.
        id_to_privileged_state (Dict[str, str]): Mapping of log IDs to their respective privileged states.
        log_dir (str): Directory containing the log files.
        output_log_dir (str): Directory to save the corrected log files.
        id_field_name (str): Field name used to extract the ID from the log.
        model (AutoModelForCausalLM): Pretrained language model for generating corrections.
        tokenizer (AutoTokenizer): Tokenizer associated with the model.

    Returns:
        None
    """
    for filename in tqdm(log_files, desc="Process log files"):
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r') as file:
            log = json.load(file)

        id, student_trajectory = log[id_field_name], log['trajectory']
        task = log['task']
        privileged_state = id_to_privileged_state[str(id)]
        corrected_trajectory = correct_student_trajectory(student_trajectory, privileged_state, correction_oracle_template, model, tokenizer, task)
        corrected_log = log
        corrected_log['trajectory'] = corrected_trajectory
        output_filepath = os.path.join(output_log_dir, filename)
        with open(output_filepath, 'w') as file:
            json.dump(corrected_log, file, indent=4)


def load_config(file_path: str, iter: int) -> Dict[str, any]:
    """
    Loads the configuration for the current iteration from a YAML file.

    Args:
        file_path (str): Path to the YAML configuration file.
        iter (int): Current iteration number to replace placeholders in the configuration.

    Returns:
        Dict[str, any]: The loaded and updated configuration.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    config = config['correct_student_trajectory']
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = value.replace('{iter}', str(iter))
    return config
    
def main():
    parser = argparse.ArgumentParser(description="Correct student trajectory.")
    parser.add_argument('--config', type=str, required=True, help='Path to dataproc config file')
    parser.add_argument('--iter', type=int, required=True, help='Iteration number')
    args = parser.parse_args()

    config = load_config(args.config, args.iter)

    log_dir = config['log_dir']
    output_log_dir = config['output_log_dir']
    privileged_state_file = config['privileged_state_file']
    prompt_file = config['prompt_file']
    id_field_name = config['id_field_name']
    correct_score_threshold = config['correct_score_threshold']

    os.makedirs(output_log_dir, exist_ok=True)
    with open(prompt_file, "r") as file:
        correction_oracle_template = Template(file.read())

    with open(privileged_state_file, "r") as file:
        id_to_privileged_state = json.load(file)

    df_summary = pd.read_csv(os.path.join(log_dir, 'summary.csv'))
    filtered_df = df_summary[df_summary['score'] <= correct_score_threshold] 
    filtered_df = filtered_df[filtered_df['env_idx'] >=979] 
    log_files = [f"{idx}.json" for idx in filtered_df['env_idx'].tolist()]

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], truncation=True, padding=True)
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    process_logs(log_files, correction_oracle_template, id_to_privileged_state, log_dir, output_log_dir, id_field_name, model, tokenizer)

if __name__ == '__main__':
    main()