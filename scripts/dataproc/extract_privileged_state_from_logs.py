import os
import json
import argparse
import yaml
from jinja2 import Template
from tqdm import tqdm
from typing import List, Tuple, Dict
from leap_llm.utils.openai import generate_from_openai_completion

def extract_privileged_state_from_trajectory(
    trajectory: List[dict], 
    extract_privileged_state_template: Template
) -> Tuple[str, float]:
    """
    Extracts the privileged state from a given trajectory using a provided template.

    Args:
        trajectory (List[dict]): The input trajectory as a list of dictionaries, where each dictionary represents a step in the trajectory.
        extract_privileged_state_template (Template): A Jinja2 template used to format the input and system prompts.

    Returns:
        Tuple[str, float]: A tuple containing the extracted privileged state (as a string) and the cost of the API call.
    """    
    trajectory_str = json.dumps(trajectory)
    
    system_prompt = extract_privileged_state_template.render(system=True)
    input_prompt = extract_privileged_state_template.render(system=False, input=trajectory_str)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o"
    )

    privileged_state = response

    return privileged_state, cost

def process_logs(
    log_files: List[str], 
    extract_privileged_state_template: Template, 
    log_dir: str, 
    privleged_state_filename: str, 
    id_field_name: str
) -> None:
    """
    Processes log files to extract privileged state from trajectories and saves the results.

    Args:
        log_files (List[str]): List of log file names to process.
        extract_privileged_state_template (Template): A Jinja2 template used to format the input prompts for privileged state extraction.
        log_dir (str): Directory containing the log files.
        privleged_state_filename (str): Path to the file where privileged state results will be saved.
        id_field_name (str): The field name used to extract the ID from the log, which varies by dataset (e.g., 'gamefile', 'env_idx').

    Returns:
        None
    """
    cumulative_cost = 0
    pbar = tqdm(log_files, desc="Extract privileged state from log")

    privileged_state_dict = {}
    for i, filename in enumerate(pbar):
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r') as file:
            log = json.load(file)

        id, trajectory = log[id_field_name], log['trajectory']

        privileged_state, cost = extract_privileged_state_from_trajectory(trajectory, extract_privileged_state_template)
        privileged_state_dict[id] = privileged_state

        with open(privleged_state_filename, 'w') as file:
            json.dump(privileged_state_dict, file, indent=4)

        cumulative_cost += cost
        average_cost_per_file = cumulative_cost / (i + 1)
        projected_total_cost = average_cost_per_file * len(log_files)
        pbar.set_description(f"Cost: ${cumulative_cost:.2f}, Projected: ${projected_total_cost:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Extract privileged state from logs.")
    parser.add_argument('--config', type=str, required=True, help='Path to dataproc config file')
    parser.add_argument('--files', type=int, nargs='+', help="List of specific JSON files to process")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = config['extract_privileged_state_from_logs']
    
    
    prompt_template_file = config['prompt_template_file']
    log_dir = config['log_dir']
    privleged_state_filename = config['privleged_state_filename']
    id_field_name = config['id_field_name'] # For alfworld this is gamefile, for webshop this is env_idx

    with open(prompt_template_file, "r") as file:
        extract_privileged_state_template = Template(file.read())

    if args.files:
        log_files = [f"{file}.json" for file in args.files]
    else:
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]
        log_files.sort(key=lambda f: int(os.path.splitext(f)[0]))

    process_logs(log_files, extract_privileged_state_template, log_dir, privleged_state_filename, id_field_name)

if __name__ == '__main__':
    main()