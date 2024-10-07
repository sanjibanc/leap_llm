import os
import json
import yaml
import argparse
import pandas as pd
from jinja2 import Template
from tqdm import tqdm
from typing import List, Tuple
from leap_llm.utils.openai import generate_from_openai_completion
from leap_llm.utils.parser import parse_json

def correct_student_trajectory(student_trajectory: List[dict], privileged_state: str, correction_oracle_template: Template) -> Tuple[Optional[Dict], float, str]:
    """
    Corrects the student trajectory using a correction oracle based on privileged state information.

    Args:
        student_trajectory (List[dict]): The student's trajectory consisting of observations, actions, and reasons.
        privileged_state (str): The privileged state information used for corrections.
        correction_oracle_template (Template): A Jinja2 template used to format the correction oracle prompts.

    Returns:
        Tuple[Optional[Dict], float, str]: A tuple containing the corrected trajectory and summary, the cost of the API call, 
        and the original response from the API. The corrected trajectory is returned as a dictionary. If the response fails, None is returned.
    """
    system_prompt = correction_oracle_template.render(system=True)
    input_prompt = correction_oracle_template.render(system=False, student_trajectory=student_trajectory, privileged_state=privileged_state)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o"
    )

    correction = parse_json(response=response)
    return correction, cost, response

def check_corrected_trajectory(corrected_trajectory: List[dict], original_trajectory: List[dict]) -> None:
    """
    Validates the corrected trajectory to ensure its length matches the original trajectory and that all required keys are present.

    Args:
        corrected_trajectory (List[dict]): The corrected trajectory returned by the correction oracle.
        original_trajectory (List[dict]): The original trajectory provided as input.

    Raises:
        ValueError: If the lengths of the trajectories don't match or if required keys are missing.
    """
    if len(corrected_trajectory) != len(original_trajectory):
        raise ValueError(f"Len of corrected trajectory {len(corrected_trajectory)} != len of original trajectory {len(original_trajectory)}")

    for idx, traj_step in enumerate(corrected_trajectory):
        for key in ["corrected_reason", "corrected_action"]: 
            if key not in traj_step:
                raise ValueError(f"Trajectory index {idx} does not have {key} ")
        
        original_traj_step = original_trajectory[idx]
        if 'original_observation' not in traj_step:
            traj_step['original_observation'] = original_traj_step['observation']
        if 'original_reason' not in traj_step:
            traj_step['original_reason'] = original_traj_step['reason']
        if 'original_action' not in traj_step:
            traj_step['original_action'] = original_traj_step['action']
        if 'candidate_actions' in original_traj_step:
            if 'candidate_actions' not in traj_step:
                traj_step['candidate_actions'] = original_traj_step['candidate_actions']
            if ('is_corrected' in traj_step) and (traj_step['corrected_action'] not in traj_step['candidate_actions']):
                traj_step['is_corrected'] = False

def process_logs(log_files: List[str], correction_oracle_template: Template, id_to_privileged_state: Dict[str, str], log_dir: str, output_log_dir: str, id_field_name: str) -> None:
    """
    Processes a set of log files, applying trajectory corrections and saving the results.

    Args:
        log_files (List[str]): List of log files to process.
        correction_oracle_template (Template): The Jinja2 template for the correction oracle.
        id_to_privileged_state (Dict[str, str]): A mapping from trajectory IDs to privileged state strings.
        log_dir (str): Directory containing the input log files.
        output_log_dir (str): Directory where corrected logs will be saved.
        id_field_name (str): Field name for the ID in the logs.
    """
    cumulative_cost = 0
    pbar = tqdm(log_files, desc="Process log files")

    for i, filename in enumerate(pbar):
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r') as file:
            log = json.load(file)

        id, original_trajectory = log[id_field_name], log['trajectory']

        student_trajectory = [{'timestep': i,
                               'observation': datapoint['observation'],
                               'candidate_actions': datapoint.get('candidate_actions', None),
                               'reason': datapoint['reason'],
                               'action': datapoint['action']} for i, datapoint in enumerate(original_trajectory)]

        privileged_state = id_to_privileged_state[str(id)]

        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                correction, cost, response = correct_student_trajectory(student_trajectory=student_trajectory, privileged_state=privileged_state, correction_oracle_template=correction_oracle_template)
                if correction is None:
                    raise ValueError("Failed to parse a trajectory from response")
                
                summary = correction['summary']
                corrected_trajectory = correction['trajectory']
                check_corrected_trajectory(corrected_trajectory, original_trajectory)
                break
            except Exception as e:
                retries += 1
                summary = None
                corrected_trajectory = None
                print(response)
                print(f"Error processing {filename}: {e}. Retrying {retries}/{max_retries}...")

        if corrected_trajectory is not None:
            corrected_log = log
            corrected_log['summary'] = summary
            corrected_log['trajectory'] = corrected_trajectory
            output_filepath = os.path.join(output_log_dir, filename)
            with open(output_filepath, 'w') as file:
                json.dump(corrected_log, file, indent=4)
        else:
            print(f"Failed to correctly generate trajectory for {filename}")

        cumulative_cost += cost
        average_cost_per_file = cumulative_cost / (i + 1)
        projected_total_cost = average_cost_per_file * len(log_files)
        pbar.set_description(f"Cost: ${cumulative_cost:.2f}, Projected: ${projected_total_cost:.2f}")

def load_config(file_path: str, iter: int) -> Dict:
    """
    Loads and updates the configuration for the current iteration.

    Args:
        file_path (str): Path to the YAML configuration file.
        iter (int): The current iteration number, used to update the config.

    Returns:
        Dict: The updated configuration dictionary.
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
    log_files = [f"{idx}.json" for idx in filtered_df['env_idx'].tolist()]

    process_logs(log_files, correction_oracle_template, id_to_privileged_state, log_dir, output_log_dir, id_field_name)

if __name__ == '__main__':
    main()