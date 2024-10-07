import os
import json
import argparse
import yaml
from jinja2 import Template
from tqdm import tqdm
from typing import List, Tuple
from leap_llm.utils.openai import generate_from_openai_completion
from leap_llm.utils.parser import parse_json

def generate_reasoning_trajectory(trajectory: List[dict], reasoning_template: Template) -> Tuple[List[dict], float, str]:
    """
    Generates a reasoning-annotated trajectory using the OpenAI completion API.

    Args:
        trajectory (List[dict]): The input trajectory with observations and actions.
        reasoning_template (Template): Jinja2 template to format the prompts for reasoning generation.

    Returns:
        Tuple[List[dict], float, str]: A tuple containing the updated trajectory with reasoning,
        the cost of the OpenAI completion request, and the original response from the API.
    """
    system_prompt = reasoning_template.render(system=True)
    input_prompt = reasoning_template.render(system=False, input=trajectory)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": input_prompt}
    ]

    response, cost = generate_from_openai_completion(
        messages=messages, model="gpt-4o"
    )

    trajectory_w_reason = parse_json(response)    
    return trajectory_w_reason, cost, response

def process_logs(log_files: List[str], reasoning_template: Template, log_dir: str, output_log_dir: str, max_retries: int) -> None:
    """
    Processes log files to generate reasoning annotations and updates logs accordingly.

    Args:
        log_files (List[str]): List of log files to process.
        reasoning_template (Template): The Jinja2 template for formatting the reasoning prompts.
        log_dir (str): Directory containing the input log files.
        output_log_dir (str): Directory to save the processed log files with reasoning annotations.
        max_retries (int): Maximum number of retries allowed for processing each log in case of errors.
    """
    cumulative_cost = 0
    pbar = tqdm(log_files, desc="Generating reasoning on log")

    for i, filename in enumerate(pbar):
        filepath = os.path.join(log_dir, filename)
        with open(filepath, 'r') as file:
            log = json.load(file)

        trajectory = log["trajectory"]
        input_trajectory = [{'timestep': i,'observation': datapoint['observation'], 'action': datapoint['action']} for i, datapoint in enumerate(trajectory)]
        retries = 0
        success = False
        while retries < max_retries:
            try:
                trajectory_w_reason, cost, original_response = generate_reasoning_trajectory(input_trajectory, reasoning_template)
                if len(input_trajectory) != len(trajectory_w_reason):
                    raise ValueError(f"The input trajectory has {len(input_trajectory)} elements while trajectory_w_reason has {len(trajectory_w_reason)}.")
                success = True
                new_trajectory = [{**d1, 'reason': d2['reason']} for d1, d2 in zip(trajectory, trajectory_w_reason)]
                break
            except Exception as e:
                retries += 1
                print(f"Error processing {filename}: {e}. Retrying {retries}/{max_retries}...")

        if not success:
            print(f"Failed to process {filename} after {max_retries} retries. Skipping...")
            continue

        log["trajectory"] = new_trajectory
        output_filepath = os.path.join(output_log_dir, filename)
        with open(output_filepath, 'w') as file:
            json.dump(log, file, indent=4)

        cumulative_cost += cost
        average_cost_per_file = cumulative_cost / (i + 1)
        projected_total_cost = average_cost_per_file * len(log_files)
        pbar.set_description(f"Cost: ${cumulative_cost:.2f}, Projected: ${projected_total_cost:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Generate reasoning logs.")
    parser.add_argument('--config', type=str, required=True, help='Path to dataproc config file')
    parser.add_argument('--limit', type=int, default=None, help="Limit the number of files to process")
    parser.add_argument('--max_retries', type=int, default=2, help="Maximum number of retries for processing a log")
    parser.add_argument('--files', type=int, nargs='+', help="List of specific JSON files to process")
    parser.add_argument('--resume', action='store_true', help="Resume processing only unprocessed log files")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    prompt_file = config["annotate_reason"]["prompt_template_file"]
    log_dir = config["annotate_reason"]["log_dir"]
    output_log_dir = config["annotate_reason"]["output_log_dir"]

    # Ensure the output directory exists
    os.makedirs(output_log_dir, exist_ok=True)

    with open(prompt_file, "r") as file:
        reasoning_template = Template(file.read())

    if args.files:
        log_files = [f"{file}.json" for file in args.files]
    else:
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.json')]

        # Apply resume functionality
        if args.resume:
            processed_files = set(os.listdir(output_log_dir))
            log_files = [f for f in log_files if f not in processed_files]

        log_files.sort(key=lambda f: int(os.path.splitext(f)[0]))

        if args.limit:
            log_files = log_files[:args.limit]

    process_logs(log_files, reasoning_template, log_dir, output_log_dir, args.max_retries)

if __name__ == '__main__':
    main()