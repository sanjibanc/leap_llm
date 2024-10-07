import json
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import random
import argparse
import yaml

from jinja2 import Template

def create_dataset_from_logs(
    logs_dir: str,
    logfiles: List[str],
    prompt_template_file: str,
    train_method: str,
    input_fields: Dict[str, str],
    output_fields: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Creates a dataset from a list of log files by applying the provided prompt templates.

    Args:
        logs_dir (str): Directory where log files are stored.
        logfiles (List[str]): List of log file names to process.
        prompt_template_file (str): Path to the prompt template file (Jinja2 template).
        train_method (str): Method of training, either 'sft' (supervised fine-tuning) or 'pref' (preference learning).
        input_fields (Dict[str, str]): Dictionary mapping input field names to keys in the trajectory.
        output_fields (Dict[str, str]): Dictionary mapping output field names to keys in the trajectory.

    Returns:
        List[Dict[str, Any]]: A list of processed datapoints in the form of dictionaries.
    """
    with open(prompt_template_file, "r") as file:
        prompt_template = Template(file.read())

    dataset = []
    for fidx, filename in enumerate(tqdm(logfiles)):
        with open(f"{logs_dir}/{filename}", "r") as file:
            log = json.load(file)

        if "trajectory" not in log:
            print(f"Skipping log {filename}. No 'trajectory' found in log")
            continue

        trajectory = log["trajectory"]

        for tidx, traj_step in enumerate(trajectory):
            # skip datapoint if is_corrected flag exists and is set to false
            if ("is_corrected" in traj_step) and (traj_step["is_corrected"] == False):
                continue

            observation_action_history = [
                {
                    "observation": tr[input_fields["observation"]],
                    "action": tr[input_fields["action"]],
                }
                for tr in trajectory[0:tidx]
            ]

            input_data = {
                "mode": "input",
                "observation": traj_step[input_fields["observation"]],
                "candidate_actions": (
                    traj_step[input_fields["candidate_actions"]]
                    if (("candidate_actions" in input_fields) and (input_fields["candidate_actions"] in traj_step))
                    else ""
                ),
                "task": (
                    log[input_fields["task"]]
                    if ("task" in input_fields and input_fields["task"])
                    else ""
                ),
                "observation_action_history": observation_action_history,
            }
            prompt = prompt_template.render(**input_data)

            if train_method == "sft":
                output_data = {
                    "mode": "output",
                    "reason": traj_step[output_fields["reason"]],
                    "action": traj_step[output_fields["action"]],
                }
                response = prompt_template.render(**output_data)
                datapoint = {
                    "prompt": [{"role": "user", "content": prompt}],
                    "response": [{"role": "assistant", "content": response}],
                }
            elif train_method == "pref":
                output_data = {
                    "mode": "output",
                    "reason": traj_step[output_fields["chosen_reason"]],
                    "action": traj_step[output_fields["chosen_action"]],
                }
                chosen_response = prompt_template.render(**output_data)
                output_data = {
                    "mode": "output",
                    "reason": traj_step[output_fields["rejected_reason"]],
                    "action": traj_step[output_fields["rejected_action"]],
                }
                rejected_response = prompt_template.render(**output_data)
                datapoint = {
                    "prompt": [{"role": "user", "content": prompt}],
                    "chosen": [{"role": "assistant", "content": chosen_response}],
                    "rejected": [{"role": "assistant", "content": rejected_response}],
                }
            else:
                raise ValueError(
                    f"Invalid training method: {train_method}. Must be 'sft' or 'pref'."
                )

            dataset.append(datapoint)

    return dataset


def save_train_test_dataset(
    logs_dir: str,
    prompt_template_file: str,
    data_dir: str,
    train_method: str,
    input_fields: Dict[str, str],
    output_fields: Dict[str, str],
    train_split: float,
    score_threshold: Optional[float],
    max_length_threshold: Optional[int],
) -> None:
    """
    Saves train and test datasets by splitting the logs based on the given parameters.

    Args:
        logs_dir (str): Directory where log files are stored.
        prompt_template_file (str): Path to the prompt template file (Jinja2 template).
        data_dir (str): Directory to save the train and test datasets.
        train_method (str): Method of training, either 'sft' (supervised fine-tuning) or 'pref' (preference learning).
        input_fields (Dict[str, str]): Dictionary mapping input field names to keys in the trajectory.
        output_fields (Dict[str, str]): Dictionary mapping output field names to keys in the trajectory.
        train_split (float): Ratio of train/test split.
        score_threshold (Optional[float]): Minimum score threshold for logs to be included.
        max_length_threshold (Optional[int]): Maximum length of trajectory for logs to be included.

    Returns:
        None
    """
    logfiles = [f for f in os.listdir(logs_dir) if f.endswith(".json")]

    if score_threshold is not None:
        logfiles_filtered = []
        for filename in logfiles:
            with open(f"{logs_dir}/{filename}", "r") as file:
                log = json.load(file)
            if log["trajectory"][-1]["score"] >= score_threshold:
                logfiles_filtered.append(filename)
        logfiles = logfiles_filtered
    
    if max_length_threshold is not None:
        logfiles_filtered = []
        for filename in logfiles:
            with open(f"{logs_dir}/{filename}", "r") as file:
                log = json.load(file)
            if len(log["trajectory"]) <= max_length_threshold:
                logfiles_filtered.append(filename)
        logfiles = logfiles_filtered

    print(logfiles)
    random.shuffle(logfiles)
    split_index = int(len(logfiles) * train_split)
    train_logfiles, test_logfiles = logfiles[:split_index], logfiles[split_index:]

    train_dataset = create_dataset_from_logs(
        logs_dir=logs_dir,
        logfiles=train_logfiles,
        prompt_template_file=prompt_template_file,
        train_method=train_method,
        input_fields=input_fields,
        output_fields=output_fields,
    )
    test_dataset = create_dataset_from_logs(
        logs_dir=logs_dir,
        logfiles=test_logfiles,
        prompt_template_file=prompt_template_file,
        train_method=train_method,
        input_fields=input_fields,
        output_fields=output_fields,
    )

    print(
        f"Saving (train, test) dataset of size ({len(train_dataset)}, {len(test_dataset)}) to {data_dir}/*.json"
    )

    os.makedirs(data_dir, exist_ok=True)

    with open(f"{data_dir}/train.json", "w") as file:
        json.dump(train_dataset, file, indent=4)
    with open(f"{data_dir}/test.json", "w") as file:
        json.dump(test_dataset, file, indent=4)


def replace_placeholders(d: Dict[str, Any], **kwargs) -> None:
    """
    Recursively replaces placeholders in a dictionary with given keyword arguments.

    Args:
        d (Dict[str, Any]): The dictionary with placeholders.
        kwargs: Keyword arguments to replace placeholders in the dictionary.

    Returns:
        None
    """
    for key, value in d.items():
        if isinstance(value, dict):
            replace_placeholders(value, **kwargs)
        elif isinstance(value, str):
            d[key] = value.format(**kwargs)


def load_config(file_path: str, iter: int, train_method: str) -> Dict[str, Any]:
    """
    Loads and processes the configuration file, replacing placeholders with iteration-specific values.

    Args:
        file_path (str): Path to the YAML configuration file.
        iter (int): Current iteration number.
        train_method (str): Training method, either 'sft' or 'pref'.

    Returns:
        Dict[str, Any]: The processed configuration dictionary.
    """
    with open(file_path, "r") as file:
        full_config = yaml.safe_load(file)
    if iter == 0:
        return full_config["create_training_data"]["base_sft"]
    else:
        config = full_config["create_training_data"][train_method]
        prev_iter = iter - 1
        replace_placeholders(config, iter=iter, prev_iter=prev_iter)
        return config


def main():
    parser = argparse.ArgumentParser(description="Create training data")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to dataproc config file"
    )
    parser.add_argument("--iter", type=int, required=True, help="Iteration number")
    parser.add_argument(
        "--train_method", type=str, required=True, default="sft",
        choices=["sft", "pref"], help=f"Training method",
    )
    parser.add_argument('--seed', type=int, default=42, help='Seed for shuffling (default: 42)')

    args = parser.parse_args()

    config = load_config(args.config, args.iter, args.train_method)
    print(config)

    random.seed(args.seed)
    
    save_train_test_dataset(
        logs_dir=config["logs_dir"],
        prompt_template_file=config["prompt_template_file"],
        data_dir=config["data_dir"],
        train_method=args.train_method,
        input_fields=config["input_fields"],
        output_fields=config["output_fields"],
        train_split=config["train_split"],
        score_threshold=config.get("score_threshold", None),  
        max_length_threshold=config.get("max_length_threshold", None)
    )


if __name__ == "__main__":
    main()
