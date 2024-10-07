import json
import argparse
from tqdm import tqdm
import os
import yaml

parser = argparse.ArgumentParser(description='Collect webshop logs')
parser.add_argument('--config', type=str, required=True, help='Path to dataproc config file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
config = config['collect_logs_webshop']

# Load webshop training data
training_logs_filepath = config['training_logs_filepath']
with open(training_logs_filepath, 'r') as file:
    training_logs = [json.loads(line) for line in file]

# Create directory for saving logs
log_dir = config['log_dir']
os.makedirs(log_dir, exist_ok=True)

# Iterate over all logs
for log_idx, training_log in enumerate(tqdm(training_logs, desc="Processing logs")):
    trajectory = []
    for i in range(len(training_log['states'])):
        obs = training_log['states'][i]
        available_actions = training_log['available_actions'][i]
        if len(available_actions) == 0:
            # Assume that search is always available
            available_actions = ["search[search query]"]

        action_idx = training_log['action_idxs'][i]
        expert_action = training_log['actions'][i] if action_idx < 0 else available_actions[action_idx]
        trajectory.append({'observation': obs, 'action': expert_action, 'candidate_actions': available_actions})

    log_file_path = os.path.join(log_dir, f"{log_idx}.json")
    log = {'trajectory': trajectory}
    with open(log_file_path, 'w') as log_file:
        json.dump(log, log_file, indent=4)