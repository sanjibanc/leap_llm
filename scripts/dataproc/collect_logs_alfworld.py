import sys
import argparse
import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from tqdm import tqdm
import json
import os
import yaml
from leap_llm.utils.alfworld import parse_gamefile, extract_task_from_observation

def preprocess_args():
    parser = argparse.ArgumentParser(description='Collect alfword logs')
    parser.add_argument("--alfworld_config", type=str, default="configs/env_config/alfworld_config.yaml", help="Path to the Alfred base config file")
    parser.add_argument('--config', type=str, required=True, help='Path to alfworld dataproc config file')
    args = parser.parse_args()
    sys.argv = [sys.argv[0], args.alfworld_config]
    return args

args = preprocess_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
config = config['collect_logs_alfworld']

# Load config
alfworld_config = generic.load_config()
env_type = alfworld_config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# Setup environment
env = getattr(environment, env_type)(alfworld_config, train_eval='train')
num_games = env.num_games
env = env.init_env(batch_size=1)

# Create directory for saving logs
log_dir = config['log_dir']
os.makedirs(log_dir, exist_ok=True)

# Iterate over all games
for env_idx in tqdm(range(num_games), desc="env_idxs"):
    obss, info = env.reset()
    obs = obss[0]
    task = extract_task_from_observation(obs)
    trajectory = []
    
    while True:
        # Log observation and action
        expert_action = info['extra.expert_plan'][0][0]
        trajectory.append({'observation': obs, 'action': expert_action, 'candidate_actions': info['admissible_commands'][0]})
        
        # Step with expert action
        obss, scores, dones, info = env.step([expert_action])
        obs, score, done = obss[0], scores[0], dones[0]

        if done:
            # Save game log as JSON file
            log_file_path = os.path.join(log_dir, f"{env_idx}.json")
            gamefile = parse_gamefile(env)
            log = {'gamefile': gamefile, 'task': task, 'trajectory': trajectory}
            with open(log_file_path, 'w') as log_file:
                json.dump(log, log_file, indent=4)
            break
