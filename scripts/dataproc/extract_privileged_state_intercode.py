import json
import argparse
import yaml
from typing import Dict, List
from intercode.envs import (
    BashEnv, PythonEnv, SqlEnv, CTFEnv, SWEEnv
)

def preprocess_sql(record: Dict) -> List:
    db = record["db"]
    return [f"use {db}"]

intercode_privileged_state_template = lambda task, gold_command, gold_obs: f"""
TASK: 
{task}
PRIVILEGED OPTIMAL ACTION:
{gold_command}
PRIVILEGED EXPECTED OBSERVATION:
{gold_obs}
"""

def main():
    parser = argparse.ArgumentParser(description="Extract privileged state from Intercode environment.")
    parser.add_argument('--config', type=str, required=True, help='Path to dataproc config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = config['extract_privileged_state_intercode']
    
    privleged_state_filename = config['privleged_state_filename']

    if config['env_type'] == "sql":            
        env = SqlEnv(image_name=config['env_image_name'], data_path=config['env_data_path'], verbose=config['env_verbose'], preprocess=preprocess_sql)
        # split in train and test
        all_env_idxs = range(len(env.data_loader))
        train_frac = 0.7
        split_idx = int(len(all_env_idxs) * train_frac)
        train_idxs = all_env_idxs[:split_idx]
        test_idxs = all_env_idxs[split_idx:]
        env_idxs = train_idxs 
    elif config['env_type'] == "bash":            
        env = BashEnv(image_name=config['env_image_name'], data_path=config['env_data_path'], verbose=config['env_verbose'], preprocess=config['env_preprocess'])
        # split in train and test
        all_env_idxs = range(len(env.data_loader))
        train_frac = 0.7
        split_idx = int(len(all_env_idxs) * train_frac)
        train_idxs = all_env_idxs[:split_idx]
        test_idxs = all_env_idxs[split_idx:]
        env_idxs = train_idxs 
    
    privileged_state = {}
    for env_idx in env_idxs:
        init_obs, _ = env.reset(env_idx)
        obs, _, _, _ = env.step(env.gold)
        _, reward, _, info = env.step("submit")
        privileged_state[env_idx] = intercode_privileged_state_template(init_obs, env.gold, obs)

    with open(privleged_state_filename, 'w') as file:
        json.dump(privileged_state, file, indent=4)

if __name__ == '__main__':
    main()