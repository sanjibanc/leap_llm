import os
import time
import json
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
from typing import Dict, List
from decimal import Decimal
from datetime import datetime

from leap_llm.agents.openai_agent import OpenAIAgent
from leap_llm.agents.hf_agent import HFAgent
from leap_llm.agents.hf_spaces_agent import HFSpaceAgent
from leap_llm.agents.sglang_server_agent import SGLangServerAgent

from intercode.envs import BashEnv, PythonEnv, SqlEnv

from leap_llm.utils.parser import (
    parse_reason_and_action_intercode,
    parse_reason_and_action_intercode_sql,
    substitute_placeholders,
)


def decimal_default(obj):
    """
    Custom JSON encoder for Decimal and datetime objects.
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return json.JSONEncoder().default(obj)


def preprocess_sql(record: Dict) -> List[str]:
    """
    Preprocess the SQL environment record by selecting the appropriate database.
    """
    db = record["db"]
    return [f"use {db}"]


def serialize_if_dict(input_data):
    """
    Serialize input data to a JSON string if it's a dictionary; return string otherwise.
    """
    if isinstance(input_data, dict):
        return json.dumps(input_data)
    elif isinstance(input_data, str):
        return input_data
    else:
        raise ValueError("Input must be either a dictionary or a string")


def parse_and_load_config():
    """
    Parse command-line arguments and load the appropriate configuration.
    """
    parser = argparse.ArgumentParser(description="Evaluate agent on Intercode")
    parser.add_argument('--eval_config', type=str, help='Path to the evaluation config file')
    parser.add_argument('--training_config', type=str, help='Path to the training config file')
    parser.add_argument('--collect_logs', action='store_true', help="Initial data collection for SFT")
    parser.add_argument('--iter', type=int, help='Iteration number')
    
    args = parser.parse_args()

    if args.eval_config:
        with open(args.eval_config, 'r') as f:
            config = yaml.safe_load(f)
    elif args.training_config and args.collect_logs:
        with open(args.training_config, 'r') as f:
            training_config = yaml.safe_load(f)
        config = training_config['collect_logs']
    elif args.training_config and args.iter is not None:
        with open(args.training_config, 'r') as f:
            training_config = yaml.safe_load(f)
        config = training_config['rollout_student_trajectory']
        config = substitute_placeholders(config, '{iter}', str(args.iter))
    else:
        parser.error("You must provide either --eval_config or both --training_config and --collect_logs or both --training_config and --iter.")
    
    return config


def initialize_environment(config):
    """
    Initialize the environment based on the configuration.
    """
    if config['env_type'] == "bash":
        env = BashEnv(
            image_name=config['env_image_name'],
            data_path=config['env_data_path'],
            verbose=config['env_verbose'],
            preprocess=config['env_preprocess']
        )
        parse_fn = parse_reason_and_action_intercode
    elif config['env_type'] == "sql":
        env = SqlEnv(
            image_name=config['env_image_name'],
            data_path=config['env_data_path'],
            verbose=config['env_verbose'],
            preprocess=preprocess_sql
        )
        parse_fn = parse_reason_and_action_intercode_sql
    elif config['env_type'] == "python":
        env = PythonEnv(
            image_name=config['env_image_name'],
            data_path=config['env_data_path'],
            verbose=config['env_verbose'],
            preprocess=None,
            is_agent=True
        )
        parse_fn = parse_reason_and_action_intercode
    else:
        raise ValueError(f"Unsupported environment type: {config['env_type']}")
    
    return env, parse_fn


def initialize_agent(agent_config, config, parse_fn):
    """
    Initialize the appropriate agent based on the agent configuration.
    """
    common_params = {
        "model_id": agent_config["model_id"],
        "prompt_template_file": agent_config["prompt_template_file"],
        "verbose": config["verbose"],
        "debug": config["debug"],
        "parse_reason_action_fn": parse_fn
    }

    if agent_config['type'] == "openai":
        return OpenAIAgent(**common_params)
    elif agent_config['type'] == "hf":
        return HFAgent(**common_params, max_length=6000)
    elif agent_config['type'] == "hf_space":
        return HFSpaceAgent(space_id=agent_config["space_id"], **common_params)
    elif agent_config['type'] == "sglang_server":
        return SGLangServerAgent(server_url=agent_config["server_url"], **common_params)
    else:
        raise ValueError(f"Unsupported agent type: {agent_config['type']}")


def log_summary(df_summary, summary_data, dstdir):
    """
    Update and save the summary file.
    """
    summary_file_path = os.path.join(dstdir, "summary.csv")
    if os.path.exists(summary_file_path):
        df_summary = pd.read_csv(summary_file_path)
    
    df_summary = pd.concat([df_summary, pd.DataFrame([summary_data])], ignore_index=True)
    df_summary.to_csv(summary_file_path, index=False)
    print(f"Current summary:\n {df_summary}")


def main():
    config = parse_and_load_config()
    
    dstdir = os.path.join(config['logdir'], time.strftime('%Y%m%d-%H%M%S')) if not config['exact_path'] else config['logdir']
    os.makedirs(dstdir, exist_ok=True)
    
    with open(os.path.join(dstdir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    df_summary = pd.DataFrame()

    env, parse_fn = initialize_environment(config)

    for agent_config in config['agents']:
        agent = initialize_agent(agent_config, config, parse_fn)
        
        logdir = os.path.join(dstdir, agent_config['model_id']) if not config['exact_path'] else config['logdir']
        os.makedirs(logdir, exist_ok=True)

        env_idxs = list(range(len(env.data_loader)))
        env_idxs = env_idxs[:min(config['max_env_idxs'], len(env_idxs))] if config["max_env_idxs"] else env_idxs

        for env_idx in tqdm(env_idxs, desc="env idxs"):
            while True:
                try:
                    obs, info = env.reset(env_idx)
                    break
                except Exception as e:
                    print(f"Exception: {e}. Retrying environment reset.")

            reward = 0
            task = obs
            max_actions = config['max_actions']
            trajectory = []
            agent.reset()

            for _ in tqdm(range(max_actions), desc=f"Actions for env idx {env_idx}"):
                try:
                    reason, action = agent.predict_reason_action(task=task, observation=obs, candidate_actions="", reward=reward)
                except Exception as e:
                    break

                data = {'observation': obs, 'reason': reason, 'action': action}

                while True:
                    try:
                        obs, reward, done, info = env.step(action)
                        break
                    except Exception as e:
                        print(f"Exception: {e}. Retrying step.")

                data['score'] = reward
                trajectory.append(data)

                if done:
                    break

            log_file_path = os.path.join(logdir, f"{env_idx}.json")
            log = {'env_idx': env_idx, 'trajectory': trajectory, 'task': task}

            with open(log_file_path, 'w') as log_file:
                json.dump(log, log_file, indent=4, default=decimal_default)

            summary_data = {'env_idx': env_idx, 'model_id': agent_config['model_id'], 'num_actions': len(trajectory), 'score': reward}
            log_summary(df_summary, summary_data, dstdir)


if __name__ == "__main__":
    main()
