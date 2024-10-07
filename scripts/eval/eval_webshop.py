import os
import time
import json
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

from leap_llm.envs.webshop import parse_args as webenv_args
from leap_llm.envs.webshop import WebEnv

from leap_llm.agents.openai_agent import OpenAIAgent
from leap_llm.agents.hf_agent import HFAgent
from leap_llm.agents.chat_agent import ChatAgent

from leap_llm.utils.parser import parse_reason_and_action_webshop, substitute_placeholders


def parse_and_load_config():
    """
    Parse command-line arguments and load the appropriate configuration file.
    """
    parser = argparse.ArgumentParser(description="Evaluate agent on Webshop")
    parser.add_argument('--eval_config', type=str, help='Path to the evaluation config file')
    parser.add_argument('--training_config', type=str, help='Path to the training config file')
    parser.add_argument('--iter', type=int, help='Iteration number')

    args = parser.parse_args()

    if args.eval_config:
        with open(args.eval_config, 'r') as f:
            config = yaml.safe_load(f)
    elif args.training_config and args.iter is not None:
        with open(args.training_config, 'r') as f:
            training_config = yaml.safe_load(f)
        config = training_config['rollout_student_trajectory']
        config = substitute_placeholders(config, '{iter}', str(args.iter))
    else:
        parser.error("You must provide either --eval_config or both --training_config and --iter.")
    
    return config


def initialize_agent(agent_config, config):
    """
    Initialize the appropriate agent based on the agent configuration.
    """
    common_params = {
        "model_id": agent_config["model_id"],
        "prompt_template_file": agent_config["prompt_template_file"],
        "verbose": config["verbose"],
        "debug": config["debug"],
        "parse_reason_action_fn": parse_reason_and_action_webshop,
    }

    if agent_config['type'] == "openai":
        return OpenAIAgent(**common_params)
    elif agent_config['type'] == "hf":
        return HFAgent(**common_params, max_length=6000)
    elif agent_config["type"] == "chat":
        generate_fn = None
        if agent_config['model_type'] == "gemini":
            from leap_llm.utils.gemini import generate_from_gemini
            generate_fn = generate_from_gemini
        elif agent_config['model_type'] == "anthropic":
            from leap_llm.utils.anthropic import generate_from_anthropic
            generate_fn = generate_from_anthropic

        return ChatAgent(**common_params, generate_fn=generate_fn)
    else:
        raise ValueError(f"Unsupported agent type: {agent_config['type']}")


def log_summary(dstdir, df_summary, summary_data):
    """
    Log the summary data for the current agent and environment.
    """
    summary_file_path = os.path.join(dstdir, "summary.csv")
    if os.path.exists(summary_file_path):
        df_summary = pd.read_csv(summary_file_path)
    
    df_summary = pd.concat([df_summary, pd.DataFrame([summary_data])], ignore_index=True)
    df_summary.to_csv(summary_file_path, index=False)
    print(f"Current summary:\n{df_summary}")


def main():
    config = parse_and_load_config()

    # Create directory for logs
    dstdir = os.path.join(config['logdir'], time.strftime('%Y%m%d-%H%M%S')) if not config['exact_path'] else config['logdir']
    os.makedirs(dstdir, exist_ok=True)
    with open(os.path.join(dstdir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)

    df_summary = pd.DataFrame()

    for agent_config in config['agents']:
        # Initialize agent
        agent = initialize_agent(agent_config, config)

        # Create directory for agent logs
        logdir = os.path.join(dstdir, agent_config['model_id']) if not config['exact_path'] else config['logdir']
        os.makedirs(logdir, exist_ok=True)

        # Load Webshop environment
        args = webenv_args()[0]
        env = WebEnv(args, split=config['eval_set'])

        # Get goal indexes for the environment
        env_idxs = env.goal_idxs[:min(config['max_env_idxs'], len(env.goal_idxs))] if config["max_env_idxs"] else env.goal_idxs

        # Iterate over goal indices
        for env_idx in tqdm(env_idxs, desc="env idxs"):
            obs, info = env.reset(idx=env_idx)

            if config["start_env_idx"] and (env_idx < config["start_env_idx"]):
                continue

            max_actions = config['max_actions']
            trajectory = []
            agent.reset()

            # Iterate over actions
            for _ in tqdm(range(max_actions), desc=f"Actions for env idx {env_idx}"):
                try:
                    reason, action = agent.predict_reason_action(task="", observation=obs, candidate_actions=env.get_valid_actions())
                except Exception as e:
                    print(f"Error occurred: {e}")
                    reward = 0
                    break

                # Log data from this step
                data = {
                    'observation': obs,
                    'candidate_actions': env.get_valid_actions(),
                    'reason': reason,
                    'action': action,
                    'score': reward
                }

                obs, reward, done, info = env.step(action)
                trajectory.append(data)

                if done:
                    break

            # Save trajectory log
            log_file_path = os.path.join(logdir, f"{env_idx}.json")
            log = {'env_idx': env_idx, 'trajectory': trajectory, 'info': info}
            with open(log_file_path, 'w') as log_file:
                json.dump(log, log_file, indent=4)

            # Log summary data
            summary_data = {
                'env_idx': env_idx,
                'model_id': agent_config['model_id'],
                'num_actions': len(trajectory),
                'score': reward
            }
            log_summary(dstdir, df_summary, summary_data)


if __name__ == "__main__":
    main()
