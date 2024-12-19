import os
import sys
import time
import json
import argparse
import yaml
import pandas as pd
from tqdm import tqdm

import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

from leap_llm.agents.openai_agent import OpenAIAgent
from leap_llm.agents.hf_agent import HFAgent
from leap_llm.agents.hf_spaces_agent import HFSpaceAgent
from leap_llm.agents.chat_agent import ChatAgent

from leap_llm.utils.alfworld import parse_gamefile, extract_task_from_observation
from leap_llm.utils.parser import parse_reason_and_action_alfworld, substitute_placeholders


def parse_and_load_config():
    """
    Parse command-line arguments and load the appropriate config file.
    """
    parser = argparse.ArgumentParser(description="Evaluate agent on ALFWorld")
    parser.add_argument("--alfworld_config", type=str, default="configs/env_config/alfworld_config.yaml", help="Path to the Alfred base config file")
    parser.add_argument("--eval_config", type=str, help="Path to the evaluation config file")
    parser.add_argument("--training_config", type=str, help="Path to the training config file")
    args = parser.parse_args()

    # Update sys.argv for compatibility with generic.load_config
    sys.argv = [sys.argv[0], args.alfworld_config]

    if args.eval_config:
        with open(args.eval_config, "r") as f:
            config = yaml.safe_load(f)
    elif args.training_config:
        with open(args.training_config, "r") as f:
            training_config = yaml.safe_load(f)
        config = training_config["collect_privileged_sft_demonstrations"]
    else:
        parser.error("You must provide either --eval_config or --training_config.")
    
    return config


def initialize_agent(agent_config, config):
    """
    Initialize the appropriate agent based on the agent configuration.
    """
    agent_type = agent_config["type"]
    common_params = {
        "model_id": agent_config["model_id"],
        "prompt_template_file": agent_config["prompt_template_file"],
        "verbose": config["verbose"],
        "debug": config["debug"],
        "parse_reason_action_fn": parse_reason_and_action_alfworld,
    }

    if agent_type == "openai":
        return OpenAIAgent(**common_params)
    elif agent_type == "hf":
        return HFAgent(**common_params, max_length=6000)
    elif agent_type == "hf_space":
        return HFSpaceAgent(space_id=agent_config["space_id"], **common_params)
    elif agent_type == "chat":
        generate_fn = None
        if agent_config['model_type'] == "gemini":
            from leap_llm.utils.gemini import generate_from_gemini
            generate_fn = generate_from_gemini
        elif agent_config['model_type'] == "anthropic":
            from leap_llm.utils.anthropic import generate_from_anthropic
            generate_fn = generate_from_anthropic

        return ChatAgent(**common_params, generate_fn=generate_fn)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def log_summary(dstdir, df_summary, summary_data):
    """
    Log the summary data for the current agent and environment.
    """
    summary_file_path = os.path.join(dstdir, "summary.csv")
    if os.path.exists(summary_file_path):
        df_summary = pd.read_csv(summary_file_path)
    
    df_summary = pd.concat([df_summary, pd.DataFrame([summary_data])], ignore_index=True)
    df_summary.to_csv(summary_file_path, index=False)
    print(f"Current summary:\n {df_summary}")


def main():
    config = parse_and_load_config()
    alfworld_config = generic.load_config()

    dstdir = os.path.join(config['logdir'], time.strftime('%Y%m%d-%H%M%S')) if not config["exact_path"] else config["logdir"]
    os.makedirs(dstdir, exist_ok=True)

    with open(os.path.join(dstdir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    privileged_state_file = config['privileged_state_file']
    with open(privileged_state_file, "r") as file:
        id_to_privileged_state = json.load(file)

    df_summary = pd.DataFrame()

    for agent_config in config["agents"]:
        agent = initialize_agent(agent_config, config)

        logdir = os.path.join(dstdir, agent_config['model_id']) if not config["exact_path"] else config["logdir"]
        os.makedirs(logdir, exist_ok=True)
        print(f"Creating log directory {logdir}")

        env_type = alfworld_config["env"]["type"]
        env = getattr(environment, env_type)(alfworld_config, train_eval=config["eval_set"])
        max_env_idxs = min(env.num_games, config["max_env_idxs"]) if config["max_env_idxs"] else env.num_games

        env = env.init_env(batch_size=1)

        for env_idx in tqdm(range(max_env_idxs), desc="env_idx"):
            obss, info = env.reset()

            if config["start_env_idx"] and (env_idx < config["start_env_idx"]):
                continue

            gamefile = parse_gamefile(env)
            max_actions = 20 #env.batch_env.envs[0].max_episode_steps
            obs = obss[0]
            task = extract_task_from_observation(obs)
            trajectory = []
            agent.reset()

            privileged_state = id_to_privileged_state[str(gamefile)]

            for _ in tqdm(range(max_actions), desc=f"Actions for env_idx: {env_idx + 1}"):
                try:
                    reason, action = agent.predict_reason_action(
                        task=task,
                        observation=obs,
                        candidate_actions=info["admissible_commands"][0],
                        privileged_state=privileged_state
                    )
                except Exception as e:
                    print(f"Error occurred: {e}")
                    break

                data = {
                    "observation": obs,
                    "candidate_actions": info["admissible_commands"][0],
                    "reason": reason,
                    "action": action,
                }

                obss, scores, dones, info = env.step([action])
                obs, score, done = obss[0], scores[0], dones[0]
                data["score"] = score
                trajectory.append(data)

                if done:
                    break

            log_file_path = os.path.join(logdir, f"{env_idx}.json")
            log = {"gamefile": gamefile, "trajectory": trajectory, "task": task}
            print(f"Saving log to {log_file_path}")
            with open(log_file_path, "w") as log_file:
                json.dump(log, log_file, indent=4)

            summary_data = {
                "env_idx": env_idx,
                "gamefile": gamefile,
                "model_id": agent_config["model_id"],
                "num_actions": len(trajectory),
                "score": score,
            }
            log_summary(dstdir, df_summary, summary_data)


if __name__ == "__main__":
    main()
