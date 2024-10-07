import json
import argparse
import yaml
from leap_llm.envs.webshop import parse_args as webenv_args
from leap_llm.envs.webshop import WebEnv  

webshop_privileged_state_template = lambda goal: f"""
**Task** 
User instruction: {goal['instruction_text']}
**Attributes**
Product must have these attributes: {goal['attributes']}
**Price**
Product must have price less than: {goal['price_upper']}
**Options**
Product must have the following options selected: {goal['goal_options']}
**Example Product**
While many products may satisfy the conditons above, here's an example of a product that exactly satsifies the criteria:
{goal['name']}
"""

def main():
    parser = argparse.ArgumentParser(description="Extract privileged state from WebShop environment.")
    parser.add_argument('--config', type=str, required=True, help='Path to dataproc config file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = config['extract_privileged_state_webshop']
    
    privleged_state_filename = config['privleged_state_filename']

    webenv_arg = webenv_args()[0]
    env = WebEnv(webenv_arg, split='train')
    goals = env.env.server.goals

    privileged_state = {}
    for env_idx, goal in enumerate(goals):
        webshop_privileged_state = webshop_privileged_state_template(goal)
        privileged_state[env_idx] = webshop_privileged_state

    with open(privleged_state_filename, 'w') as file:
        json.dump(privileged_state, file, indent=4)

if __name__ == '__main__':
    main()