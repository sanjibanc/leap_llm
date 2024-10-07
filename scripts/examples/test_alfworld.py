import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic

# load config
config = generic.load_config()
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
num_games = env.num_games
env = env.init_env(batch_size=1)

# interact
obs, info = env.reset()
import pdb; pdb.set_trace()
while True:
    # Print observation
    print(f"Observation: {obs}")

    # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
    candidate_actions = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using candidate_actions

    # Print all possible actions
    print("Possible actions:")
    for i, cmd in enumerate(candidate_actions[0]):
        print(f"{i + 1}: {cmd}")
    
    # Print expert action
    print(f"[CHEATING] Expert action: {info['extra.expert_plan']}")

    # Get user input
    user_action_idx = int(input("Choose an action (enter the number): ")) - 1
    user_action = [candidate_actions[0][user_action_idx]]

    # # Get user input
    # user_action = [input("Type an action: ")]
    
    # step
    obss, scores, dones, info = env.step(user_action)
    obs, score, done = obss[0], scores[0], dones[0]

    if done:
        print(f"Final score: {score}")
        break