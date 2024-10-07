from leap_llm.envs.webshop import parse_args as webenv_args
from leap_llm.envs.webshop import WebEnv  

args = webenv_args()[0]
env = WebEnv(args, split='train')
print('env loaded')

while True:
    env_idx = input("Env Idx: ")
    env_idx = int(env_idx)
    obs, info = env.reset(env_idx)
    print(info['goal'])
    import pdb; pdb.set_trace()

    while True:
        print("******** OBSERVATION ********\n")
        print(obs)
        available_actions = env.get_valid_actions()
        print('******** AVAILABLE ACTIONS ********\n')
        print('\n'.join(available_actions))
        action = input("Action: ")
        obs, reward, done, info = env.step(action)
        if done:
            print(f"Reward: {reward}")
            break

