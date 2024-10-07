from intercode.envs import (
    BashEnv, PythonEnv, SqlEnv, CTFEnv, SWEEnv
)
from typing import Dict, List

def preprocess_sql(record: Dict) -> List:
    db = record["db"]
    return [f"use {db}"]
    
if __name__ == '__main__':
    #env = BashEnv(image_name="intercode-nl2bash", data_path="env_assets/intercode/data/nl2bash/nl2bash_fs_1.json", verbose=True, preprocess=None)
    #env = SqlEnv(image_name="docker-env-sql", data_path="env_assets/intercode/data/sql/spider/ic_spider_dev.json", verbose=True, preprocess=preprocess_sql)
    env = PythonEnv(image_name="intercode-python", data_path="env_assets/intercode/data/python/mbpp/ic_mbpp.json", verbose=True, preprocess=None, is_agent=True)
    print(f'Num envs: {len(env.data_loader)}')
    try:
        for idx in range(24): # 24 data points in the test set
            env.reset(idx) # pass the index to prevent random data selection
            obs, done = env.observation, False # obs here is the natural language prompt
            while not done:
                action = input('> ')
                obs, reward, done, info = env.step(action)
                # After passing 'submit' to action, reward contains the score for that iteration
                # Note: Success Rate = (number of scores == 1.0 / total number of scores)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected")
    finally:
        env.close()