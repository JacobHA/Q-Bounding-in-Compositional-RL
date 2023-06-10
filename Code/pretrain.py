import os
from gym.wrappers import TimeLimit
from stable_baselines3.dqn.dqn import DQN

from envs import ModifiedFrozenLake

frozen_lake_maps = ["6x6D","6x6L","6x6L_AND_D"]


def train_frozen_lake():
    for map_id in frozen_lake_maps:
        print("Training DQN on", map_id, "environment")
        env = ModifiedFrozenLake(map_name=map_id)
        env = TimeLimit(env, max_episode_steps=200)
        model = DQN(
            "MlpPolicy", env,
            learning_starts=5_000,
            buffer_size=1_000_000,
            tensorboard_log="./tmp/dqn_lnd/",
            verbose=1,
            tau=1.0,
            batch_size=100,
            learning_rate=1e-5,
            gamma=0.99,
        )
        model.learn(total_timesteps=500_000)
        pth = os.path.join(os.path.dirname(__file__), "models", f"DQN_{map_id}.pkl")
        model.save(pth)


if __name__ == "__main__":
    train_frozen_lake()