from gym.wrappers import TimeLimit
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import gym

from envs import ModifiedFrozenLake, MAPS


def custom_env(
        map_name="6x6L_AND_D",
        learning_rate=1e-4,
        learning_starts=5_000,
        buffer_size=1_000_000,
        exploration_fraction=0.1,
        gamma=0.99,
        tau=1.0,
        batch_size=32,
        tensorboard_log="./tmp/dqn_standard/",
        total_timesteps=200_000,
        do_eval=True
):
    env = ModifiedFrozenLake(map_name=map_name)
    env = TimeLimit(env, max_episode_steps=200)

    model = DQN(
        "MlpPolicy", env,
        learning_rate=learning_rate,
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        tensorboard_log=tensorboard_log,
        exploration_fraction=exploration_fraction,
    )
    eval_callback = EvalCallback(env, best_model_save_path=f'./tmp/dqn_mod_frzlk_{map_name}/',
                                 log_path=f'./tmp/dqn_mod_frzlk_{map_name}/', eval_freq=5000,
                                 deterministic=True, render=False, n_eval_episodes=30)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback if do_eval else None)
    return model


def standard_env(
        map_name="6x6L_AND_D_G",
        lr=1e-3,
        learning_starts=1_000,
        buffer_size=1_000_000,
        exploration_fraction=0.1,
        gamma=0.99,
        tau=1.0,
        batch_size=32,
        tensorboard_log="./tmp/dqn_standard/",
        total_timesteps=200_000):
    desc = MAPS[map_name]
    env = gym.make("FrozenLake-v1", desc=desc)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=lr,
        learning_starts=learning_starts,
        buffer_size=buffer_size,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        tensorboard_log=tensorboard_log,
        exploration_fraction=exploration_fraction,
        verbose=1
    )
    eval_callback = EvalCallback(env, best_model_save_path='./tmp/dqn_standard/',
                                 log_path='./tmp/dqn_standard/', eval_freq=5000,
                                 deterministic=True, render=False, n_eval_episodes=30)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)


if __name__ == "__main__":
    custom_env(map_name="6x6L_AND_D")