import csv

from gym.wrappers import TimeLimit
from stable_baselines3 import DQN
import argparse

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy

from CompBoundedDQN import CompBoundedDQN, WarmCompBoundedDQN
from dqn_baseline import custom_env
from envs import ModifiedFrozenLake

env_names = {
    "6x6D", "6x6L", "6x6L_AND_D", "6x6L_OR_D"
}
comp_env_names = {
    "6x6L_AND_D", "6x6L_OR_D"
}
env_to_comp_type = {
    "6x6L_AND_D": "and",
    "6x6L_OR_D": "or",
}


def target_from_baseline(
        env_name: str, clip_method:str='soft', total_timesteps=200_000,
        tensorboard_log:str=None, comp_type:str=None,
        learning_starts=5_000,
        buffer_size=1_000_000,
        tau=1.0,
        batch_size=100,
        learning_rate=1e-4,
        gamma=0.99,
        exploration_fraction=0.1,
        version="v2"
):
    """use pre-trained SAC on a sub-environment to warmstart in a composed environment"""
    assert env_name in comp_env_names, f"Unknown environment {env_name}; must be one of {comp_env_names}"
    env = ModifiedFrozenLake(map_name=env_name)
    env = TimeLimit(env, max_episode_steps=200)
    model1 = DQN("MlpPolicy", env).load("models/DQN_6x6D.pkl")
    model2 = DQN("MlpPolicy", env).load("models/DQN_6x6L.pkl")
    if version == "v1":
        model = CompBoundedDQN(
            [model1, model2], comp_type, clip_method, "MlpPolicy", env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            tau=tau,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
        )
    elif version == "v2":
        model = WarmCompBoundedDQN(
            env, [model1, model2], comp_type, clip_method,
            verbose=1,
            tensorboard_log=tensorboard_log,
            learning_starts=learning_starts,
            buffer_size=buffer_size,
            tau=tau,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
        )
    else:
        raise ValueError(f"Unknown version {version}")
    if tensorboard_log is None:
        model.tensorboard_log = f"./tmp/bcdqn_{comp_type}_{clip_method}_{env_name}/"
    else:
        model.tensorboard_log = tensorboard_log
    model.set_env(env)
    eval_callback = EvalCallback(
        env, best_model_save_path='./tmp/dqn_standard/',
        log_path='./tmp/dqn_standard/', eval_freq=1000,
        deterministic=True, render=False, n_eval_episodes=1
    )
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)


def target_primitive_zeroshot(target_env_name: str, primitive_env_name: str, tensorboard_log:str=None):
    """use pre-trained SAC on a sub-environment to warmstart in a composed environment"""
    model = custom_env(primitive_env_name, tensorboard_log=tensorboard_log, total_timesteps=100_000, do_eval=False)
    target_env = ModifiedFrozenLake(map_name=target_env_name)
    target_env = TimeLimit(target_env, max_episode_steps=200)
    mean_reward, mean_length = evaluate_policy(model, target_env, n_eval_episodes=1, deterministic=True)
    print(f"target {target_env_name} primitive {primitive_env_name} mean_reward={mean_reward:.2f}")
    return model, mean_reward


def target_zeroshot(target_env_name: str, primitive_env_names:[str], tensorboard_log:str=None, n=1):
    """train primitive tasks, then compare their individual and composed zero-shot performance"""
    assert target_env_name in comp_env_names, f"Unknown environment {target_env_name}; must be one of {comp_env_names}"
    csv_list = [['6x6D', '6x6L', args.env]]
    for _ in range(n):
        env = ModifiedFrozenLake(map_name=target_env_name)
        env = TimeLimit(env, max_episode_steps=200)
        models = []
        scores = []
        for primitive_env_name in primitive_env_names:
            # model, score = target_primitive_zeroshot(target_env_name, primitive_env_name)
            model = DQN("MlpPolicy", env).load(f"models/DQN_{primitive_env_name}.pkl")
            score, _ = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True)
            models.append(model)
            scores.append(score)
        comp_type = env_to_comp_type[target_env_name]
        clip_method = "infer"
        model = CompBoundedDQN(
            models, comp_type, clip_method, "MlpPolicy", env, verbose=1,
            learning_starts=1_000,
            buffer_size=1_000_000,
            tensorboard_log=tensorboard_log,
            tau=1.0,
            batch_size=100,
            learning_rate=1e-4,
            gamma=0.99,
        )
        model.to('cpu')
        model.set_env(env)
        reward, _ = evaluate_policy(model, env, n_eval_episodes=1, deterministic=True, render=True)
        print(f"Composed Reward: {reward}")
        csv_list.append([*scores, reward])
    with open(f"{args.env}.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(csv_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="6x6L_AND_D")
    parser.add_argument("--clipmethod", type=str, default="none")
    parser.add_argument("--comptype", type=str, default="and")
    args = parser.parse_args()
    target_from_baseline(args.env, clip_method=args.clipmethod, comp_type=args.comptype)