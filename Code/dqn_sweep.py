import wandb
import argparse

from dqn_baseline import custom_env
from dqn_compbound import target_from_baseline


PROJECTS = {
    "DQN_baseline",
    "DQN_compbound",
    "DQN_hypercompbound",
    "DQN_vshypercompbound",
    "DQN_optomalhypercompbound",
    "DQN_warmoptomalhypercompbound",
}


sweep_baseline = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout/ep_rew_mean'
    },
    "parameters": {
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1
        },
        'learning_starts': {
            "distribution": "int_uniform",
            "min": 1_000,
            "max": 10_000
        },
        'batch_size': {
            "distribution": "int_uniform",
            "min": 32,
            "max": 256
        },
        'tau': {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0
        },
        'gamma': {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0
        },
        'exploration_fraction': {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.5
        },
    },
}

sweep_full = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout/ep_rew_mean'
    },
    "parameters": {
        "clip_method": {
            "values": ["soft", "hard", "soft_hard", "none", "test"],
        },
        "map_name": {
            "values": ["6x6L_OR_D"]
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1
        },
        'batch_size': {
            "distribution": "int_uniform",
            "min": 32,
            "max": 256
        },
        'tau': {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0
        },
        'exploration_fraction': {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.3
        },
    },
}

sweep_full_vs = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout/ep_rew_mean'
    },
    "parameters": {
        "version": {
            "values": ["v1", "v2"]
        },
        "clip_method": {
            "values": ["soft", "hard", "soft_hard", "none", "test"],
        },
        "map_name": {
            "values": ["6x6L_OR_D"]
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1
        },
        'batch_size': {
            "distribution": "int_uniform",
            "min": 32,
            "max": 256
        },
        'tau': {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0
        },
        'exploration_fraction': {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.3
        },
    },
}

sweep_compbound = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout/ep_rew_mean'
    },
    "parameters": {
        "clip_method": {
            "values": ["soft", "hard", "test"],
        },
    },
}

sweep_baseline_mod = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout/ep_rew_mean'
    },
    "parameters": {
        "clip_method": {
            "values": ["soft", "hard", "soft_hard", "none"],
        },
        "map_name": {
            "values": ["6x6L_OR_D"]
        }
    },
}
sweep_optimal_clip = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout/ep_rew_mean'
    },
    "parameters": {
        "clip_method": {
            "values": ["soft", "hard", "soft_hard", "none", "test"],
        },
        "map_name": {
            "values": ["6x6L_OR_D"]
        },
    },
}

sweep_warm_optimal_clip = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'rollout/ep_rew_mean'
    },
    "parameters": {
        "clip_method": {
            "values": ["soft", "hard", "soft_hard", "none", "test"],
        },
        "map_name": {
            "values": ["6x6L_AND_D", "6x6L_OR_D"]
        },
    },
}

v1clip_to_optimal_params = {
    "soft": {
        "learning_rate": 0.003732,
        "batch_size": 247,
        "tau": 0.9898,
        "exploration_fraction": 0.1075,
    },
    "hard": {
        "learning_rate": 0.001457,
        "batch_size": 146,
        "tau": 0.5545,
        "exploration_fraction": 0.1243,
    },
    "soft_hard": {
        "learning_rate": 0.003184,
        "batch_size": 138,
        "tau": 0.7682,
        "exploration_fraction": 0.1207,
    },
    "none": {
        "learning_rate": 0.0007825,
        "batch_size": 245,
        "tau": 0.9107,
        "exploration_fraction": 0.137,
    },
    "test": {
        "learning_rate": 0.0001446,
        "batch_size": 111,
        "tau": 0.9517,
        "exploration_fraction": 0.1715,
    },
}

v2clip_to_optimal_params = {
    "soft": {
        "learning_rate": 0.0007487,
        "batch_size": 236,
        "tau": 0.8347,
        "exploration_fraction": 0.1871,
    },
    "hard": {
        "learning_rate": 0.004334,
        "batch_size": 47,
        "tau": 0.9616,
        "exploration_fraction": 0.2222,
    },
    "soft_hard": {
        "learning_rate": 0.0001676,
        "batch_size": 139,
        "tau": 0.5067,
        "exploration_fraction": 0.1088,
    },
    "none": {
        "learning_rate": 0.01235,
        "batch_size": 247,
        "tau": 0.6986,
        "exploration_fraction": 0.121,
    },
    "test": {
        "learning_rate": 0.002907,
        "batch_size": 32,
        "tau": 0.9257,
        "exploration_fraction": 0.2084,
    },
}

prj_to_config = {
    "DQN_baseline": sweep_baseline,
    "DQN_compbound": sweep_compbound,
    "DQN_hypercompbound": sweep_full,
    "DQN_vshypercompbound": sweep_full_vs,
    "DQN_optomalhypercompbound": sweep_optimal_clip,
    "DQN_warmoptomalhypercompbound": sweep_warm_optimal_clip,
}


def wandb_baseline():
    run = wandb.init(
        config=sweep_baseline['parameters'], sync_tensorboard=True)
    cfg = run.config.as_dict()
    map_name = "6x6L_AND_D"
    custom_env(
        map_name, tensorboard_log=f"runs/{run.id}", total_timesteps=200_000, **cfg)
    wandb.finish()


def wandb_compbound():
    run = wandb.init(
        config=sweep_compbound['parameters'], sync_tensorboard=True)
    cfg = run.config
    map_name = cfg.map_name
    target_from_baseline(
        map_name,
        tensorboard_log=f"runs/{run.id}",
        total_timesteps=200_000,
        comp_type="and",
        clip_method=cfg.clip_method,
    )
    wandb.finish()


map_to_comp = {
    "6x6L_OR_D": "or",
    "6x6L_AND_D": "and",
}


def wandb_compbound_vs_base():
    run = wandb.init(
        config=sweep_compbound['parameters'], sync_tensorboard=True)
    cfg = run.config
    map_name = "6x6L_OR_D"
    if cfg.clip_method == "no":
        custom_env(
            map_name, tensorboard_log=f"runs/{run.id}", total_timesteps=200_000)
    else:
        target_from_baseline(
            map_name,
            tensorboard_log=f"runs/{run.id}",
            total_timesteps=200_000,
            comp_type=map_to_comp[map_name],
            clip_method=cfg.clip_method,
        )
    wandb.finish()


def wandb_full():
    run = wandb.init(
        config=sweep_compbound['parameters'], sync_tensorboard=True)
    cfg = run.config
    cfg_dict = cfg.as_dict()
    cfg_dict.pop("clip_method")
    cfg_dict.pop("map_name")
    map_name = "6x6L_OR_D"
    target_from_baseline(
        map_name,
        tensorboard_log=f"runs/{run.id}",
        total_timesteps=200_000,
        comp_type=map_to_comp[map_name],
        clip_method=cfg.clip_method,
        **cfg_dict
    )
    wandb.finish()


def wandb_full_and():
    run = wandb.init(
        config=sweep_compbound['parameters'], sync_tensorboard=True)
    cfg = run.config
    cfg_dict = cfg.as_dict()
    cfg_dict.pop("clip_method")
    cfg_dict.pop("map_name")
    map_name = "6x6L_AND_D"
    target_from_baseline(
        map_name,
        tensorboard_log=f"runs/{run.id}",
        total_timesteps=200_000,
        comp_type=map_to_comp[map_name],
        clip_method=cfg.clip_method,
        **cfg_dict
    )
    wandb.finish()


def wandb_optimal():
    run = wandb.init(
        config=sweep_compbound['parameters'], sync_tensorboard=True)
    cfg = run.config
    map_name = "6x6L_OR_D"
    target_from_baseline(
        map_name,
        tensorboard_log=f"runs/{run.id}",
        total_timesteps=200_000,
        comp_type=map_to_comp[map_name],
        clip_method=cfg.clip_method,
        **v1clip_to_optimal_params[cfg.clip_method]
    )
    wandb.finish()


def wandb_warmstart_optimal():
    run = wandb.init(config=sweep_compbound['parameters'], sync_tensorboard=True)
    cfg = run.config
    map_name = "6x6L_AND_D"
    target_from_baseline(
        map_name,
        tensorboard_log=f"runs/{run.id}",
        total_timesteps=200_000,
        comp_type=map_to_comp[map_name],
        clip_method=cfg.clip_method,
        **v2clip_to_optimal_params[cfg.clip_method]
    )
    wandb.finish()


project_to_wandb_func = {
    "DQN_baseline": wandb_baseline,
    "DQN_compbound": wandb_compbound,
    "DQN_hypercompbound": wandb_full,
    "DQN_vshypercompbound": wandb_full_and,
    "DQN_optomalhypercompbound": wandb_optimal,
    "DQN_warmoptomalhypercompbound": wandb_warmstart_optimal,
}


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="DQN_optomalhypercompbound")
    parser.add_argument("--clip_method", type=str, default=None)  # "soft"
    args = parser.parse_args()
    assert args.project in project_to_wandb_func, f"project {args.project} not found, must be one of {project_to_wandb_func.keys()}"
    config = prj_to_config[args.project]
    # only run specific clip method
    if args.clip_method is not None:
        config["parameters"]["clip_method"]["values"] = [args.clip_method]
    sweep_id = wandb.sweep(config, project=args.project)
    wandb.agent(sweep_id, function=wandb_warmstart_optimal, count=40, project=args.project)