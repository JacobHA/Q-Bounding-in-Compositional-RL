import sys
sys.setrecursionlimit(10_000)

import matplotlib.pyplot as plt
import numpy as np
from frozen_lake_env import ModifiedFrozenLake, MAPS
from gym.wrappers import TimeLimit
from utils import gen_q_solver, get_dynamics_and_rewards, get_policy
import json
import argparse


def sparsity(beta, gamma, N, size, num_rewards, tolerance, num_runs, output_file, func, mode):

    # Define the compositionality
    def f(x,y):
        if func == 'avg':
            return 0.5*x + 0.5*y
        if func == 'add':
            return x + y
        if func == 'max':
            return np.maximum(x,y)
        if func == 'min':
            return np.minimum(x,y)

    # Set up the environment
    nA=4
    if beta == -1:
        beta = np.inf
    # Empty map
    desc = np.array(['F' * size] * size, dtype='c')
    empty_env = ModifiedFrozenLake(desc=desc, n_action=nA, step_penalization=1) # deterministic dynamics
    empty_env = TimeLimit(empty_env, N)
    nS = empty_env.nS
    _, empty_rewards = get_dynamics_and_rewards(empty_env)
    empty_rewards = empty_rewards.reshape((nS, nA))
    task_reward_dict = {0: None, 1: None, 2: None}

    kld_run = []

    for _ in range(num_runs):
        for task_num in [0,1,2]:
            remaining_states = np.arange(nS)
            rewards = empty_rewards.copy()
            for _ in range(num_rewards):
                # Choose a random (remaining) state to grant a reward
                state_index = np.random.choice(remaining_states)
                remaining_states = np.delete(remaining_states, np.where(remaining_states == state_index))
                # Provide a random reward, sampled uniformly from [0, 1] (note: output rwd is therefore [-1, 0])
                rewards[state_index, :] += np.random.uniform(0, 1)

            task_reward_dict[task_num] = rewards.flatten()

        task_reward_dict[2] = f(task_reward_dict[0], task_reward_dict[1])

        policies = {0: None, 1: None, 2: None}
        qs = {0: None, 1: None, 2: None}

        for task_num in task_reward_dict:
            reward = task_reward_dict[task_num]
            q0 = None
            if task_num == 2:
                qf = f(qs[0], qs[1])
                q0 = qf
            q, v, pi = gen_q_solver(empty_env, rewards = reward, beta=beta, gamma=gamma, tolerance=tolerance, Q0=q0)
            policies[task_num] = pi
            qs[task_num] = q
        

        pi_f = get_policy(qf, beta)

        if beta == np.inf:
            kld = (np.abs(qf-qs[2]) ).mean()
            kld_run.append(kld.tolist())
            # print(kld)
        
        else:
            if mode == 'kld':
                kld = (pi_f * np.log(pi_f / policies[2])).sum(axis=1)
            
            else:
                kld = (np.abs(qf-qs[2]) ).mean()

            kld_run.append(kld.tolist())


    my_data_dict = {
        'META': {'beta': beta, 'gamma': gamma, 'N': N, 'size': size, 'num_rewards': num_rewards, 'tolerance': tolerance, 'num_runs': num_runs, 'func': func, 'mode': mode},
        'kld_run': kld_run,
    }

    with open(output_file.replace('.json', '') + f"_n{num_rewards:03d}.json", 'w') as file:
        json.dump(my_data_dict, file, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beta',  type=float, default=1)
    parser.add_argument('-g', '--gamma',  type=float, default=0.99)
    parser.add_argument('-N', '--num_steps',  type=int, default=200)
    parser.add_argument('-s', '--size',  type=int, default=7)
    parser.add_argument('-n', '--num_rewards',  type=int, default=1)
    parser.add_argument('-f', '--func',  type=str, default='avg')
    parser.add_argument('-t', '--tolerance',  type=float, default=1e-4)
    parser.add_argument('-r', '--num_runs',  type=int, default=10)
    parser.add_argument('-o', '--output_file', type=str, default='output_data.json')
    parser.add_argument('-m', '--mode', type=str, default='kld')
    args = parser.parse_args()

    sparsity(args.beta, args.gamma, args.num_steps, args.size, args.num_rewards, args.tolerance, args.num_runs, args.output_file, args.func, args.mode)
