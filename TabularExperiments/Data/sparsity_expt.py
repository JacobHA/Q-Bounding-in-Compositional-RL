import sys
sys.setrecursionlimit(10_000)

import matplotlib.pyplot as plt
import numpy as np
from frozen_lake_env import ModifiedFrozenLake, MAPS
from gym.wrappers import TimeLimit
from utils import gen_q_solver, get_dynamics_and_rewards, get_policy
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--beta',  type=float, default=1)
parser.add_argument('-g', '--gamma',  type=float, default=0.99)
parser.add_argument('-N', '--num_steps',  type=int, default=200)
parser.add_argument('-s', '--size',  type=int, default=7)
parser.add_argument('-t', '--tolerance',  type=float, default=1e-4)
parser.add_argument('-r', '--num_runs',  type=int, default=10)
parser.add_argument('-o', '--output_file', type=str, default='output_data.json')
args = parser.parse_args()


def sparsity(beta, gamma, N, size, tolerance, num_runs, output_file):

    def f(x,y):
        return 0.5*x + 0.5*y
    nA=4

    # num_rewards = np.arange(1,2)
    kl_divs = []
    kl_stds = []

    desc = np.array(['F' * size] * size, dtype='c')
    empty_env = ModifiedFrozenLake(desc=desc, n_action=nA, step_penalization=1) # deterministic dynamics
    nS = empty_env.nS

    _, empty_rewards = get_dynamics_and_rewards(empty_env)
    empty_rewards = empty_rewards.reshape((nS, nA))
    task_reward_dict = {1: None, 2: None, 3: None}

    states_with_rewards = list(range(1,size**2 // 2,1))#0)#,15) #[1,3,5,15]
    for num_rewards in states_with_rewards:
        print(f'Current number of states with rewards: {num_rewards}')
        for _ in range(num_runs):
            kld_run = []
            for task_num in [1,2,3]:
                remaining_states = np.arange(nS)
                rewards = empty_rewards.copy()
                for _ in range(num_rewards):
                    state_index = np.random.choice(remaining_states)
                    # print(f"Subtask {task_num}. Chose state {state_index}")
                    remaining_states = np.delete(remaining_states, np.where(remaining_states == state_index))

                    rewards[state_index, :] += np.random.uniform(0, 1)
                task_reward_dict[task_num] = rewards.flatten()

            task_reward_dict[3] = f(task_reward_dict[1], task_reward_dict[2])

            policies = {}
            qs = {}

            for task_num in [1,2,3]:
                empty_env = TimeLimit(empty_env, N)

                q, v, pi = gen_q_solver(empty_env, rewards = task_reward_dict[task_num], beta=beta, gamma=gamma, tolerance=tolerance)#, savename=f'toyMDP/TOP_solution.pkl', resolve=True)
                policies[task_num] = pi
                qs[task_num] = q
                
            qf = np.max(np.array([qs[1], qs[2]]), axis=0)
            
            pi_f = get_policy(qf, beta)

            kld = (pi_f * np.log(pi_f / policies[3])).sum(axis=1)
            # kld =( (np.abs(qf-qs[3]) ).mean())

            kld_run.append(kld)
        kl_divs.append(np.mean(kld_run))
        kl_stds.append(np.std(kld_run))
        # need to mask out hole state?


    my_data_dict = {
        'META': {'beta': beta, 'gamma': gamma, 'N': N, 'size': size, 'tolerance': tolerance, 'num_runs': num_runs},
        'num_states_w_rwds': states_with_rewards,
        'kl_div_averages': kl_divs,
        'kl_div_stdevs': kl_stds,
    }

    with open(output_file, 'w') as file:
        json.dump(my_data_dict, file, indent=4)



if __name__ == "__main__":
    sparsity(args.beta, args.gamma, args.num_steps, args.size, args.tolerance, args.num_runs, args.output_file)
