import json
import matplotlib.pyplot as plt
import numpy as np
from frozen_lake_env import ModifiedFrozenLake, MAPS
from gym.wrappers import TimeLimit
from utils import extract_rwd_from_desc, sa_goal_cost, sa_gravity_cost, test_policy, gen_q_solver, get_dynamics_and_rewards, get_policy, plottable_rwds
from visualization import plot_dist, plot_errors, plot_kld

def empty_room_composition():

    size = 10
    dfe = 3
    N = 1_000
    desc = np.array(['F' * size] * size, dtype='c')
    # These 'C' must be candies not goals ('G'), otherwise subtasks would not share the same dynamics
    
    desc1 = desc.copy()
    desc1[dfe -1, dfe -1] = b'C'
    desc2 = desc.copy()
    desc2[-dfe, -dfe] = b'C'
    desc3 = desc.copy()
    desc3[dfe -1,  dfe -1] = b'C'
    desc3[-dfe, -dfe] = b'C'
    beta = 3
    gamma=0.9

    # we will collect the solutions in a list
    policies = []
    qs = []
    for name, desc in zip(['top corner', 'bottom corner', 'both'], [desc1, desc2, desc3]):

        # Set up the problem:
        env = ModifiedFrozenLake(desc=desc)
        nS, nA = (env.nS, env.nA)
        env = TimeLimit(env, N)

        # Solve the problem:
        Q, V, pi = gen_q_solver(env, beta=beta, gamma=gamma, tolerance=0.0001, savename=f'empty_room_solutions/{name}_solution_gamma_{gamma}_beta_{beta}_N_{N}.pkl')

        # Save the policy and value function for each task:
        policies.append( pi )
        qs.append( Q ) 

    # Perform the OR composition between the two subtasks
    Q_comp = np.array([qs[0], qs[1]]).max(axis=0) # take the max at each (s,a) pair
    or_composition_policy = get_policy(Q_comp, beta)

    policy_dict = {'1': policies[0], '2': policies[1], 'optimal': policies[2], 'or': or_composition_policy}

    n_replicas = 0
    if n_replicas > 1:
        policy_evals = []
        for name in policy_dict.keys():
            # Run the policies in the composite environment
            policy = policy_dict[name]
            # plot_dist(desc, [policy])

            policy_evals.append([test_policy(env, policy, beta=beta) for _ in range(n_replicas)])
            # TODO: bug alert. this works^ bc correct env is last one created
            # create a histogram
            
        # Now we can calculate bins properly:
        m = min([min(policy_eval) for policy_eval in policy_evals])
        M = max([max(policy_eval) for policy_eval in policy_evals])
        chunksize = 0.5
        bins = np.arange(m, M + 1, chunksize)
        plt.figure()
        for policy_eval, name in zip(policy_evals, policy_dict.keys()):
            _ = plt.hist(policy_eval, bins=bins, label=name, alpha=0.6)
        plt.legend()
        plt.show()

    plot_kld(desc3, policy_dict['or'], policy_dict['optimal'])

    # Check the bound on Q functions is true:
    plt.figure()
    plt.plot(qs[2].flatten() - Q_comp.flatten(), label='optimal - max comp')
    # plt.plot(Q_comp, label='max comp')
    # plt.plot(qs[0], label='1')
    # plt.plot(qs[1], label='2')
    # plt.plot(qs[2], label='optimal')
    plt.legend()
    plt.show()


def fourrooms_sweep():
    N = 500
    beta = np.inf
    gamma=0.99
    nA=4
    # TODO: add         state_value_funcs[corner_name] = V

    value_funcs = {}
    opt_policies = {}
    # initial training of subtasks
    for corner_name in ['TL', 'TR', 'BL', 'BR', 'BRandBL', 'notBL', 'BRandTR', 'TLandTR']:
        # Check if solution already exists in pickled file
                
        desc = MAPS[f'4rooms{corner_name}']

        env = ModifiedFrozenLake(desc=desc, n_action=nA)
        env = TimeLimit(env, N)

        # Solve the problem:
        Q, V, pi = gen_q_solver(env, beta=beta, gamma=gamma, tolerance=0.0001, resolve=False,\
                                savename=f'four_room_solutions/{corner_name}_solution_gamma_{gamma}_beta_{beta}_N_{N}.pkl')

        opt_policies[corner_name] = pi 
        value_funcs[corner_name] = Q 
     
    # Now we compose the two tasks BL, BR with different convex weights
    for eta in np.linspace(0.0,1.0,50):
        eta = round(eta,2)#eta=0.2
        composite_value = np.array(eta*value_funcs['BL'] + (1-eta)*value_funcs['BR'])
        # Now get optimal policy by solving the problem:
        # First must calculate the corresponding reward function
        composite_reward = np.zeros((1, env.nS * env.nA))#, dtype=np.float64)

        for corner_name, weight in zip(['BL', 'BR'], [eta, 1-eta]):

            desc = MAPS[f'4rooms{corner_name}']

            env = ModifiedFrozenLake(desc=desc, n_action=nA)
            env = TimeLimit(env, N)
            _, r = get_dynamics_and_rewards(env)

            composite_reward = weight * r + composite_reward

        # Now solve the composite task directly (for comparison):
        optimal_q, optimal_v, optimal_pi = gen_q_solver(env, beta=beta, gamma=gamma, rewards=composite_reward, tolerance=0.0001, savename=f'four_room_solutions/composite_solution_gamma_{gamma}_beta_{beta}_eta_{eta}.pkl', resolve=True)

      

        name = f'{round(eta,2)} BL, {round(1-eta,2)} BR'
        composite_policy = get_policy(composite_value, beta)
        plot_dist(MAPS['4rooms'], plottable_rwds(composite_reward, env.nS, env.nA), [composite_value.sum(axis=1), composite_policy], [optimal_v.flatten(), optimal_pi],\
            titles=['Rewards','Zero-Shot Approximation', 'Optimal Solution'], main_title=name, filename=f'BLBR_std_sweep/eta_{eta}.png', show_plot=False, dpi=200)
        
        # plt.close()
        
        
def fourrooms_slippery(name='BLandTR'):
    N = 500
    beta = np.inf
    if beta == np.inf:
        rl_type = 'Standard'
    if beta != np.inf:
        rl_type = 'MaxEnt'
    gamma = 0.9#0.7
    nA = 4
    slippery = 17
    foldername = 'fourroom_slippery17_solutions'
    action_value_funcs = {}
    state_value_funcs = {}
    opt_policies = {}
    # initial training of subtasks
    for corner_name in ['TL', 'TR', 'BL', 'BR', 'BRandBL', 'BRandTR']:
        desc = MAPS[f'4rooms{corner_name}']
        nS = len(desc) * len(desc[0])

        env = ModifiedFrozenLake(desc=desc, n_action=nA, slippery=slippery)
        env = TimeLimit(env, N)

        # Solve the problem:
        Q, V, pi = gen_q_solver(env, beta=beta, gamma=gamma, tolerance=0.00001, savename=f'{foldername}/{corner_name}_solution_gamma_{gamma}_beta_{beta}_N_{N}.pkl', verbose=False, resolve=True)

        opt_policies[corner_name] = pi 
        action_value_funcs[corner_name] = Q 
        state_value_funcs[corner_name] = V
     
            
    # name='notBL'

    NOT_prepend = env.reward_range[1] * (1 - gamma**N) / (1 - gamma)

    if name == 'BRandBL':
        composite_value = np.array([action_value_funcs['BR'], action_value_funcs['BL']]).max(axis=0)
      
    if name == 'notBL':
        # Calculate the transformed reward function
        r = extract_rwd_from_desc(MAPS[f'4roomsBL'])
        transf_reward = r.max() - r 
        # Solve the new problem:
        optimal_q, optimal_v, optimal_pi = gen_q_solver(env, beta=beta, gamma=gamma, rewards=transf_reward, tolerance=1e-6, savename=f'{foldername}/{name}_solution_gamma_{gamma}_beta_{beta}.pkl', resolve=True)
        composite_value = (NOT_prepend - action_value_funcs['BL']) 
    if name == 'BRandTR':
        composite_value = np.array([action_value_funcs['BR'], action_value_funcs['TR']]).max(axis=0)
    if name == 'TLandBR':
        composite_value = np.array([action_value_funcs['TL'], action_value_funcs['TR']]).max(axis=0)
    if name == 'TLnotTR':
        composite_value = NOT_prepend - np.array([NOT_prepend - action_value_funcs['TL'], action_value_funcs['TR']]).min(axis=0)
    if name == 'BLandTR':
        r1 = extract_rwd_from_desc(MAPS[f'4roomsBL'])
        r2 = extract_rwd_from_desc(MAPS[f'4roomsTR'])
        transf_reward = np.maximum(r1, r2)
        optimal_q, optimal_v, optimal_pi, optimal_errs = gen_q_solver(env, beta=beta, gamma=gamma, rewards=transf_reward, tolerance=1e-10, savename=f'{foldername}/{name}_solution_gamma_{gamma}_beta_{beta}.pkl', resolve=True, verbose=True)
        composite_value = np.array([action_value_funcs['BL'], action_value_funcs['TR']]).max(axis=0)
        warmstart_q, warmstart_v, warmstart_pi, warmstart_errs = gen_q_solver(env, beta=beta, gamma=gamma, rewards=transf_reward, tolerance=1e-10, savename=f'{foldername}/{name}_solution_gamma_{gamma}_beta_{beta}.pkl', resolve=True, verbose=True, Q0=composite_value)
       

    try:
        optimal_q = action_value_funcs[name]
        optimal_v = state_value_funcs[name]
        optimal_pi = opt_policies[name]
    except Exception as e:
        print(e)

    plot_qs = False
    if plot_qs:
        plt.figure()
        plt.title('Difference between optimal and OR-Composition Q functions')
        # plt.plot(action_value_funcs[name].flatten(), label='true')
        plt.plot(optimal_q.flatten(), label='optimal')
        plt.plot(composite_value.flatten(), label='composite') 
        # plt.yscale('log')
        plt.legend()
        plt.show()

    composite_policy = get_policy(composite_value, beta)
    # plot_dist(MAPS['4rooms'], composite_value.sum(axis=1), state_value_funcs[name].sum(axis=1), main_title=f'State value functions V(s) for {name}', titles=['Composition Approx.', 'Optimal'])
    # plot_dist(MAPS['4rooms'+name], composite_policy, opt_policies[name], main_title=name, titles=['Composition Approx.', 'Optimal'])

    
    plot_dist(MAPS[f'4rooms{name}'], [plottable_rwds(transf_reward, nS, nA)], [composite_value.sum(axis=1), composite_policy], [optimal_v.sum(axis=1), optimal_pi], \
        # main_title=f'State-value functions V(s) for convex combinations of Bottom Left and Bottom Right Tasks', \
            titles=['New Reward Function', 'Zero-Shot Approximation', 'Optimal Solution'], main_title=f'OR Composition: {rl_type} RL', \
                 filename=f'fourrooms_OR_composition_{rl_type}.pdf', dpi = 600)


    plot_errors(warmstart_errs, optimal_errs, fr'Images/fourrooms_OR_composition_{rl_type}_errors.pdf')

    if beta != np.inf:
        plot_kld(MAPS[f'4rooms{name}'], composite_policy, optimal_pi) 

def absorbing_experiment():
    nA = 4
    slippery = 5
    N = 150
    candy_strength = 0.99
    beta=5#np.inf
    if beta == np.inf:
        rl_type = 'Standard'
    if beta != np.inf:
        rl_type = 'MaxEnt'
    gamma=1

    full_desc = MAPS['7x11candies']
    full_desc = [row[:3] + 'F' + row[3:] for row in full_desc]
    nS = len(full_desc) * len(full_desc[0])

    full_env = ModifiedFrozenLake(desc=full_desc, slippery=slippery, n_action=nA, cyclic_mode=False, goal_attractor=1, candy_strength = candy_strength)
    full_env = TimeLimit(full_env, N)
    
    # Solve the full task:
    full_q, full_v, full_pi, full_errs = gen_q_solver(full_env, beta=beta, gamma=gamma, tolerance=1e-10, savename=f'absorbing/full_solution.pkl', resolve=True, verbose=True)

    # Set up the subtasks:
    descTOP = MAPS['7x11candyTOP']
    # Add in a column of F at the 4th position by popping into the list of strings:
    descTOP = [row[:3] + 'F' + row[3:] for row in descTOP]
    envTOP = ModifiedFrozenLake(desc=descTOP, slippery=slippery, n_action=nA, cyclic_mode=False, goal_attractor=1, candy_strength = candy_strength)
    envTOP = TimeLimit(envTOP, N)

    descBOT = MAPS['7x11candyBOT']
    descBOT = [row[:3] + 'F' + row[3:] for row in descBOT]
    envBOT = ModifiedFrozenLake(desc=descBOT, slippery=slippery, n_action=nA, cyclic_mode=False, goal_attractor=1, candy_strength=candy_strength)
    envBOT = TimeLimit(envBOT, N)

    # Solve the subtasks:
    TOP_q, TOP_v, TOP_pi = gen_q_solver(envTOP, beta=beta, gamma=gamma, tolerance=1e-10, savename=f'absorbing/TOP_solution.pkl', resolve=True)
    BOT_q, BOT_v, BOT_pi = gen_q_solver(envBOT, beta=beta, gamma=gamma, tolerance=1e-10, savename=f'absorbing/BOT_solution.pkl', resolve=True)
    
    # Combine them with "OR" operation:
    combined_q = np.maximum(TOP_q, BOT_q)
    combined_pi = get_policy(combined_q, beta)
    # Plot the solution:
    plot_dist(full_desc, [full_q.sum(axis=1), full_pi], [combined_q.sum(axis=1), combined_pi], titles=['Zero-Shot Approximation', 'Optimal Solution'], main_title=f'Absorbing State: {rl_type} RL')#, filename=f'absorbing_{rl_type}.pdf')
    
    # Do a warmstart:
    warmstart_q, warmstart_v, warmstart_pi, warmstart_errs = gen_q_solver(full_env, beta=beta, gamma=gamma, tolerance=1e-10, savename=f'absorbing/{rl_type}_warmstart_solution.pkl', resolve=True, Q0=combined_q, verbose=True)
    plot_errors(warmstart_errs, full_errs)#, filename=f'absorbing_{rl_type}_errors.pdf')

    # Plot the true q function and the combined_q:
    plt.figure()
    # plt.plot(full_q, label='Full solution')
    # plt.plot(combined_q, label='Combined')
    plt.plot(full_q - combined_q, label='Difference')
    # print(combined_q)
    # print(full_q)
    plt.legend()
    plt.show()

    if beta != np.inf:
        # Greedy policies will have infinite divergence
        plot_kld(full_desc, combined_pi, full_pi)

def subtask_figures(domain='4roomsTL'):

    if domain == 'goal':
        nA = 4
        goal_desc = MAPS['9x10grav']
        nS = len(goal_desc) * len(goal_desc[0])

        # Remove the goal state (so dynamics are the same across all subtasks)
        empty_desc = MAPS['9x10empty']
        
        # Calculate goal-based cost as Euclidean distance from the goal state 'G':
        goal_rwds = sa_goal_cost(goal_desc, nS, nA)
        # Change costs into rewards:
        goal_rwds = goal_rwds.max() - goal_rwds 
        plot_dist(empty_desc, plottable_rwds(goal_rwds, nS, nA), main_title='Goal-Based Cost', filename='Images/goal_cost.pdf')

        # Calculate gravitational cost as height from bottom of map:
        grav_rwds = sa_gravity_cost(goal_desc, nS, nA)
        # Change costs into rewards:
        grav_rwds = grav_rwds.max() - grav_rwds
        plot_dist(empty_desc, plottable_rwds(grav_rwds, nS, nA), main_title='Gravitational Cost', filename='Images/gravity_cost.pdf')

    else:
        desc = MAPS[f'{domain}']
        plot_dist(desc, filename=f'Images/subtask_{domain}.pdf', dpi=800)

def stochasticity():
    nA=4
    slips = np.linspace(2, 50, 30)

    # Set up the plot
        
    import seaborn as sns
    sns.set_style("whitegrid") # paints a grid 


    intended_prob = 1 / (nA + slips)
    intended_prob = (slips + 1) * intended_prob

    plt.rcParams["font.size"] = 36

    for beta in [3]: #[1,5,np.inf]:
        # plt.figure(figsize=(10,8))
        plt.title(fr'Stochasticity Experiment For $\beta=${beta}')

        kl_divs = []

        gamma=0.99
        N = 200
        if beta == np.inf:
            rl_type = 'Standard'
        if beta != np.inf:
            rl_type = 'MaxEnt'
        for slippery in slips:
            policies = {}
            qs = {}

            for name in ['A', 'B', 'C']:
                desc = MAPS['5x5'+name] # or 3x2uturn

                env = ModifiedFrozenLake(desc=desc, n_action=nA, slippery=slippery)
                env = TimeLimit(env, N)

                q, v, pi = gen_q_solver(env, beta=beta, gamma=gamma, tolerance=1e-6)#, savename=f'toyMDP/TOP_solution.pkl', resolve=True)
                policies[name] = pi
                qs[name] = q
                
            qf = np.max(np.array([qs['A'], qs['B']]), axis=0)
            
            pi_f = get_policy(qf, beta)

            kld = ((pi_f * np.log(pi_f / policies['C'])).sum(axis=1)).mean()
            # kld = (np.abs(qf-qs['C']) ).mean()
            kl_divs.append( kld ) #kld.mean()) # 
            # print(kl_divs)
            # need to mask out hole state?

        plt.plot(intended_prob, kl_divs, 'bo-')#, label=fr'$\beta={beta}$')
        plt.xlabel('Probability for Intended Action to Occur')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        plt.ylabel(r'KL Divergence Between $\pi^*$ and $\pi_f$')
        # plt.ylabel(r'Unweighted average of $f(Q) - \widetilde{Q}^*$')

        # plt.legend(loc='best')
        # plt.savefig(f'toyMDP/beta{beta}_klvsstoch.pdf',bbox_inches='tight', pad_inches=0)
        plt.show()

        my_data_dict = {
        'META': {'beta': beta, 'gamma': gamma, 'N': N},
        'kl_divs': kl_divs,
        'intended_prob': intended_prob,
        }

        output_file = f'toyMDP/data/kl_divs_beta{beta}.json'
        # output_file = f'toyMDP/data/qdiffs_beta{beta}.json'

        with open(output_file, 'w') as file:
            
            json.dump(my_data_dict, file, indent=4)


        # plt.savefig(f'toyMDP/beta{beta}_qdiffs.pdf')



def sparsity(beta=3):

    def f(x,y):
        return 0.5*x + 0.5*y
    nA=4

    # num_rewards = np.arange(1,2)
    kl_divs = []
    kl_stds = []

    gamma=0.99
    N = 200

    size = 7
    
    desc = np.array(['F' * size] * size, dtype='c')
    empty_env = ModifiedFrozenLake(desc=desc, n_action=nA, step_penalization=1) # deterministic dynamics
    nS = empty_env.nS

    _, empty_rewards = get_dynamics_and_rewards(empty_env)
    empty_rewards = empty_rewards.reshape((nS, nA))
    task_reward_dict = {1: None, 2: None, 3: None}

    NUM_RUNS = 5
    states_with_rewards = range(2,25,2)#0)#,15) #[1,3,5,15]
    for num_rewards in states_with_rewards:
        for _ in range(NUM_RUNS):
            kld_run = []
            for task_num in task_reward_dict:
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

                q, v, pi = gen_q_solver(empty_env, rewards = task_reward_dict[task_num], beta=beta, gamma=gamma, tolerance=1e-4)#, savename=f'toyMDP/TOP_solution.pkl', resolve=True)
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

    print(kld_run)
    kl_divs = np.array(kl_divs)
    kl_stds = np.array(kl_stds)
    print(kl_stds)
    plt.figure()
    plt.plot(states_with_rewards, kl_divs)
    plt.fill_between(states_with_rewards, kl_divs - kl_stds, kl_divs + kl_stds, alpha=0.2)
    plt.xlabel('Number of states with a non-default reward')
    plt.ylabel(r'KL Divergence Between $\pi^*$ and $\pi_f$')
    plt.legend()
    plt.savefig(f'sparsity/beta{beta}_klvsstoch.pdf')

    plt.show()


if __name__ == "__main__":
    # empty_room_composition()
    # fourrooms_slippery()
    # dense_goal()
    # absorbing_experiment()
    stochasticity()
    # sparsity()
    # subtask_figures('5x5C')
