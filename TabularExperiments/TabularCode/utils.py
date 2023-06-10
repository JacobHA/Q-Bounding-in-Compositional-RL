import os
import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix


def get_dynamics_and_rewards(env):

    ncol = env.nS * env.nA
    nrow = env.nS

    shape = (nrow, ncol)

    row_lst, col_lst, prb_lst, rew_lst = [], [], [], []

    assert isinstance(env.P, dict)
    for s_i, s_i_dict in env.P.items():
        for a_i, outcomes in s_i_dict.items():
            for prb, s_j, r_j, _ in outcomes:
                col = s_i * env.nA + a_i

                row_lst.append(s_j)
                col_lst.append(col)
                prb_lst.append(prb)
                rew_lst.append(r_j * prb)

    dynamics = csr_matrix((prb_lst, (row_lst, col_lst)), shape=shape)
    colsums = dynamics.sum(axis=0)
    assert (colsums.round(12) == 1.).all(), f"{colsums.min()=}, {colsums.max()=}"

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)), shape=shape).sum(axis=0)

    return dynamics, rewards


def find_exploration_policy(dynamics, rewards, n_states, n_actions, beta=1, alpha=0.01, prior_policy=None, debug=False, max_it=20):

    rewards[:] = 0
    prior_policy = np.matrix(np.ones((n_states, n_actions))) / n_actions if prior_policy is None else prior_policy
    if debug:
        entropy_list = []

    for i in range(1, 1 + max_it):
        u, v, optimal_policy, _, estimated_distribution, _ = solve_biased_unconstrained(beta, dynamics, rewards, prior_policy, bias_max_it=20)
        
        sa_dist = np.multiply(u, v.T)
        mask = sa_dist > 0
        r = rewards.copy()
        r[:] = 0.
        r[mask] = - np.log(sa_dist[mask].tolist()[0]) /beta
        r = r - r.max()
        rewards = (1 - alpha) * rewards + alpha * r

        if debug:
            x = sa_dist[sa_dist > 0]
            entropy = - np.multiply(x, np.log(x)).sum()
            entropy_list.append(entropy)

            print(f"{i=}\t{alpha=:.3f}\t{entropy=: 10.4f}\t", end='')

    return optimal_policy


def get_mdp_generator(env, transition_dynamics, policy):
    td_coo = transition_dynamics.tocoo()

    rows, cols, data = [], [], []
    for s_j, col, prob in zip(td_coo.row, td_coo.col, td_coo.data):
        for a_j in range(env.nA):
            row = s_j * env.nA + a_j
            rows.append(row)
            cols.append(col)
            data.append(prob * policy[s_j, a_j])

    nrow = ncol = env.nS * env.nA
    shape = (nrow, ncol)
    mdp_generator = csr_matrix((data, (rows, cols)), shape=shape)

    return mdp_generator

# TODO: clean this up
def plottable_rwds(rewards, nS, nA):
    if isinstance(rewards, np.matrix):
        print('here')
        rwds = rewards.reshape((nS, nA)).A
    if isinstance(rewards, np.ndarray): 
        rwds = rewards.reshape((nS, nA))
    else:
        raise ValueError(f"{type(rewards)} not supported.")

    rwds = np.mean(rwds, axis=1) # assuming a uniform prior
    rwds = rwds.reshape(nS,).A[0].flatten()
    return rwds

def statewise_gravity_cost(desc):
    # Calculate gravitational cost as height from bottom of map:
    nrow, ncol = len(desc), len(desc[0])

    grav_rwds = np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            grav_rwds[i, j] = nrow - i - 1 
            # Most costly to be at the top, 0 cost at bottom
    return grav_rwds
    

def statewise_goal_cost(desc):
    nrow, ncol = len(desc), len(desc[0])
    costs = np.zeros((nrow, ncol))
    goal_loc = np.where(np.array(desc, dtype='c')==b'G')
    goal_row, goal_col = goal_loc
    for row in range(nrow):
        for col in range(ncol):
            if desc[row][col] == 'G':
                costs[row, col] = -10 # TODO: make this a parameter
            costs[row, col] = np.sqrt((row - goal_row)**2 + (col - goal_col)**2)
    return costs

# TODO: combine these two functions
def sa_goal_cost(desc, nS, nA):
    # re-shape the goal cost into s,a pairs
    # with the same value for all actions
    state_goal_cost = statewise_goal_cost(desc).flatten()
    stateaction_goal_cost = np.zeros((nS , nA))
    for a in range(nA):
        stateaction_goal_cost.T[a] = state_goal_cost

    return np.matrix(stateaction_goal_cost.reshape(nS, nA)).reshape(1, nS*nA)

def sa_gravity_cost(desc, nS, nA):
    # re-shape the gravity cost into s,a pairs
    # with the same value for all actions
    state_gravity_cost = statewise_gravity_cost(desc).flatten()
    stateaction_gravity_cost = np.zeros((nS , nA))
    for a in range(nA):
        stateaction_gravity_cost.T[a] = state_gravity_cost

    return np.matrix(stateaction_gravity_cost.reshape(nS, nA)).reshape(1, nS*nA)

def softq_solver(env, prior_policy=None, steps=100_000, beta=1, gamma=1, tolerance=1e-2, savename = None, verbose=False, rewards=None, resolve=False, Q0 = None):

    # First we check if the solution is already saved somewhere, if so, we load it
    if savename is not None and not resolve:
        try:
            with open(savename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass

    if rewards is None:
        dynamics, rewards = get_dynamics_and_rewards(env)
    else:
        dynamics, _ = get_dynamics_and_rewards(env)
    
    errors_list = []

    rewards = dynamics.multiply(rewards).sum(axis=0)
    prior_policy = np.ones((env.nS, env.nA)) / env.nA if prior_policy is None else prior_policy
    mdp_generator = get_mdp_generator(env, dynamics, prior_policy)

    if Q0 is None:
        Qi = np.zeros((1, env.nS * env.nA))
        Qi = np.random.rand(1, env.nS * env.nA) * rewards.max()/(1-gamma) # the approximate scale of Q
    else:
        Qi = Q0
        Qi = Qi.reshape((1, env.nS * env.nA))

    for i in range(1, steps+1):
        Qj = np.log(mdp_generator.T.dot(np.exp(beta * Qi.T)).T) / beta
        Qi_k = rewards + gamma * Qj
        err = np.abs(Qi_k - Qi).max()
        Qi = Qi_k
  
        if verbose:
            errors_list.append(err)

        if err <= tolerance:
            if verbose:
                print(f"Converged to {tolerance=} after {i=} iterations.")
            break

    if i == steps:
        print(f'Reached max steps. Err:{err}')
    else:
        print(f"Done in {i} steps")
    
    Vi = np.log(
        np.multiply(prior_policy, np.exp(beta * Qi.reshape((env.nS, env.nA)))).sum(axis=1)
    ) / beta

    policy = np.multiply(prior_policy, np.exp(beta * (Qi.reshape((env.nS, env.nA)) - Vi)))
    pi = policy.A

    Qi = np.array(Qi).reshape((env.nS, env.nA))

    Vi = np.array(Vi).reshape((env.nS, 1))

    if savename is not None:
        # TODO: allow for arbitrary directory to be made...
        foldername, filename = savename.split('/')
        exists = os.path.exists(foldername)
        if not exists:
            # Create a new directory because it does not exist 
            os.makedirs(foldername)

        with open(savename, 'wb+') as f:
            pickle.dump((Qi, Vi, pi), f)

    if verbose:
        return Qi, Vi, pi, errors_list
    else:
        return Qi, Vi, pi
    

def gen_q_solver(env, prior_policy=None, beta=1, steps=100_000, gamma=1, tolerance=1e-2, savename = None, verbose=False, rewards=None, resolve=False, Q0 = None):
    
    if beta == np.inf:
        return q_solver(env, prior_policy=prior_policy, steps=steps, gamma=gamma, tolerance=tolerance, savename = savename, verbose=verbose, rewards=rewards, resolve=resolve, Q0 = Q0)
    
    else:
        return softq_solver(env, prior_policy=prior_policy, beta=beta,  steps=steps, gamma=gamma, tolerance=tolerance, savename = savename, verbose=verbose, rewards=rewards, resolve=resolve, Q0 = Q0)

def q_solver(env, prior_policy=None, steps=100_000, gamma=1, tolerance=1e-2, savename = None, verbose=False, rewards=None, resolve=False, Q0 = None):

    # First we check if the solution is already saved somewhere, if so, we load it
    if savename is not None and not resolve:
        try:
            with open(savename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            pass

    if rewards is None:
        dynamics, rewards = get_dynamics_and_rewards(env)
    else:
        dynamics, _ = get_dynamics_and_rewards(env)

    errors_list = []

    rewards = dynamics.multiply(rewards).sum(axis=0).reshape(env.nS, env.nA)
    dynamics = dynamics.A
    prior_policy = np.ones((env.nS, env.nA)) / env.nA if prior_policy is None else prior_policy
    
    if Q0 is None:
        Qi = np.zeros((env.nS, env.nA))
        Qi = np.random.rand(env.nS , env.nA)*rewards.max()*100

    else:
        assert isinstance(Q0, np.ndarray)
        assert Q0.shape == (env.nS, env.nA)
        Qi = Q0
    

    for i in range(1, steps+1):
        Qj = Qi.max(axis=1) # get max over actions
        # "sample" by dotting over expectation of s' ~ P(s'|s,a)
        Qj = np.dot(dynamics.T, Qj)#.reshape((env.nS, 1)))

        # reshape into S, A 
        Qj = Qj.reshape((env.nS, env.nA))
        Qi_k = rewards + gamma * Qj

        err = np.abs(Qi_k - Qi).max()
        Qi = Qi_k
        
        if verbose:
            errors_list.append(err)

        if err <= tolerance:
            if verbose:
                print(f"Converged to {tolerance=} after {i=} iterations.")
            break
        
    Vi = Qi.reshape((env.nS, env.nA)).max(axis=1)
        
    pi = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        best_action = np.argmax(Qi[s])
        # Always take the best action
        pi[s, best_action] = 1.0

    Qi = np.array(Qi).reshape((env.nS, env.nA))

    Vi = np.array(Vi).reshape((env.nS, 1))

    if savename is not None:
        # TODO: allow for arbitrary directory to be made...
        foldername, filename = savename.split('/')
        exists = os.path.exists(foldername)
        if not exists:
            # Create a new directory because it does not exist 
            os.makedirs(foldername)

        with open(savename, 'wb+') as f:
            pickle.dump((Qi, Vi, pi), f)

    if verbose:
        return Qi, Vi, pi, errors_list
    else:
        return Qi, Vi, pi
    



def test_policy(env, policy, quiet=True, rng=None, beta=np.inf, prior_policy=None):

    if rng is not None:
        random_choice = rng.choice
    else:
        random_choice = np.random.choice

    if prior_policy is None:
        prior_policy = np.ones((env.nS, env.nA)) / env.nA

    state = env.reset()

    done = False
    episode_reward = 0
    while not done:
        # Sample action from action probability distribution
        action = random_choice(env.action_space.n, p=policy[state])

        # Apply the sampled action in our environment
        state, reward, done, _ = env.step(action)
        episode_reward += reward

        if beta != np.inf:
            # Add the entropic cost!
            episode_reward -= (1/beta) * np.log(policy[state,action]/prior_policy[state, action])


    if not quiet:
        print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' '*10 + '\n', flush=True)

    return episode_reward


def solve_unconstrained(beta, dynamics, rewards, prior_policy, eig_max_it=10000, tolerance=1e-8):
    tolerance *= beta

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()
    M_scale = 1.

    # left eigenvector
    u = np.matrix(np.ones((nSnA, 1)))
    u_scale = np.sum(u)

    # right eigenvector
    v = np.matrix(np.ones((nSnA, 1))) * nSnA ** 2
    v_scale = np.sum(v)

    lol = float('inf')
    hil = 0.

    for i in range(1, eig_max_it+1):

        uk = (Mt).dot(u)
        lu = np.sum(uk) / u_scale
        mask = np.logical_and(uk > 0., uk < np.inf)
        rescale = 1. / np.sqrt(uk[mask].max()*uk[mask].min())
        uk = uk / lu * rescale
        u_scale *= rescale

        vk = M.dot(v)
        lv = np.sum(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk
        lol = min(lol, lu)
        hil = max(hil, lu)

        if i % 100 == 0:
            rescale = 1 / np.sqrt(lu)
            Mt = Mt * rescale
            M_scale *= rescale


        if u_err <= tolerance and v_err <= tolerance:
        # if u_err <= tolerance:
            l = lu / M_scale
            print(f"{i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {M_scale=:.4e}, {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
            break
    else:
        l = lu / M_scale
        print(f"Did not converge: {i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {M_scale=:.4e}, {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")

    l = lu / M_scale

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    X = dynamics.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(u, v.T).reshape((nS, nA)).sum(axis=1)).flatten()

    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def solve_unconstrained_v1(beta, dynamics, rewards, prior_policy, eig_max_it=10000, tolerance=1e-8):
    
    scale = 1 / np.exp(beta * rewards.min())

    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    # The MDP transition matrix (biased)
    P = get_mdp_transition_matrix(dynamics, prior_policy)
    # Diagonal of exponentiated rewards
    T = lil_matrix((nSnA, nSnA))
    T.setdiag(np.exp(beta * np.array(rewards).flatten()))
    T = T.tocsc()
    # The twisted matrix (biased problem)
    M = P.dot(T).tocsr()
    Mt = M.T.tocsr()

    # left eigenvector
    u = np.matrix(np.ones((nSnA, 1))) * scale
    u_scale = np.linalg.norm(u)

    # right eigenvector
    v = np.matrix(np.ones((nSnA, 1))) * scale
    v_scale = np.linalg.norm(v)

    lol = float('inf')
    hil = 0.

    metrics_list = []

    for i in range(1, eig_max_it+1):

        uk = (Mt).dot(u)
        lu = np.linalg.norm(uk) / u_scale
        uk = uk / lu

        vk = M.dot(v)
        lv = np.linalg.norm(vk) / v_scale
        vk = vk / lv

        # computing errors for convergence estimation
        mask = np.logical_and(uk > 0, u > 0)
        u_err = np.abs((np.log(uk[mask]) - np.log(u[mask]))).max() + np.logical_xor(uk <= 0, u <= 0).sum()
        mask = np.logical_and(vk > 0, v > 0)
        v_err = np.abs((np.log(vk[mask]) - np.log(v[mask]))).max() + np.logical_xor(vk <= 0, v <= 0).sum()

        # update the eigenvectors
        u = uk
        v = vk
        lol = min(lol, lu)
        hil = max(hil, lu)

        if i % 100_000 == 0:
            metrics_list.append(dict(
                lu=lu,
                lv=lv,
                u_err=u_err,
                v_err=v_err,
            ))

        if u_err <= tolerance and v_err <= tolerance:
            l = lu
            print(f"{i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")
            break
    else:
        l = lu
        print(f"Did not converge: {i: 8d}, {u.min()=:.4e}, {u.max()=:.4e}. {lu=:.4e}, {l=:.4e}, {u_err=:.4e}, {v_err=:.4e}")

    l = lu

    # make it a row vector
    u = u.T

    optimal_policy = np.multiply(u.reshape((nS, nA)), prior_policy)
    scale = optimal_policy.sum(axis=1)
    optimal_policy[np.array(scale).flatten() == 0] = 1.
    optimal_policy = np.array(optimal_policy / optimal_policy.sum(axis=1))

    chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
    X = dynamics.multiply(chi).tocsc()
    for start, end in zip(X.indptr, X.indptr[1:]):
        if len(X.data[start:end]) > 0 and X.data[start:end].sum() > 0.:
            X.data[start:end] = X.data[start:end] / X.data[start:end].sum()
    optimal_dynamics = X

    v = v / v.sum()
    u = u / u.dot(v)

    estimated_distribution = np.array(np.multiply(u, v.T).reshape((nS, nA)).sum(axis=1)).flatten()

    return l, u, v, optimal_policy, optimal_dynamics, estimated_distribution


def solve_biased_unconstrained(beta, prior_dynamics, rewards, prior_policy=None, target_dynamics=None, eig_max_it=10000, alpha=0.9999, bias_max_it=200, ground_truth_policy=None, tolerance=1e-6):

    nS, nSnA = prior_dynamics.shape
    nA = nSnA // nS

    if prior_policy is None:
        prior_policy = np.matrix(np.ones((nS, nA))) / nA

    if target_dynamics is None:
        target_dynamics = prior_dynamics

    ### initialization ###
    td_bias = prior_dynamics.copy()
    td_bias.data[:] = 1.
    rw_bias = np.zeros_like(rewards)
    biased_dynamics = prior_dynamics.copy()
    biased_rewards = rewards

    error_policy_list = []
    error_dynamics_list = []
    policy_list = []
    for i in range(1, bias_max_it+1):

        l, u, v, optimal_policy, optimal_dynamics, estimated_distribution = solve_unconstrained(beta, biased_dynamics, biased_rewards, prior_policy, eig_max_it=eig_max_it)
        policy_list.append(optimal_policy)
        if ground_truth_policy is not None:
            error_policy = compute_max_kl_divergence(optimal_policy, ground_truth_policy, axis=1)
            error_policy_list.append(error_policy)

        x = target_dynamics.tocoo()
        optimal = np.array(optimal_dynamics[x.row, x.col]).flatten()
        mask = optimal > 0.
        # x.data[mask] = np.log(x.data[mask] / optimal[mask]) * x.data[mask]
        # x.data[~mask] = 0
        x.data[mask] = np.log(optimal[mask] / x.data[mask]) * optimal[mask]
        x.data[~mask] = 0

        kl_err = np.abs(x.sum(axis=0)).max()
        error_dynamics_list.append(kl_err)
        if kl_err < tolerance:
            print(f'Solved in {i} iterations')
            break


        ratio = prior_dynamics.tocoo()
        mask = ratio.data > 0
        ratio.data[mask] = np.array(target_dynamics[ratio.row, ratio.col]).flatten()[mask] / ratio.data[mask]
        ratio.data[~mask] = 0.

        chi = np.multiply(u.reshape((nS, nA)), prior_policy).sum(axis=1)
        chi_inv = np.array(chi).flatten()
        mask = chi_inv > 0
        chi_inv[mask] = 1 / chi_inv[mask]
        chi_inv[chi_inv == np.inf] = 0.
        chi_inv = np.matrix(chi_inv).T

        next_td_bias = ratio.multiply(chi_inv)
        scale = prior_dynamics.multiply(next_td_bias).sum(axis=0)
        scale_inv = np.array(scale).flatten()
        mask = scale_inv > 0
        scale_inv[mask] = 1 / scale_inv[mask]
        scale_inv = np.matrix(scale_inv)

        next_td_bias = next_td_bias.multiply(scale_inv).tocsr()
        td_bias = td_bias + alpha * (next_td_bias - td_bias)

        biased_dynamics = prior_dynamics.multiply(td_bias)

        elem = target_dynamics.tocoo()
        biased = np.array(biased_dynamics[elem.row, elem.col]).flatten()
        biased_inv = 1 / biased
        biased_inv[biased_inv == np.inf] = 1.
        mask = (biased > 0) & (elem.data > 0)
        elem.data[mask] = np.log(elem.data[mask] * biased_inv[mask]) * elem.data[mask]
        elem.data[~mask] = 0.
        rw_bias = elem.sum(axis=0) / beta

        biased_rewards = rewards + rw_bias
        reward_offset = - biased_rewards.max()
        biased_rewards +=  reward_offset

    if i == bias_max_it:
        print(f'Did not finish after {i} iterations')

    info = dict(
        error_dynamics_list=error_dynamics_list,
        error_policy_list=error_policy_list,
        policy_list=policy_list,
        iterations_completed=i,
    )
    return u, v, optimal_policy, optimal_dynamics, estimated_distribution, info


def compute_max_kl_divergence(dist_a, dist_b, axis=0):
    numer = csr_matrix(dist_a)
    denom = coo_matrix(dist_b)
    kldiv = denom.copy()
    numer = np.array(numer[denom.row, denom.col]).flatten()
    kldiv.data = np.log(numer / denom.data) * numer
    kldiv = kldiv.sum(axis=axis)

    return kldiv.max()


def compute_policy_induced_distribution(dynamics, policy, steps, isd=None):
    nS, nSnA = dynamics.shape
    nA = nSnA // nS

    mdp_generator = get_mdp_transition_matrix(dynamics, policy)

    if isd is not None:
        x = np.multiply(np.matrix(isd).T, policy).flatten().T
    else:
        x = np.matrix(np.ones((nS * nA, 1))) / nS / nA

    for _ in range(steps):
        x = mdp_generator.dot(x)

    return np.array(x).reshape((nS, nA)).sum(axis=1)


def get_mdp_transition_matrix(transition_dynamics, policy):

    nS, nSnA = transition_dynamics.shape
    nA = nSnA // nS

    td_coo = transition_dynamics.tocoo()

    rows = (td_coo.row.reshape((-1, 1)) * nA + np.array(list(range(nA)))).flatten()
    cols = np.broadcast_to(td_coo.col.reshape((-1, 1)), (len(td_coo.col), nA)).flatten()
    data = np.broadcast_to(td_coo.data, (nA, len(td_coo.data))).T.flatten()

    mdp_transition_matrix = csr_matrix((data, (rows ,cols)), shape=(nSnA, nSnA)).multiply(policy.reshape((-1, 1)))

    return mdp_transition_matrix


def largest_eigs_dense(A, n_eigs=1):

    if 'toarray' in dir(A):
        # need to be a dense matrix
        A = A.toarray()

    eigvals, eigvecs = np.linalg.eig(A)
    try:
        eigvals, eigvecs = process_complex_eigs(eigvals, eigvecs)
    except ValueError:
        raise

    return eigvals[:n_eigs], eigvecs[:, :n_eigs]


def calculate_greedy_policy(Q):
    nS, nA = Q.shape
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([nS, nA])
    for s in range(nS):
        # One step lookahead to find the best action for this state
        best_action = np.argmax(Q[s])
        # Always take the best action
        policy[s, best_action] = 1.0
    
    return policy

def get_policy(Q, beta):
    if beta != np.inf:
        #  "-=" below was mutating the original matrix, so I made a new array first
        Q_sub = Q - Q.min(axis=1, keepdims=1) # helps with overflow errors

        policy = np.exp(beta*Q_sub) / np.exp(beta*Q_sub).sum(axis=1, keepdims=1)
    else:
        policy = calculate_greedy_policy(Q)

    return policy

def extract_rwd_from_desc(desc, nA=4):
    from frozen_lake_env import ModifiedFrozenLake
    env = ModifiedFrozenLake(desc=desc, n_action=nA)
    _, r = get_dynamics_and_rewards(env)
    return r