import os
from joblib import Parallel, delayed, cpu_count
import pickle
import time
import numpy as np
from numpy.random import SeedSequence, default_rng
import matplotlib.pyplot as plt
from gym import spaces
from gym.envs.classic_control import CartPoleEnv, MountainCarEnv, PendulumEnv, AcrobotEnv
from gym.wrappers import TimeLimit, TransformObservation, TransformReward
from gym import spaces, ActionWrapper, Env
from scipy.sparse import coo_matrix, lil_matrix

from utils import solve_biased_unconstrained


REWARD_OFFSET = {
    'CartPole': -1,
    'MountainCar': 0,
    'Pendulum': 0,
    'Pendulum3': 0,
    'Acrobot': 0,
}


class ExtendedCartPoleEnv(CartPoleEnv):
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if self.steps_beyond_done == 0:
            # need to make this an instantaneus reward drop when done
            reward = 0.

        return next_state, reward, done, info


class ExtendedMountainCarEnv(MountainCarEnv):
    def step(self, action):
        next_state, reward, done, info = super().step(action)

        if self.state[0] >= self.goal_position:
            # need to make this an instantaneus reward when done
            reward = 0.
        return next_state, reward, done, info


class ExtendedPendulum(PendulumEnv):
    def __init__(self, g=10):
        super().__init__(g)
        high = np.array([ np.pi,  8.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # have to deal with non-standard observation manipulation
    def _get_obs(self):
        th, thdot = self.state

        # make the angle periodic
        th = (th + np.pi) % (2 * np.pi) - np.pi

        return np.array([th, thdot], dtype=np.float32)


class ExtendedAcrobot(AcrobotEnv):
    def _get_ob(self):
        th1, th2, th1dot, th2dot = self.state
        th1 = (th1 + np.pi) % (2 * np.pi) - np.pi
        th2 = (th2 + np.pi) % (2 * np.pi) - np.pi

        return np.array([th1, th2, th1dot, th2dot], dtype=np.float32)


class DiscretizeObservation(TransformObservation):
    def __init__(self, env, nbins):
        assert isinstance(env.observation_space, spaces.Box), "Observation space must be a Box (continuous valued)"
        assert (env.observation_space.low > -np.inf).all(), "Observation space must have finite bounds"
        assert (env.observation_space.high < np.inf).all(), "Observation space must have finite bounds"
        lo_clip = env.observation_space.low
        hi_clip = env.observation_space.high
        nvars, = env.observation_space.shape

        assert type(nbins) in [list, int, np.ndarray]
        if type(nbins) == int:
            nbins = np.ones(env.observation_space.shape, dtype=int) * nbins
        else:
            assert all([np.issubdtype(type(b), np.integer) for b in nbins])
            nbins = np.array(nbins, dtype=int)
            assert nbins.shape == env.observation_space.shape

        self.bounds = bounds = [np.linspace(l, h, n+1).tolist() for l, h, n in zip(lo_clip, hi_clip, nbins)]
        n = [1] + [np.prod(nbins[:i]) for i in range(1, nvars)]
        bin_edges = [b[1:-1] for b in bounds]
        def f(state):
            x = [np.digitize(s, e) for s, e in zip(state, bin_edges)]
            idx = sum([n[i] * x[i] for i in range(nvars)])
            return idx

        super().__init__(env, f)


class DiscretizeAction(ActionWrapper):
    def __init__(self, env: Env, nbins: int) -> None:
        super().__init__(env)

        assert isinstance(env.action_space, spaces.Box)
        assert len(env.action_space.shape) == 1

        self.ndim_actions, = env.action_space.shape
        self.powers = [nbins ** (i-1) for i in range(self.ndim_actions, 0, -1)]

        low = env.action_space.low
        high = env.action_space.high
        self.action_mapping = np.linspace(low, high, nbins)
        self.action_space = spaces.Discrete(nbins ** self.ndim_actions)
    
    def action(self, action):
        
        a = action
        unwrapped_action = np.zeros((self.ndim_actions,), dtype=float)

        for i, p in enumerate(self.powers):
            idx, a = a // p, a % p
            unwrapped_action[i] = self.action_mapping[idx, i]

        return  unwrapped_action


def render_random_agent(env):
    s = env.reset()
    d = False
    while not d:
        a = env.action_space.sample()
        s, r, d, i = env.step(a)
        env.render()


def run_random_agent(env, N=1):
    s = env.reset()
    l = np.zeros_like(s)
    h = np.zeros_like(s)

    for i in range(N):
        print(f"Iteration {i}", end='\r', flush=True)
        s = env.reset()
        d = False
        while not d:
            a = env.action_space.sample()
            s, r, d, i = env.step(a)
            l = np.minimum(l, s)
            h = np.maximum(h, s)

    print()
    print(np.array([l, h]).T)


def get_bounds(nbins, env_name: str):

    if env_name == 'CartPole':
        lo_clip = np.array([-2.5, -3.5, -0.25 , -2.5])
        hi_clip = np.array([ 2.5,  3.5,  0.25 ,  2.5])
    elif env_name == 'MountainCar':
        lo_clip = np.array([-1.2, -0.07])
        hi_clip = np.array([ 0.6,  0.07])
    elif env_name in ['Pendulum', 'Pendulum3']:
        lo_clip = np.array([-np.pi, -8.0])
        hi_clip = np.array([ np.pi,  8.0])
    elif env_name == 'Acrobot':
        lo_clip = np.array([-np.pi, -np.pi, -12.566371, -28.274334])
        hi_clip = np.array([ np.pi,  np.pi,  12.566371,  28.274334])
    else:
        raise ValueError('Not a valid env_name')

    return np.linspace(lo_clip, hi_clip, nbins + 1)


def get_model_estimation(env_name: str, nbins, n_tries=250):
    os.makedirs(env_name, exist_ok=True)
    output_file = os.path.join(env_name, f"model_estimation_{nbins:02d}bins.pkl")
    if os.path.exists(output_file):
        print(f'loading previously computed model for {env_name} with {nbins} bins')
        with open(output_file, 'rb') as file:
            dynamics, rewards = pickle.load(file)
        return dynamics, rewards

    env = get_environment(env_name, nbins, max_episode_steps=0, reward_offset=0)
    _ = env.reset()

    bounds = get_bounds(nbins, env_name=env_name)
    _, n_vars = bounds.shape

    cols = []
    rows = []
    data = []
    observed_rewards = {}

    def sweep_and_do(depth):
        if depth == 0:
            low, high = bounds[:2]
            for a in range(env.action_space.n):
                for _ in range(n_tries):
                    env.unwrapped.state = env.np_random.uniform(low=low, high=high)
                    s = env.f(env.unwrapped.state)
                    ns, r, d, _ = env.step(a)

                    i = s * env.action_space.n + a
                    j = ns
                    observed_rewards[i] = observed_rewards.get(i, []) + [r]
                    cols.append(i)
                    rows.append(j)
                    data.append(1)

                    if d:
                        ns = env.reset()

        else:
            for _ in range(nbins):
                sweep_and_do(depth-1)
                bounds[:, depth-1] = np.roll(bounds[:, depth-1], -1)
            bounds[:, depth-1] = np.roll(bounds[:, depth-1], -1)

    sweep_and_do(n_vars)

    data = np.array(data)
    rows = np.array(rows)
    cols = np.array(cols)

    state_space_size = nbins ** n_vars
    N = state_space_size
    M = state_space_size * env.action_space.n
    dynamics = coo_matrix((data, (rows, cols)), shape=(N, M), dtype=float).tocsc()
    rewards = lil_matrix((1, M), dtype=float)

    for i, (start, end) in enumerate(zip(dynamics.indptr, dynamics.indptr[1:])):
        rewards[0, i] = np.mean(observed_rewards.get(i, [0.]))
        if len(dynamics.data[start:end]) > 0:
            dynamics.data[start:end] = dynamics.data[start:end] / dynamics.data[start:end].sum()

    rewards = rewards.todense()

    with open(output_file, 'wb') as file:
        pickle.dump((dynamics, rewards), file)
    
    return dynamics, rewards


def get_environment(env_name: str, nbins, max_episode_steps=0, reward_offset=0):

    bounds = get_bounds(nbins, env_name=env_name)
    _, n_vars = bounds.shape
    n = [nbins ** i for i in range(n_vars)]
    low = bounds.min(axis=0)
    scale = bounds.max(axis=0) - low
    bin_edges = np.linspace(0., 1, nbins + 1)[1:-1]

    def f(state):
        x = np.digitize((state - low)/scale, bin_edges)
        return sum([n[i] * x[i] for i in range(n_vars)])

    if env_name == 'CartPole':
        env = ExtendedCartPoleEnv()
    elif env_name == 'MountainCar':
        env = ExtendedMountainCarEnv()
    elif env_name == 'Pendulum':
        env = ExtendedPendulum()
        env = DiscretizeAction(env, nbins=7)
    elif env_name == 'Pendulum3':
        env = ExtendedPendulum()
        env = DiscretizeAction(env, nbins=3)
    elif env_name == 'Acrobot':
        env = ExtendedAcrobot()
    else:
        raise ValueError(f'wrong environment name {env_name}')

    env = TransformObservation(env, f)
    if reward_offset != 0:
        env = TransformReward(env, lambda r: r + reward_offset)
    if max_episode_steps > 0:
        env = TimeLimit(env, max_episode_steps)

    return env


def test_policy(env, policy, reward_offset=0, render=True, fps=20, quiet=False, rng=None):

    if rng is not None:
        random_choice = rng.choice
    else:
        random_choice = np.random.choice

    while True:
        state = env.reset()

        done = False
        episode_reward = 0
        while not done:
            if render:
                if not quiet:
                    print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' '*10 + '\r', flush=True)
                _ = env.render()
                time.sleep(1/fps)

            # Sample action from action probability distribution
            action = random_choice(env.action_space.n, p=policy[state])

            # Apply the sampled action in our environment
            state, reward, done, _ = env.step(action)
            episode_reward += reward - reward_offset

        if not quiet:
            print(f"{state = : 6d}, {episode_reward = : 6.0f}", end=' '*10 + '\n', flush=True)

        if not render:
            return episode_reward

        user_input = input("Again? [y]/n: ")
        if user_input in ['n', 'no']:
            env.close()
            break


def compute_policies_for_plots(env_name: str, nbins_list=range(2, 12+1), beta_list=[1, 2, 5, 10], bias_max_it=150, alpha=0.9999):
    policy_filename_template = os.path.join(env_name, 'optimal_policy_beta{beta:04.1f}_{nbins:02d}bins.npy')
    info_filename_template = os.path.join(env_name, 'info_beta{beta:04.1f}_{nbins:02d}bins.pkl')

    for nbins in nbins_list:
        if any([not os.path.exists(policy_filename_template.format(beta=beta, nbins=nbins)) for beta in beta_list]):
            print(f"{nbins=: 3d}", flush=True)
            dynamics, rewards = get_model_estimation(env_name, nbins=nbins, n_tries=500)
            rewards += REWARD_OFFSET[env_name]
        else:
            continue
        for beta in beta_list:
            filename = policy_filename_template.format(beta=beta, nbins=nbins)
            if not os.path.exists(filename):
                print(f"{nbins=: 3d}, {beta=: 5.1f}", flush=True)
                u, v, optimal_policy, optimal_dynamics, estimated_distribution, info = solve_biased_unconstrained(beta, dynamics, rewards, alpha=alpha, bias_max_it=bias_max_it)
                if info['iterations_completed'] < bias_max_it:
                    np.save(filename, optimal_policy)
                    with open(info_filename_template.format(beta=beta, nbins=nbins), 'wb') as file:
                        pickle.dump(info, file)
                else:
                    print(f'W: no solution found for {env_name} with {nbins} bins and beta = {beta:.1f}')


def compute_evaluation_for_plots(env_name: str, nbins_list=range(2, 12+1), beta_list=[1, 2, 5, 10], max_episode_steps=500, n_episodes=100):
    policy_filename_template = os.path.join(env_name, 'optimal_policy_beta{beta:04.1f}_{nbins:02d}bins.npy')
    results_filename = os.path.join(env_name, 'results.pkl')

    results = {}
    for beta in beta_list:
        results[beta] = {}
        for nbins in nbins_list:
            print(f"{nbins=: 3d}, {beta=: 5.1f}", flush=True)
            env = get_environment(env_name, nbins=nbins, max_episode_steps=max_episode_steps, reward_offset=REWARD_OFFSET[env_name])
            optimal_policy = np.load(policy_filename_template.format(beta=beta, nbins=nbins))
            results[beta][nbins] = [test_policy(env, optimal_policy, REWARD_OFFSET[env_name], render=False, quiet=True) for _ in range(n_episodes)]

            with open(results_filename, 'wb') as file:
                pickle.dump(results, file)


def compute_per_iter_evaluation(env_name: str, nbins, beta, max_episode_steps=500, n_episodes=100, force=False):
    info_filename = os.path.join(env_name, f'info_beta{beta:04.1f}_{nbins:02d}bins.pkl')
    results_filename = os.path.join(env_name, f'results_beta{beta:04.1f}_{nbins:02d}bins.pkl')

    print(f"{nbins=: 3d}, {beta=: 5.1f}", flush=True)

    if os.path.exists(results_filename) and not force:
        print("This result is already available")
        return

    with open(info_filename, 'rb') as file:
        info = pickle.load(file)

    ncpu = cpu_count()

    ss = SeedSequence()
    child_seeds = ss.spawn(ncpu)
    rngs = [default_rng(s) for s in child_seeds]

    def work(policy, rng):
        env = get_environment(env_name, nbins=nbins, max_episode_steps=max_episode_steps)
        return [test_policy(env, policy, render=False, quiet=True, rng=rng) for _ in range(n_episodes//ncpu)]

    results = {}
    for i, policy in enumerate(info['policy_list']):
        print(f'evaluating iteration {i+1: 3d} / {info["iterations_completed"]}')
        results[i] = sum(Parallel(n_jobs=ncpu)(delayed(work)(policy, rng) for rng in rngs), [])

    with open(results_filename, 'wb') as file:
        pickle.dump(results, file)


def get_final_result_summary(env_name: str, nbins, beta):
    info_filename = os.path.join(env_name, f'info_beta{beta:04.1f}_{nbins:02d}bins.pkl')
    results_filename = os.path.join(env_name, f'results_beta{beta:04.1f}_{nbins:02d}bins.pkl')

    with open(info_filename, 'rb') as file:
        info = pickle.load(file)
    with open(results_filename, 'rb') as file:
        results = pickle.load(file)

    n = info['iterations_completed']
    x = np.array(results[n-1])
    m, s = x.mean(), x.std()
    l = len(x)
    print(f"{(env_name, nbins, beta)}: Finished in {n: 3d} iterations. Rewards = {m:.2f} ({s:.2f}) in {l} episodes.")
    print()


def load_and_test_policy(env_name: str, nbins, beta, iteration=-1, max_episode_steps=500):
    info_filename = os.path.join(env_name, f'info_beta{beta:04.1f}_{nbins:02d}bins.pkl')

    with open(info_filename, 'rb') as file:
        info = pickle.load(file)

    n = info['iterations_completed']
    policy = info['policy_list'][iteration]
    env = get_environment(env_name=env_name, nbins=nbins, max_episode_steps=max_episode_steps)
    test_policy(env, policy, render=True, fps=60)


def finalize_plot(env_name: str, beta_list=None, plot_type='scatter'):
    results_filename = os.path.join(env_name, 'results.pkl')
    with open(results_filename, 'rb') as file:
        results = pickle.load(file)

    beta_list = beta_list if beta_list is not None else list(results.keys())

    if plot_type == 'bar':
        n = len(beta_list)
        w = (1 / n) * 0.8
        bw = w * n
        x_shift = bw / 2
        for i, beta in enumerate(beta_list):
            res = results[beta]
            x = np.array(list(res.keys()))
            r = np.array(list(res.values()))
            _ = plt.bar(x+i*w - x_shift, r.mean(axis=1), width=w, yerr=r.std(axis=1), align='edge', log=True, label=f'{beta=:4.1f}')
        plt.legend()
        plt.show()

    elif plot_type == 'scatter':

        for i, beta in enumerate(beta_list):
            res = results[beta]
            x = np.array(list(res.keys()))
            r = np.array(list(res.values()))
            _ = plt.scatter(x, r.mean(axis=1), label=f"{beta=:3.1f}")
        plt.legend()
        plt.show()

    else:
        raise ValueError("Wrong plot type")


def plot_per_iter_evaluation(env_name: str, nbins, beta):
    results_filename = os.path.join(env_name, f'results_beta{beta:04.1f}_{nbins:02d}bins.pkl')
    with open(results_filename, 'rb') as file:
        results = pickle.load(file)

    x = np.array(list(results.keys()))
    r = np.array(list(results.values()))
    _ = plt.scatter(x, r.mean(axis=1), label=env_name)
    plt.show()

    
def main(env_name: str, nbins, beta=10, max_episode_steps=500, render=False):
    dynamics, rewards = get_model_estimation(env_name, nbins=nbins, n_tries=500)

    reward_offset = REWARD_OFFSET[env_name]
    env = get_environment(env_name=env_name, nbins=nbins, max_episode_steps=max_episode_steps, reward_offset=reward_offset)

    u, v, optimal_policy, optimal_dynamics, estimated_distribution, _ = solve_biased_unconstrained(beta, dynamics, rewards, bias_max_it=500)
    test_policy(env, optimal_policy, reward_offset=reward_offset, render=render, fps=60)


EXPERIMENTS = [
    ('Pendulum3', 8, 1),
    ('MountainCar', 12, 2),
    ('CartPole', 8, 10),
    ('Acrobot', 12, 25),
]


def generate_paper_data():
    for env_name, nbins, beta in EXPERIMENTS:
        compute_policies_for_plots(env_name, nbins_list=[nbins], beta_list=[beta], bias_max_it=150)
        compute_per_iter_evaluation(env_name, nbins=nbins, beta=beta, max_episode_steps=500, n_episodes=1024)
        plot_per_iter_evaluation(env_name, nbins=nbins, beta=beta)
        get_final_result_summary(env_name, nbins, beta)


def figure_1(w = 5):
    results_filename_template = os.path.join('{env_name}', 'results_beta{beta:04.1f}_{nbins:02d}bins.pkl')
    plt.figure(dpi=150)
    for env_name, nbins, beta in EXPERIMENTS:
        with open(results_filename_template.format(env_name=env_name, beta=beta, nbins=nbins), 'rb') as file:
            results = pickle.load(file)
        x = np.array(list(results.keys()))
        r = np.array(list(results.values()))
        y = r.mean(axis=1)
        offset, scale = y.min(), (y - y.min()).max()
        
        y = np.array([y[max(i - w + 1, 0) : i + 1].mean() for i in range(len(y))])
        y -= offset
        y /= scale

        _ = plt.plot(x, y, '-', label=env_name)
    plt.ylabel('Returns (normalized)')
    plt.xlabel('Biasing iteration')
    plt.title('Biasing algorithm improves solution performance')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # generate_paper_data()
    # figure_1(w=5)

    main('Pendulum3', nbins=8, beta=1., max_episode_steps=500, render=True)
    # main('CartPole', nbins=8, beta=10, max_episode_steps=1000, render=True)
    # main('MountainCar', nbins=12, beta=2, max_episode_steps=500, render=True)
    # main('Acrobot', nbins=10, beta=20, max_episode_steps=500, render=True)

