"""Customized Frozen lake enviroment"""
import sys
from contextlib import closing
from scipy.sparse import csr_matrix
from gym.envs.toy_text import discrete
from gym import utils
import numpy as np

from six import StringIO


class ModifiedFrozenLake(discrete.DiscreteEnv):
    """Customized version of gym environment Frozen Lake"""

    def __init__(
            self, desc=None, map_name="4x4", slippery=0, n_action=4,
            cyclic_mode=True, never_done=True,
            goal_attractor=0.,
            max_reward=1., min_reward=0.,
            candy_reward_factor=0.5,
            step_penalization=0., reward_offset_field=None):

        self.step_penalization = step_penalization
        goal_attractor = float(goal_attractor)
        self.candy_reward_factor = candy_reward_factor

        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (min_reward, max_reward)
        if reward_offset_field is None:
            self.reward_offset_field = np.zeros(desc.shape)
        else:
            self.reward_offset_field = np.array(
                reward_offset_field, dtype=float)

        a_leftdown = 4
        a_downright = 5
        a_rightup = 6
        a_upleft = 7

        if n_action == 2:
            a_left = 0
            a_down = None
            a_right = 1
            a_up = None
            a_stay = None
        elif n_action == 3:
            a_left = 0
            a_down = None
            a_right = 1
            a_up = None
            a_stay = 2
        elif n_action in [4, 5]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_stay = 4
        elif n_action in [8, 9]:
            a_left = 0
            a_down = 1
            a_right = 2
            a_up = 3
            a_stay = 8
        else:
            raise NotImplementedError(f'n_action:{n_action}')

        all_actions = set(list(range(n_action)))
        self.n_state = n_state = nrow * ncol
        self.n_action = n_action

        isd = np.array(desc == b'S').astype('float64').ravel()
        if isd.sum() == 0:
            isd = np.array(desc == b'F').astype('float64').ravel()
        isd /= isd.sum()
        self.isd = isd

        transition_dynamics = {s: {a: [] for a in all_actions}
                               for s in range(n_state)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, action):
            if action == a_left:
                col = max(col - 1, 0)
            elif action == a_down:
                row = min(row + 1, nrow - 1)
            elif action == a_right:
                col = min(col + 1, ncol - 1)
            elif action == a_up:
                row = max(row - 1, 0)
            elif action == a_leftdown:
                col = max(col - 1, 0)
                row = min(row + 1, nrow - 1)
            elif action == a_downright:
                row = min(row + 1, nrow - 1)
                col = min(col + 1, ncol - 1)
            elif action == a_rightup:
                col = min(col + 1, ncol - 1)
                row = max(row - 1, 0)
            elif action == a_upleft:
                row = max(row - 1, 0)
                col = max(col - 1, 0)
            elif action == a_stay:
                pass
            else:
                raise ValueError("Invalid action provided")
            return (row, col)

        def compute_transition_dynamics(action_set, action_intended):

            restart = letter in b'H' and cyclic_mode

            for action_executed in action_set:
                prob = 1. / (len(action_set) + slippery)
                prob = (slippery + 1) * \
                    prob if action_executed == action_intended else prob

                if not restart:
                    newrow, newcol = inc(row, col, action_executed)
                    newletter = desc[newrow, newcol]
                    newstate = to_s(newrow, newcol)

                    if letter == b'G':
                        newletter = letter
                        newstate = state

                    wall_hit = newletter == b'W'
                    if wall_hit:
                        newstate = state
                    is_in_hole = letter == b'H'
                    is_in_goal = letter == b'G'
                    ate_candy = letter == b'C'
                    step_nail = letter == b'N'

                    rew = 0.

                    rew -= step_nail * max_reward
                    rew += ate_candy * max_reward * self.candy_reward_factor
                    rew += is_in_goal * max_reward
                    rew += is_in_hole * min_reward

                    rew += reward_offset  # [action_executed]

                    done = is_in_goal and not never_done
                    if is_in_goal:
                        p = prob * goal_attractor
                        if p > 0:
                            sat_li.append((p, newstate, rew, done))
                        for ini_state, start_prob in enumerate(isd):
                            p = start_prob * prob * (1 - goal_attractor)
                            if p > 0.0:
                                sat_li.append((p, ini_state, rew, done))
                    else:
                        sat_li.append((prob, newstate, rew, done))
                else:
                    done = False
                    is_in_hole = letter == b'H'
                    is_in_goal = letter == b'G'

                    rew = 0.
                    rew += is_in_goal * max_reward
                    rew += is_in_hole * min_reward
                    rew += reward_offset

                    for ini_state, start_prob in enumerate(isd):
                        if start_prob > 0.0:
                            sat_li.append(
                                (start_prob * prob, ini_state, rew, done))

        for row in range(nrow):
            for col in range(ncol):
                state = to_s(row, col)

                for action_intended in all_actions:
                    sat_li = transition_dynamics[state][action_intended]
                    letter = desc[row, col]
                    reward_offset = self.reward_offset_field[row, col]

                    if slippery != 0:
                        if action_intended == a_left:
                            action_set = set([a_left, a_down, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_down:
                            action_set = set([a_left, a_down, a_right])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_right:
                            action_set = set([a_down, a_right, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_up:
                            action_set = set([a_left, a_right, a_up])
                            action_set = action_set.intersection(all_actions)
                        elif action_intended == a_stay:
                            action_set = set([a_stay])
                        else:
                            raise ValueError(
                                f"encountered undefined action: {action_intended}")

                    else:
                        action_set = set([action_intended])

                    compute_transition_dynamics(action_set, action_intended)

        super(ModifiedFrozenLake, self).__init__(
            n_state, n_action, transition_dynamics, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)

        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                ["Left", "Down", "Right", "Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
        else:
            return None


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
    assert (colsums.round(12) == 1.).all(), \
        f"min colsum={colsums.min()}, max colsum={colsums.max()}"

    rewards = csr_matrix((rew_lst, (row_lst, col_lst)),
                         shape=shape).sum(axis=0)

    return dynamics, rewards


MAPS = {
    "2x9ridge": [
        "FFFFFFFFF",
        "FSFHHHFGF"
    ],
    "3x2uturn": [
        "FF",
        "NF",
        "CF",
    ],
    "3x3uturn": [
        "SFF",
        "HHF",
        "GFF",
    ],
    "3x9ridge": [
        "FFFFFFFFF",
        "FFFFFFFFF",
        "FSFHFHFGF"
    ],
    "5x4uturn": [
        "SFFF",
        "FFFF",
        "HHFF",
        "FFFF",
        "GFFF",
    ],
    "3x4": [
        "SFFF",
        "FFFF",
        "FFFF",
    ],
    "3x5uturn": [
        "SFFFF",
        "HHHHF",
        "GFFFF",
    ],
    "3x6uturn": [
        "SFFFFF",
        "HHHHHF",
        "GFFFFF",
    ],
    "3x7uturn": [
        "SFFFFFF",
        "HHHHHHF",
        "GFFFFFF",
    ],
    "3x12ridge": [
        "FFFHHHHHHFFF",
        "FSFFFFFFFFGF",
        "FFFHHHHHHFFF"
    ],
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "4x4empty": [
        "FFFF",
        "FSFF",
        "FFGF",
        "FFFF"
    ],
    "5x5empty": [
        "FFFFF",
        "FSFFF",
        "FFFFF",
        "FFFFF",
        "FFFFF"
    ],
    "5x12ridge": [
        "FFFHHHHHHFFF",
        "FFFFFFFFFFFF",
        "FSFFFFFFFFGF",
        "FFFFFFFFFFFF",
        "FFFHHHHHHFFF"
    ],
    "6x6empty": [
        "FFFFFF",
        "FSFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFGF",
        "FFFFFF"
    ],
    "6x7Empty": [
        "FFFFFF",
        "FFFFFF",
        "FFWWWF",
        "FFWFFF",
        "FFWWWW",
        "FFFFFF",
        "FFFFFF"
    ],

    "7x11candyTOP": [
        "FFFFFFFFFFF",
        "FFFCFFFFFFF",
        "FFFFFFFFFFF",
        "SFFFFFFFFFG",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
    ],
    "7x11candyBOT": [
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "FFFFFFFFFFF",
        "SFFFFFFFFFG",
        "FFFFFFFFFFF",
        "FFFFFFFCFFF",
        "FFFFFFFFFFF",
    ],
    "7x11candyOR": [
        "FFFFFFFFFFF",
        "FFFCFFFFFFF",
        "FFFFFFFFFFF",
        "SFFFFFFFFFG",
        "FFFFFFFFFFF",
        "FFFFFFFCFFF",
        "FFFFFFFFFFF",
    ],

    "6x6D": [
        "FFFHFS",
        "FFFFFF",
        "FFFFFH",
        "FFFFFF",
        "FWFFFF",
        "CCCCCC"
    ],
    "6x6L": [
        "CFFHFS",
        "CFFFFF",
        "CFFFFH",
        "CFFFFF",
        "CWFFFF",
        "CFFFFF",
    ],
    "6x6L_AND_D": [
        "FFFHFS",
        "FFFFFF",
        "FFFFFH",
        "FFFFFF",
        "FWFFFF",
        "CFFFFF"
    ],
    "6x6L_OR_D": [
        "CFFHFS",
        "CFFFFF",
        "CFFFFH",
        "CFFFFF",
        "CWFFFF",
        "CCCCCC",
    ],
    "6x6D_G": [
        "FFFFFS",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "GGGGGG"
    ],
    "6x6L_G": [
        "GFFFFS",
        "GFFFFF",
        "GFFFFF",
        "GFFFFF",
        "GFFFFF",
        "GFFFFF",
    ],
    "6x6L_AND_D_G": [
        "FFFFFS",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "GFFFFF"
    ],
    "6x6L_OR_D_G": [
        "GFFFFS",
        "GFFFFF",
        "GFFFFF",
        "GFFFFF",
        "GFFFFF",
        "GGGGGG",
    ],
    "13x13L": [
        "CFFFFFFFFFFFF",
        "CFFFFFFFFFFFF",
        "CFFFWWWWWFFFF",
        "CFFFWFFFFFFFF",
        "CFFFWFFFFFWFF",
        "CFFFWFFFFFWFF",
        "CFFFWFFFFFWFF",
        "CFFFWSFFFFWFF",
        "CFFFWWWWWWWFF",
        "CFFFFFFFFFFFF",
        "CFFFFFFFFFFFF",
        "CFFFFFFFFFFFF",
        "CFFFFFFFFFFFF"
    ],
    "13x13D": [
        "FFFFFFFFFFFFF",
        "FFFFFFFFFFFFF",
        "FFFFWWWWWFFFF",
        "FFFFWFFFFFFFF",
        "FFFFWFFFFFWFF",
        "FFFFWFFFFFWFF",
        "FFFFWFFFFFWFF",
        "FFFFWSFFFFWFF",
        "FFFFWWWWWWWFF",
        "FFFFFFFFFFFFF",
        "FFFFFFFFFFFFF",
        "FFFFFFFFFFFFF",
        "CCCCCCCCCCCCC"
    ],
    "13x13L_AND_D": [
        "FFFFFFFFFFFFF",
        "FFFFFFFFFFFFF",
        "FFFFWWWWWFFFF",
        "FFFFWFFFFFFFF",
        "FFFFWFFFFFWFF",
        "FFFFWFFFFFWFF",
        "FFFFWFFFFFWFF",
        "FFFFWSFFFFWFF",
        "FFFFWWWWWWWFF",
        "FFFFFFFFFFFFF",
        "FFFFFFFFFFFFF",
        "FFFFFFFFFFFFF",
        "CFFFFFFFFFFFF"
    ],

}


def generate_random_map(size=8, p=0.8):
    """Generates a random valid map (one that has a path from start to goal)
    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # DFS to check that it's a valid path.
    def is_valid(res):
        frontier, discovered = [], set()
        frontier.append((0, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                        continue
                    if res[r_new][c_new] == 'G':
                        return True
                    if res[r_new][c_new] not in '#H':
                        frontier.append((r_new, c_new))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        res[0][0] = 'S'
        res[-1][-1] = 'G'
        valid = is_valid(res)
    return ["".join(x) for x in res]
