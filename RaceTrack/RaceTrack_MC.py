#############################################################################################
# Start date: 05/11/2024                                                                    #
# Project name: Race track                                                                  #
# Author: Tri Duc Tran - trantriduc00@gmail.com												#
# Phone: 0794400840	                                                                        #
#############################################################################################

#IMPORTS
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import itertools
from IPython.display import display
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from typing import Callable

from IPython.display import HTML
from celluloid import Camera
from typing import Iterable



SMALL_TRACK = """
###      F
##       F
##       F
#      ###
#      ###
#      ###
#SSSSS####
"""

def ascii_to_array(ascii_track: str) -> np.ndarray:
    """Convert the ascii (string) map to a NumPy array."""
    
    lines = [line for line in ascii_track.split("\n") if len(line) > 0]
    byte_lines = [list(bytes(line, encoding="utf-8")) for line in lines]
    return np.array(byte_lines, dtype = np.uint8)

track = ascii_to_array(SMALL_TRACK)
print(track)
position = np.array([0, 0])
track[tuple(position)] == int.from_bytes(b'#', "big")

class const:
    @staticmethod
    def gamma():
        return 0.95
    
    @staticmethod
    def on_track():
        return -1
    
    @staticmethod
    def off_track():
        return -5
    
    @staticmethod
    def off_track():
        return -5
    
    @staticmethod
    def cross_line():
        return 1
    
class State(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray
    
class RacetrackEnv:
    """Racetrack environment"""

    EMPTY = int.from_bytes(b" ", "big")
    WALL = int.from_bytes(b"#", "big")
    START = int.from_bytes(b"S", "big")
    FINISH = int.from_bytes(b"F", "big")
    def __init__(
        self,
        ascii_track: str,
        noise_prob: float = 0.1,
        seed: int = 0,
    ) -> None:
        self._track = ascii_to_array(ascii_track)
        self._max_height, self._max_width = self._track.shape
        self._noise_prob = noise_prob
        self._actions = np.array(list(itertools.product([-1, 0, 1], [-1, 0, 1])))
        self._no_accel = 4
        self._random_state = np.random.RandomState(seed=seed)
        self._start_positions = np.argwhere(self._track == self.START)
        self._state_indices = None
        self._ax = None
        self._agent_fig = None
        self._arrow_fig = None
        
    def state_index(self, state: State) -> int:
        """Returns a state index"""
        (y, x), (vy, vx) = state
        return y * self._max_width * 25 + x * 25 + vy * 5 + vx
        

    def _all_passed_positions(
        self,
        start: np.ndarray,
        velocity: np.ndarray,
    ) -> Tuple[List[np.ndarray], bool]:
        """
        List all positions that the agent passes over.
        Here we assume that the y-directional velocity is already flipped by -1.
        """

        maxv = np.max(np.abs(velocity))
        if maxv == 0:
            return [start], False
        one_step_vector = velocity / maxv
        pos = start + 0.0
        traj = []
        for i in range(maxv):
            pos = pos + one_step_vector
            ceiled = np.ceil(pos).astype(int)
            if self._is_out(ceiled):
                return traj, True
            traj.append(ceiled)
        # To prevent numerical issue
        traj[-1] = start + velocity
        return traj, False

    def _is_out(self, position: np.ndarray) -> bool:
        """Returns whether the given position is out of the map."""
        y, x = position
        return y < 0 or x >= self._max_width

    def step(self, state: State, action: int) -> Tuple[State, float, bool]:
        """
        Taking the current state and an agents' action, returns the next state,
        reward and a boolean flag that indicates that the current episode terminates.
        """
        position, velocity = state
        if self._random_state.rand() < self._noise_prob:
            accel = self._actions[self._no_accel]
        else:
            accel = self._actions[action]
        # velocity is clipped so that only ↑→ directions are possible
        next_velocity = np.clip(velocity + accel, a_min=0, a_max=4)
        # If both of velocity is 0, cancel the acceleration
        if np.sum(next_velocity) == 0:
            next_velocity = velocity
        # List up trajectory. y_velocity is flipped to adjust the coordinate system.
        traj, went_out = self._all_passed_positions(
            position,
            next_velocity * np.array([-1, 1]),
        )
        passed_wall, passed_finish = False, False
        for track in map(lambda pos: self._track[tuple(pos)], traj):
            passed_wall |= track == self.WALL
            passed_finish |= track == self.FINISH
        if not passed_wall and passed_finish:  # Goal!
            return State(traj[-1], next_velocity), 0, True
        elif passed_wall or went_out:  # Crasshed to the wall or run outside
            return self.reset(), -1.0, False
        else:
            return State(traj[-1], next_velocity), -1, False

    def reset(self) -> State:
        """Randomly assigns a start position of the agent."""
        n_starts = len(self._start_positions)
        # initial_pos_idx = self._random_state.random(n_starts)
        initial_pos_idx = np.random.randint(0, n_starts)
        print(initial_pos_idx)
        initial_pos = self._start_positions[initial_pos_idx]
        initial_velocity = np.array([0, 0])
        return State(initial_pos, initial_velocity)

    def render(
        self,
        state: Optional[State] = None,
        movie: bool = False,
        ax: Optional[Axes] = None,
    ) -> Axes:
        """Render the map and (optinally) the agents' position and velocity."""
        if self._ax is None or ax is not None:
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            # Map the track to one of [0, 1, 2, 3] to that simple colormap works
            map_array = np.zeros_like(track)
            symbols = [self.EMPTY, self.WALL, self.START, self.FINISH]
            for i in range(track.shape[0]):
                for j in range(track.shape[1]):
                    map_array[i, j] = symbols.index(self._track[i, j])
            cm = ListedColormap(
                ["w", ".75", "xkcd:reddish orange", "xkcd:kermit green"]
            )
            map_img = ax.imshow(
                map_array,
                cmap=cm,
                vmin=0,
                vmax=4,
                alpha=0.8,
            )
            if ax.get_legend() is None:
                descriptions = ["Empty", "Wall", "Start", "Finish"]
                for i in range(1, 4):
                    if np.any(map_array == i):
                        ax.plot([0.0], [0.0], color=cm(i), label=descriptions[i])
                ax.legend(fontsize=12, loc="lower right")
            self._ax = ax
        if state is not None:
            if not movie and self._agent_fig is not None:
                self._agent_fig.remove()
            if not movie and self._arrow_fig is not None:
                self._arrow_fig.remove()
            pos, vel = state
            self._agent_fig = self._ax.plot(pos[1], pos[0], "k^", markersize=20)[0]
            # Show velocity
            self._arrow_fig = self._ax.annotate(
                "",
                xy=(pos[1], pos[0] + 0.2),
                xycoords="data",
                xytext=(pos[1] - vel[1], pos[0] + vel[0] + 0.2),
                textcoords="data",
                arrowprops={"color": "xkcd:blueberry", "alpha": 0.6, "width": 2},
            )
        return self._ax
    
smalltrack = RacetrackEnv(SMALL_TRACK)
state = smalltrack.reset()
print(state)
new_fig = smalltrack.render(state=state).get_figure()
plt.show()
display(smalltrack.render(state=state).get_figure())
next_state, reward, termination = smalltrack.step(state, 7)
print(next_state)
smalltrack.render(state=next_state)

def get_p_and_r(env: RacetrackEnv) -> Tuple[np.ndarray, np.ndarray]:
    """Taking RacetrackEnv, returns the transition probability p and reward fucntion r of the env."""
    n_states = env._max_height * env._max_width * 25
    n_actions = len(env._actions)
    p = np.zeros((n_states, n_actions, n_states))
    r = np.ones((n_states, n_actions, n_states)) * -1
    noise = env._noise_prob

    def state_prob(*indices):
        """Returns a |S| length zero-initialized array where specified elements are filled"""
        prob = 1.0 / len(indices)
        res = np.zeros(n_states)
        for idx in indices:
            res[idx] = prob
        return res

    # List up all states and memonize starting states
    states = []
    starting_states = []
    for y in range(env._max_height):
        for x in range(env._max_width):
            track = env._track[y][x]
            for y_velocity in range(5):
                for x_velocity in range(5):
                    state = State(np.array([y, x]), np.array([y_velocity, x_velocity]))
                    states.append(state)
                    if track == env.START:
                        starting_states.append(env.state_index(state))

    for state in states:
        position, velocity = state
        i = env.state_index(state)
        track = env._track[tuple(position)]
        # At a terminating state or unreachable, the agent cannot move
        if (
            track == env.FINISH
            or track == env.WALL
            or (np.sum(velocity) == 0 and track != env.START)
        ):
            r[i] = 0
            p[i, :] = state_prob(i)
        # Start or empty states
        else:
            # First, compute next state probs without noise
            next_state_prob = []
            for j, action in enumerate(env._actions):
                next_velocity = np.clip(velocity + action, a_min=0, a_max=4)
                if np.sum(next_velocity) == 0:
                    next_velocity = velocity
                traj, went_out = env._all_passed_positions(
                    position,
                    next_velocity * np.array([-1, 1]),
                )
                passed_wall, passed_finish = False, False
                for track in map(lambda pos: env._track[tuple(pos)], traj):
                    passed_wall |= track == env.WALL
                    passed_finish |= track == env.FINISH
                if passed_wall or (went_out and not passed_finish):
                    #  Run outside or crasheed to the wall
                    next_state_prob.append(state_prob(*starting_states))
                else:
                    next_state_idx = env.state_index(State(traj[-1], next_velocity))
                    next_state_prob.append(state_prob(next_state_idx))
                    if passed_finish:
                        r[i, j, next_state_idx] = 0.0
            # Then linearly mix the transition probs with noise
            for j in range(n_actions):
                p[i][j] = (
                    noise * next_state_prob[env._no_accel]
                    + (1.0 - noise) * next_state_prob[j]
                )

    return p, r

import datetime


class ValueIterationResult(NamedTuple):
    q: np.ndarray
    v: np.ndarray
    elapsed: datetime.timedelta
    n_iterations: int


def value_iteration(
    p: np.ndarray,
    r: np.ndarray,
    discount: float,
    epsilon: float = 1e-6,
) -> ValueIterationResult:
    n_states, n_actions, _ = p.shape
    q = np.zeros((n_states, n_actions))
    v = np.zeros(n_states)
    n_iterations = 0
    start = datetime.datetime.now()
    while True:
        n_iterations += 1
        v_old = v.copy()
        for s in range(n_states):
            # Q(s, a) = ∑ p(s, a, s') * (r(s, a, s') + γ v(s')
            for a in range(n_actions):
                q[s, a] = np.dot(p[s, a], r[s, a] + discount * v)
            # V(s) = max_a Q(s, a)
            v[s] = np.max(q[s])
        if np.linalg.norm(v - v_old, ord=np.inf) < epsilon:
            break
    return ValueIterationResult(q, v, datetime.datetime.now() - start, n_iterations)


p, r = get_p_and_r(smalltrack)
vi_result = value_iteration(p, r, discount=0.95)
print(f"Elapsed: {vi_result.elapsed.total_seconds()} n_iter: {vi_result.n_iterations}")

def valid_states(env: RacetrackEnv) -> List[State]:
    states = []
    for y in range(env._max_height):
        for x in range(env._max_width):
            track = env._track[y][x]
            if track == env.WALL:
                continue
            for y_velocity in range(5):
                for x_velocity in range(5):
                    state = State(np.array([y,x]), np.array([y_velocity, x_velocity]))
                    if track != env.START and (x_velocity > 0 or y_velocity > 0):
                        states.append(state)
    return states

# class MonteCarloAgent:
#     def __init__(self, env: RacetrackEnv,
#                     n_episodes: int,
#                     record_all_values: bool = False,
#                     seed: int = 999,):
#         n_states = env._max_width*env._max_height*25
#         n_actions = len(env._actions)
#         random_state = np.random.RandomState(seed=seed)
#         self._q_values = np.zeros((n_states, n_actions)) 
#         self._epsilon = 0
#         self._alpha_t = 0
#         self._
        
def on_policy_fist_visit_mc(
    env: RacetrackEnv,
    n_episodes: int,
    discount: float = 0.95,
    epsilon: float = 0.1,
    epsilon_final: float = 0.1,
    record_all_values: bool = False,
    seed: int = 999,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """On-policy first visit Monte-Carlo"""

    n_states = env._max_width * env._max_height * 25
    n_actions = len(env._actions)
    random_state = np.random.RandomState(seed=seed)
    q = np.zeros((n_states, n_actions))
    pi = random_state.randint(9, size=n_states)
    all_values = []
    all_returns = [[[] for _ in range(n_actions)] for _ in range(n_states)]
    possible_starts = valid_states(env)
    max_episode_length = int(np.ceil(np.log(1e-6) / np.log(discount)))
    epsilon_decay = (epsilon - epsilon_final) / n_episodes
    for i in range(n_episodes):
        if record_all_values:
            all_values.append(q.copy())
        state = env.reset()
        visited_states = [env.state_index(state)]
        taken_actions = []
        received_rewards = []
        for _ in range(max_episode_length):
            # ε-soft policy
            if random_state.rand() < epsilon:
                action = random_state.randint(9)
            else:
                action = pi[env.state_index(state)]
            taken_actions.append(action)
            # Step the simulator
            state, reward, termination = env.step(state, action)
            visited_states.append(env.state_index(state))
            received_rewards.append(reward)
            if termination:
                break
        # Below code is the same as mc_es
        # Compute return
        traj_len = len(received_rewards)
        returns = np.zeros(traj_len)
        # Gt = Rt when t = T
        returns[-1] = received_rewards[-1]
        # Iterating from T - 2, T - 1, ..., to 0
        for t in reversed(range(traj_len - 1)):
            # Gt = Rt + γGt+1
            returns[t] = received_rewards[t] + discount * returns[t + 1]
        updated = set()
        # Update the value
        for i, (state, action) in enumerate(zip(visited_states[:-1], taken_actions)):
            # If the state is already visited, skip it
            if (state, action) in updated:
                continue
            updated.add((state, action))
            all_returns[state][action].append(returns[i].item())
            # Q(St, At) ← average(Returns(St, At))
            q[state, action] = np.mean(all_returns[state][action])
            pi[state] = np.argmax(q[state])
        epsilon -= epsilon_decay
    return q, all_values

on_mc_result = on_policy_fist_visit_mc(
    smalltrack,
    3000,
    epsilon=0.5,
    epsilon_final=0.01,
    record_all_values=True,
)

Policy = Callable[[int], int]

def show_rollout(
    env: RacetrackEnv,
    policy: Policy,
    v: np.ndarray = vi_result.v,
    title: Optional[str] = None,
) -> HTML:
    state = env.reset()
    prev_termination = False
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    camera = Camera(fig)
    initial = True
    while True:
        env.render(state=state, movie=True, ax=ax)
        state_idx = env.state_index(state)
        ax.text(3, 0.5, f"V(s): {v[state_idx]:02}", c="red")
        camera.snap()
        if prev_termination:
            break
        state, _, prev_termination = env.step(state, policy(state_idx))
    if title is not None:
        ax.text(3, 0.1, title, c="k")
    return camera.animate(interval=1000)

for q, name in zip([on_mc_result[0]], [ "On-Policy"]):
    animation = show_rollout(smalltrack, lambda i: np.argmax(q[i]), np.argmax(q, axis=-1), name)
    # animation.save('animation.mp4', writer='ffmpeg')
    plt.show()
    