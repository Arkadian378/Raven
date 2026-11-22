"""Adapters to the standard RL ecosystems.

* :class:`LabyrinthAEC`     -- PettingZoo AEC environment (multi-agent, turn based)
* :class:`SingleAgentEnv`   -- Gymnasium env: one learning seat, the other seats
                               are played by fixed policies (baselines, snapshots)
"""
from __future__ import annotations

import numpy as np

from .core import NUM_ACTIONS
from .env import LabyrinthEnv


def _spaces(env: LabyrinthEnv):
    from gymnasium import spaces

    obs = spaces.Dict({
        "planes": spaces.Box(0.0, 1.0, env.planes_shape, np.float32),
        "vector": spaces.Box(0.0, 1.0, env.vector_shape, np.float32),
        "action_mask": spaces.MultiBinary(NUM_ACTIONS),
    })
    return obs, spaces.Discrete(NUM_ACTIONS)


try:
    from pettingzoo import AECEnv
except ImportError:  # pettingzoo is optional
    AECEnv = object


class LabyrinthAEC(AECEnv):
    metadata = {"render_modes": ["rgb_array", "ansi"], "name": "labyrinth_v0", "is_parallelizable": False}

    def __init__(self, **kwargs):
        super().__init__()
        self.env = LabyrinthEnv(**kwargs)
        self.render_mode = self.env.render_mode
        self.possible_agents = [f"player_{i}" for i in range(self.env.num_players)]
        obs_space, act_space = _spaces(self.env)
        self.observation_spaces = {a: obs_space for a in self.possible_agents}
        self.action_spaces = {a: act_space for a in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed)
        self.agents = list(self.possible_agents)
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}
        self.agent_selection = self.possible_agents[self.env.current_player]

    def observe(self, agent):
        obs = self.env.observe(self.possible_agents.index(agent))
        obs["action_mask"] = obs["action_mask"].astype(np.int8)  # PettingZoo convention
        return obs

    def step(self, action):
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[self.agent_selection] = 0.0
        _, r, term, trunc, info = self.env.step(action)
        for i, a in enumerate(self.possible_agents):
            self.rewards[a] = float(r[i])
            self.terminations[a] = term
            self.truncations[a] = trunc
            self.infos[a] = info
        self.agent_selection = self.possible_agents[self.env.current_player]
        self._accumulate_rewards()

    def render(self):
        return self.env.render()

    def close(self):
        pass


try:
    import gymnasium as gym
    _GymBase = gym.Env
except ImportError:
    _GymBase = object


class SingleAgentEnv(_GymBase):
    """Gymnasium view of one seat; opponents act inside ``step``.

    ``opponents`` is a list of agents with ``act(obs, state, player)`` for the
    other seats (in seat order, skipping the learner). Rewards collected while
    the opponents play are credited to the learner's next transition.
    """

    metadata = LabyrinthEnv.metadata

    def __init__(self, opponents, seat: int | None = 0, **kwargs):
        self.env = LabyrinthEnv(**kwargs)
        self.observation_space, self.action_space = _spaces(self.env)
        self.opponents = opponents
        self.fixed_seat = seat
        self.render_mode = self.env.render_mode

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed)
        n = self.env.num_players
        self.seat = self.fixed_seat if self.fixed_seat is not None else int(self.env.rng.integers(n))
        others = [p for p in range(n) if p != self.seat]
        self.seat_agent = dict(zip(others, self.opponents))
        obs, _, done, info = self._play_opponents(obs, info)
        return obs, info

    def _play_opponents(self, obs, info):
        acc, done = 0.0, False
        while self.env.current_player != self.seat:
            p = self.env.current_player
            a = self.seat_agent[p].act(obs, self.env.state, p)
            obs, r, term, trunc, info = self.env.step(a)
            acc += float(r[self.seat])
            if term or trunc:
                return self.env.observe(self.seat), acc, (term, trunc), info
        return obs, acc, False, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        reward = float(r[self.seat])
        if not (term or trunc):
            obs, extra, done, info = self._play_opponents(obs, info)
            reward += extra
            if done:
                term, trunc = done
        else:
            obs = self.env.observe(self.seat)
        return obs, reward, term, trunc, info

    def action_masks(self):  # sb3-contrib MaskablePPO convention
        return self.env.action_mask()

    def render(self):
        return self.env.render()
