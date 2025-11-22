#!/usr/bin/env python3
"""
Biophysical chemotaxis trainer (PPO+GAE+Lagrangian) – multi-agent, interacting,
evolving morphologies (size/shape/speed) with comprehensive
biophysical, thermodynamic and information-theoretic metrics logging.

Key features:
- Multiple agents in a shared concentration field.
- Each agent has its own size, shape, and speed.
- Physical interactions: soft repulsion, density-based dissipation.
- Cooperative & competitive terms: reward sharing, crowding penalties.
- Evolution of morphology based on chemotactic performance vs dissipation.
- Per-update CSV logs:
    - biophys_metrics.csv
    - gamma_history.csv
    - morphology_history.csv
- Simple plots: batch_total_perf.png, morphology_evolution.png

CLI:
    python biophys_multiagent_evo.py --help
"""

import os
import math
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Bernoulli


# -----------------------------
# Concentration field
# -----------------------------

def make_patchy_field(seed=19, n_plumes=6, xmin=-20, xmax=20, ymin=-20, ymax=20):
    rng = np.random.RandomState(seed)
    centers = rng.uniform([xmin, ymin], [xmax, ymax], (n_plumes, 2))
    amps    = np.abs(rng.randn(n_plumes)) * 2.0
    widths  = rng.uniform(1.0, 4.0, n_plumes)

    def concentration_field(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        field = np.zeros_like(x, dtype=float)
        for (cx, cy), A, w in zip(centers, amps, widths):
            field += A * np.exp(
                -0.5 * (((x - cx)/w)**2 + ((y - cy)/w)**2)
            )
        field += 0.03 * np.sin(0.2 * x)
        field += 0.02 * np.cos(0.15 * y)
        return np.clip(field, 1e-8, None)

    meta = {"centers": centers.tolist(), "amps": amps.tolist(), "widths": widths.tolist()}
    return concentration_field, meta


# -----------------------------
# Single-agent Biophysical environment
# -----------------------------

class BiophysEnv:
    def __init__(self,
                 concentration_field,
                 rng=None,
                 dt=0.05,
                 v=1.0,
                 D_r=0.5,
                 kappa=3.0,
                 tau_m=20.0,
                 A0=0.5,
                 run_cost=0.01,
                 energy_per_methyl=0.02,
                 probe_eps=0.5,
                 init_pos=(0.0, 0.0),
                 # morphology / motility
                 size=1.0,
                 shape=1.0,
                 speed=None):
        """
        Single chemotactic agent in a static concentration field.

        Morphology:
            size  ~ effective radius
            shape ~ anisotropy / aspect ratio
            speed ~ run speed (overrides v if given)
        """
        self.cf = concentration_field
        self.dt = dt

        # morphology / motility
        self.size = float(size)
        self.shape = float(shape)
        self.v = float(speed) if speed is not None else float(v)

        # rotational diffusion stored explicitly (for evolution)
        self.D_r = float(D_r)
        self.sqrt_2Dr_dt = math.sqrt(2.0 * self.D_r * self.dt)
        self.kappa = float(kappa)
        self.tau_m = float(tau_m)
        self.A0 = float(A0)

        self.run_cost = float(run_cost)
        self.energy_per_methyl = float(energy_per_methyl)
        self.probe_eps = float(probe_eps)

        # thermodynamic params (tunable)
        self.reward_scale = 1.0
        self.time_penalty = 0.001
        self.kB_T = 1.0

        self.rng = rng if rng is not None else np.random.RandomState()
        self.reset(init_pos)

    def reset(self, init_pos=(0.0, 0.0)):
        self.x, self.y = init_pos
        self.theta = self.rng.rand() * 2*np.pi
        self.m = 0.0
        c0 = float(self.cf(self.x, self.y))
        self.A = self._sense(c0, self.m)
        self.prev_A = self.A
        return self._observe()

    def _sense(self, c, m):
        c = max(c, 1e-8)
        return 1.0 / (1.0 + math.exp(-(math.log(c) - m)))

    def _sample_tumble(self):
        return self.rng.vonmises(mu=0.0, kappa=self.kappa)

    def _observe(self):
        c = float(self.cf(self.x, self.y))
        # [x, y, theta, c, A, m, size, shape, v]
        return np.array(
            [self.x,
             self.y,
             self.theta,
             c,
             self.A,
             self.m,
             self.size,
             self.shape,
             self.v],
            dtype=np.float32
        )

    def _symmetric_probe(self, eps=None):
        if eps is None:
            eps = self.probe_eps
        fx_p = self.x + eps * math.cos(self.theta)
        fy_p = self.y + eps * math.sin(self.theta)
        fx_m = self.x - eps * math.cos(self.theta)
        fy_m = self.y - eps * math.sin(self.theta)
        return float(self.cf(fx_p, fy_p)), float(self.cf(fx_m, fy_m))

    def set_morphology(self, size=None, shape=None, speed=None,
                       D_r=None, kappa=None,
                       run_cost=None, energy_per_methyl=None):
        """Update morphology + associated motility/energetic parameters."""
        if size is not None:
            self.size = float(size)
        if shape is not None:
            self.shape = float(shape)
        if speed is not None:
            self.v = float(speed)
        if D_r is not None:
            self.D_r = float(D_r)
            self.sqrt_2Dr_dt = math.sqrt(2.0 * self.D_r * self.dt)
        if kappa is not None:
            self.kappa = float(kappa)
        if run_cost is not None:
            self.run_cost = float(run_cost)
        if energy_per_methyl is not None:
            self.energy_per_methyl = float(energy_per_methyl)

    def step(self, action, gamma=None):
        c_before = float(self.cf(self.x, self.y))
        self.prev_A = self.A
        self.A = self._sense(c_before, self.m)
        dA = self.A - self.prev_A

        # movement
        if action == 0:
            # tumble
            self.theta = (self.theta + self._sample_tumble()) % (2*np.pi)
        else:
            # run
            self.theta = (self.theta + self.rng.normal(scale=self.sqrt_2Dr_dt)) % (2*np.pi)
            self.x += self.v * math.cos(self.theta)
            self.y += self.v * math.sin(self.theta)

        c_after = float(self.cf(self.x, self.y))
        c_plus, c_minus = self._symmetric_probe(self.probe_eps)

        # methylation update
        tau_eff = self.tau_m
        if gamma is not None:
            tau_eff = max(1.0, self.tau_m / (1.0 + 0.01 * (gamma - 1.0)))
        methyl_dm = (self.A - self.A0) * (self.dt / tau_eff)
        self.m += methyl_dm
        methyl_E = self.energy_per_methyl * abs(methyl_dm)

        run_energy = self.run_cost * (1.0 if action == 1 else 0.0)

        # performance signal
        eps = max(self.probe_eps, 1e-6)
        log_cp = math.log(max(c_plus, 1e-12))
        log_cm = math.log(max(c_minus, 1e-12))
        log_grad = (log_cp - log_cm) / (2.0 * eps)
        forward_alignment = (c_plus - c_before) / max(eps, 1e-8)

        grad_term = self.reward_scale * log_grad
        forward_term = 0.1 * forward_alignment
        perf_reward = grad_term + forward_term - self.time_penalty
        perf_reward = float(max(-5.0, min(5.0, perf_reward)))

        E_dissipated = run_energy + methyl_E
        entropy_approx = E_dissipated / max(1e-12, self.kB_T)

        info = {
            "c_before": c_before,
            "c_after": c_after,
            "c_plus": c_plus,
            "c_minus": c_minus,
            "log_grad": log_grad,
            "forward_alignment": forward_alignment,
            "run_energy": run_energy,
            "methyl_E": methyl_E,
            "E_dissipated": E_dissipated,
            "entropy_approx": entropy_approx,
            "A": self.A,
            "m": self.m,
            "dA": dA,
            "action": action,
            # morphology / motility metadata
            "size": self.size,
            "shape": self.shape,
            "speed": self.v
        }

        return self._observe(), perf_reward, False, info


# -----------------------------
# Multi-Agent wrapper
# -----------------------------

class MultiAgentBiophysEnv:
    """
    Multi-agent wrapper around BiophysEnv.

    - Agents share the same concentration field.
    - Each agent has its own morphology (size, shape, speed).
    - Interactions: soft repulsion, density-based dissipation.
    - Cooperative & competitive terms: reward sharing, crowding penalties.
    - Evolution of morphology based on fitness.
    """
    def __init__(self,
                 concentration_field,
                 num_agents=8,
                 seed=19,
                 # interaction params
                 interaction_radius=1.0,
                 interaction_strength=1.0,
                 interaction_energy=0.02,
                 # coop / competition
                 coop_radius=2.0,
                 coop_bonus=0.05,
                 crowd_radius=3.0,
                 crowd_penalty=0.1,
                 crowd_threshold=2,
                 density_sigma=2.5,
                 density_factor=0.15,
                 # evolution params
                 evo_interval=10,
                 evo_elite_frac=0.5,
                 evo_mutation_std=(0.05, 0.1, 0.1)):

        self.cf = concentration_field
        self.num_agents = num_agents
        self.rng = np.random.RandomState(seed)

        # morphology ranges
        self.size_range = (0.5, 1.5)
        self.shape_range = (0.5, 2.0)
        self.speed_range = (0.5, 1.8)

        # interactions
        self.interaction_radius = float(interaction_radius)
        self.interaction_strength = float(interaction_strength)
        self.interaction_energy = float(interaction_energy)

        # cooperative / competitive
        self.coop_radius = float(coop_radius)
        self.coop_bonus = float(coop_bonus)
        self.crowd_radius = float(crowd_radius)
        self.crowd_penalty = float(crowd_penalty)
        self.crowd_threshold = int(crowd_threshold)
        self.density_sigma = float(density_sigma)
        self.density_factor = float(density_factor)

        # evolution
        self.evo_interval = int(evo_interval)
        self.evo_elite_frac = float(evo_elite_frac)
        self.evo_mutation_std = evo_mutation_std  # (size, shape, speed)
        self.evo_scores = np.zeros(num_agents, dtype=float)
        self.episodes_since_evo = 0

        self._build_population()

    def _build_population(self):
        self.agents = []
        for _ in range(self.num_agents):
            size = self.rng.uniform(*self.size_range)
            shape = self.rng.uniform(*self.shape_range)
            speed = self.rng.uniform(*self.speed_range)

            base_D_r = 0.5
            base_kappa = 3.0
            base_run_cost = 0.02
            base_energy_per_methyl = 0.02

            D_r = base_D_r / (size * shape)
            kappa = base_kappa * shape
            run_cost = base_run_cost * size
            energy_per_methyl = base_energy_per_methyl * size

            env = BiophysEnv(
                concentration_field=self.cf,
                rng=np.random.RandomState(self.rng.randint(1, 1_000_000_000)),
                dt=0.05,
                v=speed,
                D_r=D_r,
                kappa=kappa,
                tau_m=20.0,
                A0=0.5,
                run_cost=run_cost,
                energy_per_methyl=energy_per_methyl,
                probe_eps=0.8,
                init_pos=(0.0, 0.0),
                size=size,
                shape=shape,
                speed=speed
            )
            env.reward_scale = 1.0
            env.time_penalty = 0.001
            env.kB_T = 1.0
            self.agents.append(env)

    def reset(self):
        obs_list = []
        for env in self.agents:
            o = env.reset()
            obs_list.append(o)
        return np.stack(obs_list, axis=0)

    def step(self, actions, gammas=None):
        """
        actions: (N,)
        gammas:  (N,) or None
        """
        obs_next = []
        rewards = []
        dones = []
        infos = []

        for i, env in enumerate(self.agents):
            gamma_i = None if gammas is None else gammas[i]
            o2, r, d, info = env.step(int(actions[i]), gamma=gamma_i)
            obs_next.append(o2)
            rewards.append(r)
            dones.append(d)
            infos.append(info)

        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.asarray(dones, dtype=bool)

        # apply interactions (modifies positions, costs, entropies, rewards)
        self._apply_interactions(infos, rewards)

        # recompute observations after interactions
        obs_next = [env._observe() for env in self.agents]

        return np.stack(obs_next, axis=0), rewards, dones, infos

    # ---------------- Interactions ----------------

    def _apply_interactions(self, infos, rewards):
        N = self.num_agents
        xs = np.array([env.x for env in self.agents])
        ys = np.array([env.y for env in self.agents])
        sizes = np.array([env.size for env in self.agents])

        # Pairwise soft repulsion + interaction energy
        for i in range(N):
            for j in range(i + 1, N):
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                dist = math.hypot(dx, dy)
                if dist == 0.0:
                    dx = 1e-3 * (self.rng.rand() - 0.5)
                    dy = 1e-3 * (self.rng.rand() - 0.5)
                    dist = math.hypot(dx, dy)

                min_dist = self.interaction_radius * 0.5 * (sizes[i] + sizes[j])
                if dist < min_dist:
                    overlap = min_dist - dist
                    ux = dx / dist
                    uy = dy / dist
                    push = self.interaction_strength * overlap
                    # symmetric push
                    self.agents[i].x -= 0.5 * push * ux
                    self.agents[i].y -= 0.5 * push * uy
                    self.agents[j].x += 0.5 * push * ux
                    self.agents[j].y += 0.5 * push * uy

                    # extra energetic cost
                    extra_E = self.interaction_energy * overlap
                    for k in (i, j):
                        infos[k]["E_interact"] = infos[k].get("E_interact", 0.0) + extra_E
                        infos[k]["E_dissipated"] += extra_E
                        env_k = self.agents[k]
                        infos[k]["entropy_approx"] = infos[k]["E_dissipated"] / max(1e-12, env_k.kB_T)

        # Cooperative reward sharing (neighbors within coop_radius)
        xs = np.array([env.x for env in self.agents])
        ys = np.array([env.y for env in self.agents])
        for i in range(N):
            neighbors = 0
            for j in range(N):
                if i == j:
                    continue
                d = math.hypot(xs[j] - xs[i], ys[j] - ys[i])
                if d < self.coop_radius:
                    neighbors += 1
            rewards[i] += self.coop_bonus * neighbors

        # Competitive crowding penalty (extra dissipation)
        for i in range(N):
            neighbors = 0
            for j in range(N):
                if j == i:
                    continue
                d = math.hypot(xs[j] - xs[i], ys[j] - ys[i])
                if d < self.crowd_radius:
                    neighbors += 1

            if neighbors > self.crowd_threshold:
                excess = neighbors - self.crowd_threshold
                extra_cost = self.crowd_penalty * excess
                infos[i]["E_dissipated"] += extra_cost
                env_i = self.agents[i]
                infos[i]["entropy_approx"] = infos[i]["E_dissipated"] / max(1e-12, env_i.kB_T)

        # Density-based dissipation (smooth density field)
        for i in range(N):
            density = 0.0
            for j in range(N):
                if i == j:
                    continue
                dx = xs[j] - xs[i]
                dy = ys[j] - ys[i]
                d2 = dx*dx + dy*dy
                density += math.exp(-d2 / (2 * self.density_sigma**2))
            extra_E = self.density_factor * density
            infos[i]["E_dissipated"] += extra_E
            env_i = self.agents[i]
            infos[i]["entropy_approx"] = infos[i]["E_dissipated"] / max(1e-12, env_i.kB_T)

    # ---------------- Evolution ----------------

    def record_episode_fitness(self, total_perf, total_cost, beta=0.1):
        """
        total_perf, total_cost: arrays of length N for one episode.
        beta: trade-off coefficient: fitness = perf - beta * cost
        """
        total_perf = np.asarray(total_perf, dtype=float)
        total_cost = np.asarray(total_cost, dtype=float)
        fitness = total_perf - beta * total_cost

        self.evo_scores += fitness
        self.episodes_since_evo += 1

        if self.episodes_since_evo >= self.evo_interval:
            self._evolve_population()
            self.evo_scores[:] = 0.0
            self.episodes_since_evo = 0

    def _evolve_population(self):
        N = self.num_agents
        if N == 0:
            return

        indices = np.argsort(-self.evo_scores)  # descending
        elite_k = max(1, int(self.evo_elite_frac * N))
        elites = indices[:elite_k]

        sizes = np.array([env.size for env in self.agents])
        shapes = np.array([env.shape for env in self.agents])
        speeds = np.array([env.v for env in self.agents])

        new_sizes = sizes.copy()
        new_shapes = shapes.copy()
        new_speeds = speeds.copy()

        sigma_size, sigma_shape, sigma_speed = self.evo_mutation_std

        for i in range(N):
            parent = self.rng.choice(elites)
            new_sizes[i] = sizes[parent] + sigma_size * self.rng.randn()
            new_shapes[i] = shapes[parent] + sigma_shape * self.rng.randn()
            new_speeds[i] = speeds[parent] + sigma_speed * self.rng.randn()

        # clamp
        new_sizes = np.clip(new_sizes, *self.size_range)
        new_shapes = np.clip(new_shapes, *self.shape_range)
        new_speeds = np.clip(new_speeds, *self.speed_range)

        base_D_r = 0.5
        base_kappa = 3.0
        base_run_cost = 0.02
        base_energy_per_methyl = 0.02

        for i, env in enumerate(self.agents):
            size = new_sizes[i]
            shape = new_shapes[i]
            speed = new_speeds[i]
            D_r = base_D_r / (size * shape)
            kappa = base_kappa * shape
            run_cost = base_run_cost * size
            energy_per_methyl = base_energy_per_methyl * size

            env.set_morphology(
                size=size,
                shape=shape,
                speed=speed,
                D_r=D_r,
                kappa=kappa,
                run_cost=run_cost,
                energy_per_methyl=energy_per_methyl
            )


# -----------------------------
# Actor-Critic network
# -----------------------------

class ActorCritic(nn.Module):
    def __init__(self, input_dim=9, hidden=128, output_gamma=False):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden, 1)   # logit for Bernoulli
        self.value_head = nn.Linear(hidden, 1)
        self.output_gamma = output_gamma
        if output_gamma:
            self.gamma_head = nn.Linear(hidden, 1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        h = self.shared(x)
        logit = self.policy_head(h).squeeze(-1)
        p_run = torch.sigmoid(logit)
        value = self.value_head(h).squeeze(-1)
        gamma = None
        if self.output_gamma:
            gamma_raw = self.gamma_head(h).squeeze(-1)
            gamma = torch.nn.functional.softplus(gamma_raw) + 1e-6
        return p_run, value, gamma


# -----------------------------
# Utility estimators for info-theoretic metrics
# -----------------------------

def fisher_activity_est(A, eps_noise=0.01):
    s = A * (1.0 - A)
    return (s ** 2) / (eps_noise ** 2 + 1e-12)


def gaussian_MI_est(var_signal, var_noise):
    if var_noise <= 0 or var_signal <= 0:
        return 0.0
    snr = var_signal / var_noise
    return 0.5 * math.log(1.0 + snr + 1e-12)


def kl_bernoulli(p, q):
    p = float(np.clip(p, 1e-8, 1-1e-8))
    q = float(np.clip(q, 1e-8, 1-1e-8))
    return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))


def mutual_info_pearson_est(x, y):
    if len(x) < 3:
        return 0.0
    x = np.asarray(x); y = np.asarray(y)
    xm = x - x.mean(); ym = y - y.mean()
    denom = math.sqrt(np.sum(xm*xm) * np.sum(ym*ym))
    if denom == 0:
        return 0.0
    rho = float(np.sum(xm*ym) / denom)
    rho = np.clip(rho, -0.9999, 0.9999)
    return -0.5 * math.log(1.0 - rho*rho)


# -----------------------------
# Trainer (PPO+GAE+Lagrangian)
# -----------------------------

def train_with_full_metrics(
        outdir="output_biophys_metrics",
        seed=19,
        episodes=800,
        episode_len=400,
        batch_episodes=16,
        lr=1e-4,
        hidden=256,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        train_gamma=False,
        ppo_epochs=4,
        mini_batch_size=512,
        clip_eps=0.2,
        gamma_discount=0.99,
        gae_lambda=0.95,
        value_clip=True,
        # Lagrangian
        cost_limit=5.0,
        lambda_lr=3e-3,
        init_lambda=0.0,
        save_stepwise=False
    ):
    os.makedirs(outdir, exist_ok=True)
    if save_stepwise:
        print("Warning: save_stepwise currently not supported in multi-agent version; ignoring.")
        save_stepwise = False

    # seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # field
    cf, meta = make_patchy_field(seed=seed, n_plumes=6)

    # multi-agent environment (single world reused)
    env = MultiAgentBiophysEnv(
        cf,
        num_agents=8,
        seed=seed,
        interaction_radius=1.0,
        interaction_strength=1.0,
        interaction_energy=0.02,
        coop_radius=2.0,
        coop_bonus=0.05,
        crowd_radius=3.0,
        crowd_penalty=0.1,
        crowd_threshold=2,
        density_sigma=2.5,
        density_factor=0.15,
        evo_interval=10,
        evo_elite_frac=0.5,
        evo_mutation_std=(0.05, 0.1, 0.1)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(input_dim=9, hidden=hidden, output_gamma=train_gamma).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    metrics_records = []
    gamma_hist = []
    morph_records = []

    lambda_dual = init_lambda
    total_episodes = episodes
    ep_counter = 0

    # helper: run one multi-agent episode
    def run_episode(env, deterministic=False):
        N = env.num_agents

        all_obs = [[] for _ in range(N)]
        all_acts = [[] for _ in range(N)]
        all_rews = [[] for _ in range(N)]
        all_costs = [[] for _ in range(N)]
        all_infos = [[] for _ in range(N)]
        all_vals = [[] for _ in range(N)]
        all_logps = [[] for _ in range(N)]
        all_p = [[] for _ in range(N)]

        obs = env.reset()  # (N, obs_dim)

        for t in range(episode_len):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            p_run, v, gamma_t = net(obs_tensor)

            if deterministic:
                acts = (p_run.detach().cpu().numpy() > 0.5).astype(int)
            else:
                p_np = p_run.detach().cpu().numpy()
                acts = (np.random.rand(N) < p_np).astype(int)

            v_np = v.detach().cpu().numpy()
            p_np = np.clip(p_run.detach().cpu().numpy(), 1e-8, 1 - 1e-8)
            logp = np.where(acts == 1, np.log(p_np), np.log(1 - p_np))

            gammas_np = None
            if gamma_t is not None:
                gammas_np = gamma_t.detach().cpu().numpy()

            obs2, rew, done, infos = env.step(acts, gammas=gammas_np)

            for i in range(N):
                all_obs[i].append(obs[i])
                all_acts[i].append(int(acts[i]))
                all_rews[i].append(float(rew[i]))
                all_costs[i].append(float(infos[i].get("E_dissipated", 0.0)))
                all_infos[i].append(infos[i])
                all_vals[i].append(float(v_np[i]))
                all_logps[i].append(float(logp[i]))
                all_p[i].append(float(p_np[i]))

            obs = obs2

        # per-agent totals for evolution
        total_perf = np.array([sum(all_rews[i]) for i in range(N)], dtype=float)
        total_cost = np.array([sum(all_costs[i]) for i in range(N)], dtype=float)
        env.record_episode_fitness(total_perf, total_cost, beta=0.1)

        # convert to list of per-agent trajectories
        trajectories = []
        for i in range(N):
            trajectories.append({
                "obs": all_obs[i],
                "acts": all_acts[i],
                "perf_rews": all_rews[i],
                "costs": all_costs[i],
                "info": all_infos[i],
                "vals": all_vals[i],
                "logps": all_logps[i],
                "p_runs": all_p[i]
            })
        return trajectories

    # outer PPO loop
    update_idx = 0
    while ep_counter < total_episodes:
        batch_trajs = []
        for _ in range(batch_episodes):
            traj_list = run_episode(env, deterministic=False)
            batch_trajs.extend(traj_list)
            ep_counter += 1
            if ep_counter >= total_episodes:
                break

        # Flatten and compute GAE using combined reward r = perf - lambda*cost
        flat_obs = []
        flat_acts = []
        flat_returns = []
        flat_oldvals = []
        flat_oldlogp = []
        flat_costs = []
        flat_p_runs = []
        per_episode_metrics = []

        for traj in batch_trajs:
            rews = np.array(traj["perf_rews"], dtype=np.float32)
            costs = np.array(traj["costs"], dtype=np.float32)
            vals = np.array(traj["vals"], dtype=np.float32)
            p_runs = np.array(traj["p_runs"], dtype=np.float32)
            infos = traj["info"]

            T = len(rews)
            if T == 0:
                continue

            r_comb = rews - lambda_dual * costs

            v_next = np.append(vals[1:], 0.0)
            deltas = r_comb + gamma_discount * v_next - vals
            advs = np.zeros_like(deltas, dtype=np.float32)
            gae = 0.0
            for t in reversed(range(T)):
                gae = deltas[t] + gamma_discount * gae_lambda * gae
                advs[t] = gae
            returns = advs + vals

            flat_obs.extend(traj["obs"])
            flat_acts.extend(traj["acts"])
            flat_returns.extend(returns.tolist())
            flat_oldvals.extend(vals.tolist())
            flat_oldlogp.extend(traj["logps"])
            flat_costs.extend(costs.tolist())
            flat_p_runs.extend(p_runs.tolist())

            # per-"episode per agent" biophysical metrics
            total_perf = float(rews.sum())
            total_cost = float(costs.sum())
            total_entropy = float(np.sum([inf.get("entropy_approx", 0.0) for inf in infos]))
            run_fraction = float(np.mean([1.0 if a == 1 else 0.0 for a in traj["acts"]]))

            A_vals = np.array([inf.get("A", np.nan) for inf in infos], dtype=np.float32)
            c_vals = np.array([inf.get("c_before", np.nan) for inf in infos], dtype=np.float32)

            fisher_vals = [fisher_activity_est(a) for a in A_vals if not np.isnan(a)]
            mean_fisher = float(np.nanmean(fisher_vals)) if len(fisher_vals) > 0 else 0.0

            logc = np.log(np.clip(c_vals, 1e-12, None))
            var_A = float(np.nanvar(A_vals)) if len(A_vals) > 0 else 0.0
            eps_A = 0.01
            mi_A_logc = gaussian_MI_est(var_A, eps_A**2)

            c_next = np.append(c_vals[1:], c_vals[-1])
            pred_info = mutual_info_pearson_est(A_vals, c_next)

            kl_vals = []
            for i in range(1, len(p_runs)):
                kl_vals.append(kl_bernoulli(p_runs[i-1], p_runs[i]))
            mean_policy_kl = float(np.mean(kl_vals)) if len(kl_vals) > 0 else 0.0

            logprob_ratios = []
            kl_actionwise = []
            for i in range(len(p_runs) - 1):
                p_curr = float(p_runs[i])
                p_next = float(p_runs[i+1])
                a = traj["acts"][i]
                if a == 1:
                    logp_curr = math.log(p_curr + 1e-8)
                    logp_next = math.log(p_next + 1e-8)
                else:
                    logp_curr = math.log(1.0 - p_curr + 1e-8)
                    logp_next = math.log(1.0 - p_next + 1e-8)
                logprob_ratios.append(logp_curr - logp_next)
                kl_actionwise.append(kl_bernoulli(p_curr, p_next))

            mean_logprob_ratio = float(np.mean(logprob_ratios)) if len(logprob_ratios) > 0 else 0.0
            mean_kl_actionwise = float(np.mean(kl_actionwise)) if len(kl_actionwise) > 0 else 0.0
            efficiency = total_perf / (total_cost + 1e-12)

            # morphology stats (per-agent episode)
            size_vals = [inf.get("size", np.nan) for inf in infos]
            shape_vals = [inf.get("shape", np.nan) for inf in infos]
            speed_vals = [inf.get("speed", np.nan) for inf in infos]
            mean_size = float(np.nanmean(size_vals)) if len(size_vals) > 0 else np.nan
            mean_shape = float(np.nanmean(shape_vals)) if len(shape_vals) > 0 else np.nan
            mean_speed = float(np.nanmean(speed_vals)) if len(speed_vals) > 0 else np.nan

            per_episode_metrics.append({
                "total_perf": total_perf,
                "total_cost": total_cost,
                "total_entropy": total_entropy,
                "run_fraction": run_fraction,
                "mean_fisher": mean_fisher,
                "mi_A_logc": mi_A_logc,
                "pred_info": pred_info,
                "mean_policy_kl": mean_policy_kl,
                "mean_logprob_ratio": mean_logprob_ratio,
                "mean_kl_actionwise": mean_kl_actionwise,
                "efficiency": efficiency,
                "mean_size": mean_size,
                "mean_shape": mean_shape,
                "mean_speed": mean_speed
            })

        if len(flat_obs) == 0:
            continue

        # tensors
        obs_tensor = torch.from_numpy(np.asarray(flat_obs, dtype=np.float32)).to(device)
        acts_tensor = torch.tensor(flat_acts, dtype=torch.float32, device=device)
        returns_tensor = torch.tensor(flat_returns, dtype=torch.float32, device=device)
        old_values_tensor = torch.tensor(flat_oldvals, dtype=torch.float32, device=device)
        old_logps_tensor = torch.tensor(flat_oldlogp, dtype=torch.float32, device=device)

        advantages = returns_tensor - old_values_tensor
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        N_data = obs_tensor.shape[0]
        idxs = np.arange(N_data)

        # PPO epochs
        last_loss = 0.0
        for epoch in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N_data, mini_batch_size):
                mb_idx = idxs[start:start+mini_batch_size].tolist()
                mb_obs = obs_tensor[mb_idx]
                mb_acts = acts_tensor[mb_idx]
                mb_returns = returns_tensor[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_oldlogp = old_logps_tensor[mb_idx]
                mb_oldvals = old_values_tensor[mb_idx]

                mb_p, mb_v, mb_g = net(mb_obs)
                mb_p = mb_p.clamp(1e-8, 1-1e-8)
                new_logp = mb_acts * torch.log(mb_p) + (1.0 - mb_acts) * torch.log(1.0 - mb_p)
                ratio = torch.exp(new_logp - mb_oldlogp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if value_clip:
                    v_clipped = mb_oldvals + torch.clamp(mb_v - mb_oldvals, -clip_eps, clip_eps)
                    value_loss_unclipped = (mb_v - mb_returns).pow(2)
                    value_loss_clipped = (v_clipped - mb_returns).pow(2)
                    value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (mb_v - mb_returns).pow(2).mean()

                ent = - (mb_p * torch.log(mb_p) + (1.0 - mb_p) * torch.log(1.0 - mb_p))
                entropy_loss = ent.mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()
                last_loss = float(loss.item())

        # update dual variable lambda
        per_episode_costs = [m["total_cost"] for m in per_episode_metrics] if len(per_episode_metrics) > 0 else [0.0]
        avg_cost = float(np.mean(per_episode_costs))
        lambda_dual = max(0.0, lambda_dual + lambda_lr * (avg_cost - cost_limit))

        # aggregate metrics across the batch
        batch_total_perf = float(np.sum([m["total_perf"] for m in per_episode_metrics]))
        batch_total_cost = float(np.sum([m["total_cost"] for m in per_episode_metrics]))
        batch_eff = float(np.mean([m["efficiency"] for m in per_episode_metrics]))
        batch_mean_fisher = float(np.mean([m["mean_fisher"] for m in per_episode_metrics]))
        batch_mi = float(np.mean([m["mi_A_logc"] for m in per_episode_metrics]))
        batch_mean_logprob_ratio = float(np.mean([m["mean_logprob_ratio"] for m in per_episode_metrics]))
        batch_mean_kl_actionwise = float(np.mean([m["mean_kl_actionwise"] for m in per_episode_metrics]))

        update_idx += 1
        metrics_records.append({
            "update": update_idx,
            "episode_up_to": ep_counter,
            "batch_total_perf": batch_total_perf,
            "batch_total_cost": batch_total_cost,
            "batch_efficiency": batch_eff,
            "batch_mean_fisher": batch_mean_fisher,
            "batch_mi_A_logc": batch_mi,
            "batch_mean_logprob_ratio": batch_mean_logprob_ratio,
            "batch_mean_kl_actionwise": batch_mean_kl_actionwise,
            "avg_cost": avg_cost,
            "lambda": lambda_dual,
            "loss": last_loss
        })

        gamma_hist.append({"episode": ep_counter, "avg_gamma": np.nan, "lambda": lambda_dual})

        # environment-level morphology snapshot after this update
        sizes_env = [a.size for a in env.agents]
        shapes_env = [a.shape for a in env.agents]
        speeds_env = [a.v for a in env.agents]
        morph_records.append({
            "update": update_idx,
            "episode_up_to": ep_counter,
            "mean_size": float(np.mean(sizes_env)),
            "mean_shape": float(np.mean(shapes_env)),
            "mean_speed": float(np.mean(speeds_env))
        })

        print(f"[{datetime.now().isoformat()}] up-to-episode {ep_counter} "
              f"avg_cost={avg_cost:.4f} lambda={lambda_dual:.4f} batch_eff={batch_eff:.4f}")

    # save metrics
    df_metrics = pd.DataFrame(metrics_records)
    df_gamma = pd.DataFrame(gamma_hist)
    df_morph = pd.DataFrame(morph_records)

    df_metrics.to_csv(Path(outdir)/"biophys_metrics.csv", index=False)
    df_gamma.to_csv(Path(outdir)/"gamma_history.csv", index=False)
    df_morph.to_csv(Path(outdir)/"morphology_history.csv", index=False)

    # simple plots
    try:
        if len(df_metrics) > 0:
            plt.figure(figsize=(8,4))
            plt.plot(df_metrics["episode_up_to"], df_metrics["batch_total_perf"])
            plt.xlabel("Episode")
            plt.ylabel("Batch total performance")
            plt.title("Batch total performance")
            plt.savefig(Path(outdir)/"batch_total_perf.png", dpi=150)
            plt.close()
    except Exception:
        pass

    try:
        if len(df_morph) > 0:
            plt.figure(figsize=(10,4))
            plt.plot(df_morph["episode_up_to"], df_morph["mean_size"], label="mean size")
            plt.plot(df_morph["episode_up_to"], df_morph["mean_shape"], label="mean shape")
            plt.plot(df_morph["episode_up_to"], df_morph["mean_speed"], label="mean speed")
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Morphology")
            plt.title("Morphology evolution")
            plt.savefig(Path(outdir)/"morphology_evolution.png", dpi=150)
            plt.close()
    except Exception:
        pass

    print("Training + metrics collection complete. Artifacts in:", outdir)
    return df_metrics, df_gamma, df_morph


# -----------------------------
# CLI
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="output_biophys_metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--episode_len", type=int, default=400)
    parser.add_argument("--batch_episodes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=512)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--cost_limit", type=float, default=5.0)
    parser.add_argument("--lambda_lr", type=float, default=3e-3)
    parser.add_argument("--save_stepwise", action="store_true")
    parser.add_argument("--train_gamma", action="store_true")
    args = parser.parse_args()

    train_with_full_metrics(outdir=args.outdir, seed=args.seed, episodes=args.episodes,
                            episode_len=args.episode_len, batch_episodes=args.batch_episodes,
                            lr=args.lr, hidden=args.hidden, ppo_epochs=args.ppo_epochs,
                            mini_batch_size=args.mini_batch_size, clip_eps=args.clip_eps,
                            cost_limit=args.cost_limit, lambda_lr=args.lambda_lr,
                            save_stepwise=args.save_stepwise, train_gamma=args.train_gamma)

if __name__ == "__main__":
    main()
