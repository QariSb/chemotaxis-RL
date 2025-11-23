#!/usr/bin/env python3
"""
Biophysical chemotaxis trainer (PPO+GAE+Lagrangian) – multi-agent, interacting,
evolving morphologies (size/shape/speed) with comprehensive
biophysical, thermodynamic and information-theoretic metrics logging.

Aggressive NUMBA refactor (B2-style) + Strategy A:
- All core environment physics, field evaluation, interactions,
  energy dissipation and reward decomposition done in Numba-accelerated
  kernels over flat NumPy arrays.
- Evolution now uses Strategy A: slow-timescale evolution with mutation annealing.
- PyTorch is used only for the Actor-Critic and PPO optimization.
- Environment step is fully vectorized / JITed.

CLI:
    python biophys_multiagent_evo_ablate_numba.py --help
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

from numba import njit, prange


# ============================================================
# Numba-compatible concentration field
# ============================================================

def make_patchy_field(seed=19, n_plumes=6,
                      xmin=-20, xmax=20,
                      ymin=-20, ymax=20):
    """
    Returns (cf function compiled by numba via closure, centers, amps, widths) and meta
    """
    rng = np.random.RandomState(seed)
    centers = rng.uniform([xmin, ymin], [xmax, ymax], (n_plumes, 2))
    amps = np.abs(rng.randn(n_plumes)) * 2.0
    widths = rng.uniform(1.0, 4.0, n_plumes)

    centers = centers.astype(np.float64)
    amps = amps.astype(np.float64)
    widths = widths.astype(np.float64)

    @njit
    def cf(x, y):
        field = 0.0
        for k in range(centers.shape[0]):
            cx = centers[k, 0]
            cy = centers[k, 1]
            A = amps[k]
            w = widths[k]
            dx = x - cx
            dy = y - cy
            field += A * math.exp(-0.5 * ((dx / w) ** 2 + (dy / w) ** 2))
        field += 0.03 * math.sin(0.2 * x)
        field += 0.02 * math.cos(0.15 * y)
        if field < 1e-8:
            field = 1e-8
        return field

    meta = {
        "centers": centers.tolist(),
        "amps": amps.tolist(),
        "widths": widths.tolist(),
    }

    return cf, centers, amps, widths, meta


# ============================================================
# Numba utility kernels
# ============================================================

@njit
def logistic_activity(log_c, m):
    """
    A = 1 / (1 + exp(-(log c - m))).
    """
    x = log_c - m
    if x > 40.0:
        return 1.0
    if x < -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@njit
def fisher_activity_est_numba(A, eps_noise=0.01):
    s = A * (1.0 - A)
    denom = eps_noise * eps_noise + 1e-12
    return (s * s) / denom


@njit
def gaussian_MI_est_numba(var_signal, var_noise):
    if var_noise <= 0.0 or var_signal <= 0.0:
        return 0.0
    snr = var_signal / var_noise
    return 0.5 * math.log(1.0 + snr + 1e-12)


@njit
def mutual_info_pearson_est_numba(x, y):
    n = x.shape[0]
    if n < 3:
        return 0.0
    xm = 0.0
    ym = 0.0
    for i in range(n):
        xm += x[i]
        ym += y[i]
    xm /= n
    ym /= n

    num = 0.0
    sx = 0.0
    sy = 0.0
    for i in range(n):
        dx = x[i] - xm
        dy = y[i] - ym
        num += dx * dy
        sx += dx * dx
        sy += dy * dy
    denom = math.sqrt(sx * sy)
    if denom == 0.0:
        return 0.0
    rho = num / denom
    if rho > 0.9999:
        rho = 0.9999
    if rho < -0.9999:
        rho = -0.9999
    return -0.5 * math.log(1.0 - rho * rho + 1e-12)


@njit
def kl_bernoulli_numba(p, q):
    if p < 1e-8:
        p = 1e-8
    if p > 1.0 - 1e-8:
        p = 1.0 - 1e-8
    if q < 1e-8:
        q = 1e-8
    if q > 1.0 - 1e-8:
        q = 1.0 - 1e-8
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


# ============================================================
# JITted multi-agent step kernel
# ============================================================
@njit
def multiagent_step_kernel(
    # state arrays
    x, y, theta, m,
    size, shape, speed,
    D_r, kappa, run_cost, energy_per_methyl,
    # shared env scalars
    dt, tau_m, A0,
    probe_eps,
    reward_scale, time_penalty,
    kB_T,
    # field params
    centers, amps, widths,
    # interaction params
    interactions_on,
    interaction_radius, interaction_strength, interaction_energy,
    coop_radius, coop_bonus,
    crowd_radius, crowd_penalty, crowd_threshold,
    density_sigma, density_factor,
    # ablation toggles
    grad_term_on, forward_term_on,
    # inputs
    actions, gammas, use_gamma,
    randn,
    # outputs
    perf_reward, cost, entropy_approx,
    A_arr, c_before_arr, c_plus_arr, c_minus_arr,
    log_grad_arr, forward_align_arr,
    run_flags
):
    """
    Vectorized multi-agent step with interactions & reward decomposition.

    All arrays are length N.
    In-place update of x, y, theta, m.
    Fills perf_reward, cost, entropy_approx, and auxiliary logs.
    """
    N = x.shape[0]

    # precompute field for each agent before move
    for i in range(N):
        # concentration at current position
        c_before = 0.0
        for k in range(centers.shape[0]):
            cx = centers[k, 0]
            cy = centers[k, 1]
            A_plume = amps[k]
            w = widths[k]
            dx = x[i] - cx
            dy = y[i] - cy
            c_before += A_plume * math.exp(-0.5 * ((dx / w) ** 2 + (dy / w) ** 2))
        c_before += 0.03 * math.sin(0.2 * x[i])
        c_before += 0.02 * math.cos(0.15 * y[i])
        if c_before < 1e-8:
            c_before = 1e-8
        c_before_arr[i] = c_before

        log_c_before = math.log(c_before)
        A = logistic_activity(log_c_before, m[i])
        A_arr[i] = A

        # adaptation timescale modulation by gamma if used
        if use_gamma:
            gamma_i = gammas[i]
            tau_eff = tau_m / (1.0 + 0.01 * (gamma_i - 1.0))
            if tau_eff < 1.0:
                tau_eff = 1.0
        else:
            tau_eff = tau_m

        # methylation update
        methyl_dm = (A - A0) * (dt / tau_eff)
        m[i] += methyl_dm
        methyl_E = energy_per_methyl[i] * abs(methyl_dm)

        # movement decision by action
        act = actions[i]

        if act == 0:
            # tumble: single normal sample from caller; scale by kappa
            if kappa[i] > 0.0:
                dtheta = randn[i] / math.sqrt(kappa[i])
            else:
                dtheta = randn[i]
            theta[i] += dtheta
        else:
            # run: reuse same normal sample scaled appropriately
            dtheta = randn[i] * math.sqrt(2.0 * D_r[i] * dt)
            theta[i] += dtheta
            
            # movement scaled so that default dt ~ 0.05 matches original code scale
            x[i] += speed[i] * math.cos(theta[i]) * dt / 0.05
            y[i] += speed[i] * math.sin(theta[i]) * dt / 0.05

        # keep theta in [0, 2pi)
        two_pi = 2.0 * math.pi
        theta_i = theta[i]
        while theta_i >= two_pi:
            theta_i -= two_pi
        while theta_i < 0.0:
            theta_i += two_pi
        theta[i] = theta_i

        # run energy
        run_E = run_cost[i] * (1.0 if act == 1 else 0.0)

        # symmetric probe along heading
        fx_p = x[i] + probe_eps * math.cos(theta[i])
        fy_p = y[i] + probe_eps * math.sin(theta[i])
        fx_m = x[i] - probe_eps * math.cos(theta[i])
        fy_m = y[i] - probe_eps * math.sin(theta[i])

        c_plus = 0.0
        c_minus = 0.0
        for k in range(centers.shape[0]):
            cx = centers[k, 0]
            cy = centers[k, 1]
            A_plume = amps[k]
            w = widths[k]
            dxp = fx_p - cx
            dyp = fy_p - cy
            dxm = fx_m - cx
            dym = fy_m - cy
            c_plus += A_plume * math.exp(-0.5 * ((dxp / w) ** 2 + (dyp / w) ** 2))
            c_minus += A_plume * math.exp(-0.5 * ((dxm / w) ** 2 + (dym / w) ** 2))
        c_plus += 0.03 * math.sin(0.2 * fx_p)
        c_plus += 0.02 * math.cos(0.15 * fy_p)
        c_minus += 0.03 * math.sin(0.2 * fx_m)
        c_minus += 0.02 * math.cos(0.15 * fy_m)
        if c_plus < 1e-8:
            c_plus = 1e-8
        if c_minus < 1e-8:
            c_minus = 1e-8

        c_plus_arr[i] = c_plus
        c_minus_arr[i] = c_minus

        eps = probe_eps
        log_cp = math.log(c_plus)
        log_cm = math.log(c_minus)
        log_grad = (log_cp - log_cm) / (2.0 * eps)
        forward_alignment = (c_plus - c_before) / eps

        log_grad_arr[i] = log_grad
        forward_align_arr[i] = forward_alignment

        if grad_term_on:
            grad_term = reward_scale * log_grad
        else:
            grad_term = 0.0
        if forward_term_on:
            forward_term = 0.3 * forward_alignment
        else:
            forward_term = 0.0

        r_perf = grad_term + forward_term - time_penalty
        if r_perf > 5.0:
            r_perf = 5.0
        if r_perf < -5.0:
            r_perf = -5.0

        # energetic / entropy
        E_diss = run_E + methyl_E
        perf_reward[i] = r_perf
        cost[i] = E_diss
        entropy_approx[i] = E_diss / (kB_T + 1e-12)

        run_flags[i] = 1 if act == 1 else 0

    # interactions
    if interactions_on:
        # pairwise repulsion + interaction energy
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist == 0.0:
                    dx = (np.random.rand() - 0.5) * 1e-3
                    dy = (np.random.rand() - 0.5) * 1e-3
                    dist = math.sqrt(dx * dx + dy * dy)

                min_dist = interaction_radius * 0.5 * (size[i] + size[j])
                if dist < min_dist:
                    overlap = min_dist - dist
                    ux = dx / dist
                    uy = dy / dist
                    push = interaction_strength * overlap
                    x[i] -= 0.5 * push * ux
                    y[i] -= 0.5 * push * uy
                    x[j] += 0.5 * push * ux
                    y[j] += 0.5 * push * uy

                    extra_E = interaction_energy * overlap
                    cost[i] += extra_E
                    cost[j] += extra_E
                    entropy_approx[i] = cost[i] / (kB_T + 1e-12)
                    entropy_approx[j] = cost[j] / (kB_T + 1e-12)

        # cooperative bonus & crowding + density
        for i in range(N):
            neighbors_coop = 0
            neighbors_crowd = 0
            density = 0.0
            xi = x[i]
            yi = y[i]
            for j in range(N):
                if j == i:
                    continue
                dx = x[j] - xi
                dy = y[j] - yi
                d2 = dx * dx + dy * dy
                d = math.sqrt(d2)

                # coop neighbors
                if d < coop_radius:
                    neighbors_coop += 1

                # crowd neighbors
                if d < crowd_radius:
                    neighbors_crowd += 1

                # density kernel
                density += math.exp(-d2 / (2.0 * density_sigma * density_sigma))

            perf_reward[i] += coop_bonus * neighbors_coop

            if neighbors_crowd > crowd_threshold:
                excess = neighbors_crowd - crowd_threshold
                extra_cost = crowd_penalty * excess
                cost[i] += extra_cost

            extra_E_density = density_factor * density
            cost[i] += extra_E_density
            entropy_approx[i] = cost[i] / (kB_T + 1e-12)



# ============================================================
# Fast multi-agent environment (Python shell + Numba kernels)
# ============================================================

class MultiAgentBiophysEnv:
    """
    Aggressively JITted multi-agent biophysical environment.

    All per-step physics + interactions + costs + reward components are
    implemented in multiagent_step_kernel.

    This class is mostly a thin shell that maintains state arrays and
    handles reset + evolution counters.
    """

    def __init__(
        self,
        cf_centers,
        cf_amps,
        cf_widths,
        num_agents=8,
        seed=19,
        dt=0.05,
        tau_m=20.0,
        A0=0.5,
        probe_eps=0.8,
        reward_scale=6.0,
        time_penalty=0.001,
        kB_T=1.0,
        # morphology ranges
        size_range=(0.5, 1.5),
        shape_range=(0.5, 2.0),
        speed_range=(0.5, 1.2),
        # base morphology => motility energetics
        base_D_r=0.4,
        base_kappa=4.0,
        base_run_cost=0.01,
        base_energy_per_methyl=0.02,
        # interactions
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
        # evolution parameters
        evolution_on=True,
        evo_interval=10,
        evo_elite_frac=0.5,
        evo_mutation_std=(0.05, 0.1, 0.1),
        # ablation toggles
        interactions_on=True,
        grad_term_on=True,
        forward_term_on=True,
    ):
        self.num_agents = num_agents
        self.rng = np.random.RandomState(seed)

        self.dt = float(dt)
        self.tau_m = float(tau_m)
        self.A0 = float(A0)
        self.probe_eps = float(probe_eps)
        self.reward_scale = float(reward_scale)
        self.time_penalty = float(time_penalty)
        self.kB_T = float(kB_T)

        self.size_range = (float(size_range[0]), float(size_range[1]))
        self.shape_range = (float(shape_range[0]), float(shape_range[1]))
        self.speed_range = (float(speed_range[0]), float(speed_range[1]))

        self.base_D_r = float(base_D_r)
        self.base_kappa = float(base_kappa)
        self.base_run_cost = float(base_run_cost)
        self.base_energy_per_methyl = float(base_energy_per_methyl)

        self.cf_centers = cf_centers.astype(np.float64)
        self.cf_amps = cf_amps.astype(np.float64)
        self.cf_widths = cf_widths.astype(np.float64)

        # interaction params
        self.interactions_on = bool(interactions_on)
        self.interaction_radius = float(interaction_radius)
        self.interaction_strength = float(interaction_strength)
        self.interaction_energy = float(interaction_energy)

        self.coop_radius = float(coop_radius)
        self.coop_bonus = float(coop_bonus)
        self.crowd_radius = float(crowd_radius)
        self.crowd_penalty = float(crowd_penalty)
        self.crowd_threshold = int(crowd_threshold)
        self.density_sigma = float(density_sigma)
        self.density_factor = float(density_factor)

        # evolution (Strategy A: slow-timescale evolution)
        self.evolution_on = bool(evolution_on)
        # annealing control (multiplicative)

    

        # ablations for reward
        self.grad_term_on = bool(grad_term_on)
        self.forward_term_on = bool(forward_term_on)

        # allocate state arrays
        N = num_agents
        self.x = np.zeros(N, dtype=np.float64)
        self.y = np.zeros(N, dtype=np.float64)
        self.theta = np.zeros(N, dtype=np.float64)
        self.m = np.zeros(N, dtype=np.float64)

        self.size = np.zeros(N, dtype=np.float64)
        self.shape = np.zeros(N, dtype=np.float64)
        self.speed = np.zeros(N, dtype=np.float64)
        self.D_r = np.zeros(N, dtype=np.float64)
        self.kappa = np.zeros(N, dtype=np.float64)
        self.run_cost = np.zeros(N, dtype=np.float64)
        self.energy_per_methyl = np.zeros(N, dtype=np.float64)

        self._init_morphology()

        # working buffers for kernel outputs
        self.perf_reward = np.zeros(N, dtype=np.float64)
        self.cost = np.zeros(N, dtype=np.float64)
        self.entropy_approx = np.zeros(N, dtype=np.float64)
        self.A_arr = np.zeros(N, dtype=np.float64)
        self.c_before_arr = np.zeros(N, dtype=np.float64)
        self.c_plus_arr = np.zeros(N, dtype=np.float64)
        self.c_minus_arr = np.zeros(N, dtype=np.float64)
        self.log_grad_arr = np.zeros(N, dtype=np.float64)
        self.forward_align_arr = np.zeros(N, dtype=np.float64)
        self.run_flags = np.zeros(N, dtype=np.int64)



        # Strategy-A evolution (per-agent)
        self.evolution_on = bool(evolution_on)

        # renamed to match trainer override semantics
        self.evo_interval = int(evo_interval) if evo_interval > 0 else 10

        self.evo_elite_frac = float(evo_elite_frac)
        self.evo_mutation_std = tuple(evo_mutation_std)

        # mutation probability baseline
        self.evo_prob_per_event = 0.5

        # fitness tracking
        self.evo_scores = np.zeros(self.num_agents, dtype=np.float64)
        self.evo_score_ma = np.zeros(self.num_agents, dtype=np.float64)
        self.evo_ma_alpha = 0.2

        # timers synchronized with evo_interval
        self.evo_timer = np.random.randint(0, self.evo_interval,
                                   size=self.num_agents)


    def record_episode_fitness(self, total_perf, total_cost, beta=0.1):
        """
        Agent-wise evolution accumulation.

        total_perf, total_cost: arrays length N for one episode.
        We compute per-agent fitness = perf - beta * cost and:
          - update a moving-average of per-agent fitness
          - increment per-agent evo timer
          - when a timer exceeds evo_interval_per_agent, possibly mutate that agent
            (mutation can be triggered deterministically or probabilistically, and selection
             can favor replacing low-fitness agents with mutated offspring of better agents)
        """

        if (not self.evolution_on) or (self.evo_interval <= 0):
            return


        total_perf = np.asarray(total_perf, dtype=np.float64)
        total_cost = np.asarray(total_cost, dtype=np.float64)
        fitness = total_perf - beta * total_cost

        # update moving-average fitness (stabilizes selection), store raw scores too
        self.evo_score_ma = (1.0 - self.evo_ma_alpha) * self.evo_score_ma + self.evo_ma_alpha * fitness
        self.evo_scores += fitness  # still keep a cumulative for optional ranking

        # increment per-agent timer
        self.evo_timer += 1

        # check which agents are eligible to be considered for mutation
        eligible = np.where(self.evo_timer >= self.evo_interval)[0]
        if eligible.size == 0:
            return



        # decide which eligible agents will actually mutate:
        # strategy: low-fitness agents mutate with higher probability;
        #            high-fitness agents mutate with lower probability (exploratory)
        # compute normalized fitness (higher is better)
        ma = self.evo_score_ma
        # avoid degeneracy
        if np.all(np.isclose(ma, ma[0])):
            ma_norm = np.ones_like(ma)
        else:
            ma_min = ma.min()
            ma_max = ma.max()
            ma_norm = (ma - ma_min) / (ma_max - ma_min + 1e-12)

        # probability of mutation for eligible agents: higher if agent is lower fitness
        # p_mut = base_prob * (1 - ma_norm)  (so low-fitness -> 1-ma_norm ~1 -> higher p)
        base_prob = min(1.0, self.evo_prob_per_event)
        p_mut = base_prob * (1.0 - ma_norm[eligible])

        # sample mutations
        to_mutate_idx = []
        for idx_i, prob in zip(eligible, p_mut):
            if np.random.rand() < prob:
                to_mutate_idx.append(idx_i)

        # If none sampled but there are eligible agents, mutate at least one lowest-MA agent
        if len(to_mutate_idx) == 0 and eligible.size > 0:
            lowest = eligible[np.argmin(ma[eligible])]
            to_mutate_idx.append(int(lowest))

        # perform mutations:
        # mutation mechanism: small gaussian perturbations around current morphologies
        sigma_size, sigma_shape, sigma_speed = self.evo_mutation_std
        for i in to_mutate_idx:
            # choose whether to create a mutated child from a parent drawn from elites,
            # or to mutate the agent in-place. We do a hybrid: sample a parent from elites
            # with probability 0.5, otherwise mutate in-place
            if np.random.rand() < 0.5:
                # select a parent from top-elites by MA fitness
                elite_k = max(1, int(self.evo_elite_frac * self.num_agents))
                # argsort descending on moving-average
                sorted_idx = np.argsort(-self.evo_score_ma)
                elites = sorted_idx[:elite_k]
                parent = np.random.choice(elites)
                parent_size = self.size[parent]
                parent_shape = self.shape[parent]
                parent_speed = self.speed[parent]
                new_size = parent_size + sigma_size * np.random.randn()
                new_shape = parent_shape + sigma_shape * np.random.randn()
                new_speed = parent_speed + sigma_speed * np.random.randn()
            else:
                # mutate in-place
                new_size = self.size[i] + sigma_size * np.random.randn()
                new_shape = self.shape[i] + sigma_shape * np.random.randn()
                new_speed = self.speed[i] + sigma_speed * np.random.randn()

            # clamp into ranges (be conservative)
            new_size = float(max(self.size_range[0], min(self.size_range[1], new_size)))
            new_shape = float(max(self.shape_range[0], min(self.shape_range[1], new_shape)))
            new_speed = float(max(self.speed_range[0], min(self.speed_range[1], new_speed)))

            # apply mutation
            self.size[i] = new_size
            self.shape[i] = new_shape
            self.speed[i] = new_speed

            # recompute motility & energetics for this agent
            s = new_size
            sh = new_shape
            self.D_r[i] = self.base_D_r / (s * sh)
            self.kappa[i] = self.base_kappa * sh
            self.run_cost[i] = self.base_run_cost * s
            self.energy_per_methyl[i] = self.base_energy_per_methyl * s

            # reset per-agent timers to avoid immediate re-mutation
            self.evo_timer[i] = 0
            # optionally reset short-term MA so that parent advantages remain visible
            # self.evo_score_ma[i] = 0.0

        # clear cumulative score optionally to avoid runaway accumulation
        # self.evo_scores[:] = 0.0


    def _init_morphology(self):
        N = self.num_agents
        for i in range(N):
            s = self.rng.uniform(*self.size_range)
            sh = self.rng.uniform(*self.shape_range)
            sp = self.rng.uniform(*self.speed_range)
            self.size[i] = s
            self.shape[i] = sh
            self.speed[i] = sp

        self._update_motility_costs_from_morphology()

    def _update_motility_costs_from_morphology(self):
        N = self.num_agents
        for i in range(N):
            s = self.size[i]
            sh = self.shape[i]
            self.D_r[i] = self.base_D_r / (s * sh)
            self.kappa[i] = self.base_kappa * sh
            self.run_cost[i] = self.base_run_cost * s
            self.energy_per_methyl[i] = self.base_energy_per_methyl * s

    def reset(self):
        # randomize positions in a moderate region
        for i in range(self.num_agents):
            self.x[i] = self.rng.uniform(-5.0, 5.0)
            self.y[i] = self.rng.uniform(-5.0, 5.0)
            self.theta[i] = self.rng.rand() * 2.0 * math.pi
            self.m[i] = 0.0

        obs = self._build_obs()
        return obs

    def _build_obs(self):
        N = self.num_agents
        obs = np.zeros((N, 9), dtype=np.float32)
        for i in range(N):
            c = 0.0
            for k in range(self.cf_centers.shape[0]):
                cx = self.cf_centers[k, 0]
                cy = self.cf_centers[k, 1]
                A_plume = self.cf_amps[k]
                w = self.cf_widths[k]
                dx = self.x[i] - cx
                dy = self.y[i] - cy
                c += A_plume * math.exp(-0.5 * ((dx / w) ** 2 + (dy / w) ** 2))
            c += 0.03 * math.sin(0.2 * self.x[i])
            c += 0.02 * math.cos(0.15 * self.y[i])
            if c < 1e-8:
                c = 1e-8
            log_c = math.log(c)
            A = logistic_activity(log_c, self.m[i])

            obs[i, 0] = float(self.x[i])
            obs[i, 1] = float(self.y[i])
            obs[i, 2] = float(self.theta[i])
            obs[i, 3] = float(c)
            obs[i, 4] = float(A)
            obs[i, 5] = float(self.m[i])
            obs[i, 6] = float(self.size[i])
            obs[i, 7] = float(self.shape[i])
            obs[i, 8] = float(self.speed[i])
        return obs

    def step(self, actions, gammas=None):
        """
        actions: (N,) numpy int array {0,1}
        gammas:  (N,) numpy float array or None
        Returns:
            obs_next: (N,9)
            rewards: perf_reward (N,)
            dones:   all False vector
            info:    list of dicts (per-agent metrics)
        """
        N = self.num_agents
        if gammas is None:
            gammas_arr = np.zeros(N, dtype=np.float64)
            use_gamma = False
        else:
            gammas_arr = gammas.astype(np.float64)
            use_gamma = True

        actions_arr = actions.astype(np.int64)
        # sample one normal variate per agent for the JIT kernel (use RNG for reproducibility)
        randn_arr = self.rng.randn(N).astype(np.float64)



        multiagent_step_kernel(
            self.x, self.y, self.theta, self.m,
            self.size, self.shape, self.speed,
            self.D_r, self.kappa, self.run_cost, self.energy_per_methyl,
            self.dt, self.tau_m, self.A0,
            self.probe_eps,
            self.reward_scale, self.time_penalty,
            self.kB_T,
            self.cf_centers, self.cf_amps, self.cf_widths,
            self.interactions_on,
            self.interaction_radius, self.interaction_strength, self.interaction_energy,
            self.coop_radius, self.coop_bonus,
            self.crowd_radius, self.crowd_penalty, self.crowd_threshold,
            self.density_sigma, self.density_factor,
            self.grad_term_on, self.forward_term_on,
            actions_arr, gammas_arr, use_gamma,
            randn_arr,
            self.perf_reward, self.cost, self.entropy_approx,
            self.A_arr, self.c_before_arr, self.c_plus_arr, self.c_minus_arr,
            self.log_grad_arr, self.forward_align_arr,
            self.run_flags
        )


        obs_next = self._build_obs()
        dones = np.zeros(N, dtype=bool)

        infos = []
        for i in range(N):
            info = {
                "E_dissipated": float(self.cost[i]),
                "entropy_approx": float(self.entropy_approx[i]),
                "A": float(self.A_arr[i]),
                "c_before": float(self.c_before_arr[i]),
                "c_plus": float(self.c_plus_arr[i]),
                "c_minus": float(self.c_minus_arr[i]),
                "log_grad": float(self.log_grad_arr[i]),
                "forward_alignment": float(self.forward_align_arr[i]),
                "size": float(self.size[i]),
                "shape": float(self.shape[i]),
                "speed": float(self.speed[i]),
                "run_flag": int(self.run_flags[i]),
            }
            infos.append(info)

        return obs_next, self.perf_reward.copy().astype(np.float32), dones, infos




    def morphology_snapshot(self):
        return {
            "mean_size": float(np.mean(self.size)),
            "mean_shape": float(np.mean(self.shape)),
            "mean_speed": float(np.mean(self.speed)),
        }


# ============================================================
# Actor-Critic Network (PyTorch)
# ============================================================

class ActorCritic(nn.Module):
    def __init__(self, input_dim=9, hidden=512, output_gamma=False):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden, 1)
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


# ============================================================
# High-level trainer (PPO + GAE + Lagrangian)
# ============================================================

def train_with_full_metrics(
        outdir="output_biophys_metrics_numba",
        seed=19,
        episodes=800,
        episode_len=400,
        batch_episodes=16,
        lr=5e-4,
        hidden=512,
        entropy_coef=0.1,
        value_coef=0.5,
        max_grad_norm=0.5,
        train_gamma=False,
        ppo_epochs=4,
        mini_batch_size=512,
        clip_eps=0.3,
        gamma_discount=0.99,
        gae_lambda=0.95,
        value_clip=True,
        # Lagrangian
        cost_limit=25.0,
        lambda_lr=1e-3,
        init_lambda=0.0,
        save_stepwise=False,
        # Ablation toggles
        single_agent=False,
        no_interactions=False,
        no_evolution=False,
        no_cost_constraint=False,
        no_grad_term=False,
        no_forward_term=False,
        # Evolution control for Strategy A
        evo_interval_override=None,
        evo_mutation_anneal_decay=None
    ):
    os.makedirs(outdir, exist_ok=True)
    if save_stepwise:
        print("Warning: save_stepwise not supported in this accelerated version; ignoring.")
        save_stepwise = False

    torch.manual_seed(seed)
    np.random.seed(seed)

    # field
    cf, centers, amps, widths, meta = make_patchy_field(seed=seed, n_plumes=6)

    # ablation-derived settings
    num_agents = 1 if single_agent else 8
    interactions_on = not no_interactions
    evolution_on = not no_evolution
    grad_term_on = not no_grad_term
    forward_term_on = True
    use_cost_constraint = not no_cost_constraint

    env = MultiAgentBiophysEnv(
        centers, amps, widths,
        num_agents=num_agents,
        seed=seed,
        dt=0.05,
        tau_m=20.0,
        A0=0.5,
        probe_eps=0.8,
        reward_scale=6.0,
        time_penalty=0.001,
        kB_T=1.0,
        base_D_r=0.5,
        base_kappa=3.0,
        base_run_cost=0.01,
        base_energy_per_methyl=0.02,
        interaction_radius=1.0,
        interaction_strength=1.0 if interactions_on else 0.0,
        interaction_energy=0.02 if interactions_on else 0.0,
        coop_radius=2.0,
        coop_bonus=0.05 if interactions_on else 0.0,
        crowd_radius=3.0,
        crowd_penalty=0.1 if interactions_on else 0.0,
        crowd_threshold=2,
        density_sigma=2.5,
        density_factor=0.15 if interactions_on else 0.0,
        evolution_on=evolution_on,
        evo_interval=0 if no_evolution else 10,  # trainer will override below for Strategy A
        evo_elite_frac=0.5,
        evo_mutation_std=(0.02, 0.04, 0.04),
        interactions_on=interactions_on,
        grad_term_on=grad_term_on,
        forward_term_on=forward_term_on
    )

    # Strategy A: set conservative evo interval and optional anneal decay from trainer args
    # Recommended: set evo_interval_override to something large like 200

    if evo_interval_override is not None:
        env.evo_interval = int(evo_interval_override)

    # reinitialize timers using new interval
    env.evo_timer = np.random.randint(0, env.evo_interval,
                                  size=env.num_agents)


    if evo_mutation_anneal_decay is not None:
        env.evo_anneal_decay = float(evo_mutation_anneal_decay)
    else:
        env.evo_anneal_decay = 0.98

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = ActorCritic(input_dim=9, hidden=hidden, output_gamma=train_gamma).to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    metrics_records = []
    gamma_hist = []
    morph_records = []

    lambda_dual = init_lambda
    total_episodes = episodes
    ep_counter = 0
    update_idx = 0

    def run_episode(deterministic=False):
        N = env.num_agents

        all_obs = [[] for _ in range(N)]
        all_acts = [[] for _ in range(N)]
        all_rews = [[] for _ in range(N)]
        all_costs = [[] for _ in range(N)]
        all_infos = [[] for _ in range(N)]
        all_vals = [[] for _ in range(N)]
        all_logps = [[] for _ in range(N)]
        all_p = [[] for _ in range(N)]

        obs = env.reset()

        for t in range(episode_len):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            p_run, v, gamma_t = net(obs_tensor)

            if deterministic:
                acts = (p_run.detach().cpu().numpy() > 0.5).astype(np.int64)
            else:
                p_np = p_run.detach().cpu().numpy()
                acts = (np.random.rand(N) < p_np).astype(np.int64)

            v_np = v.detach().cpu().numpy()
            p_np = np.clip(p_run.detach().cpu().numpy(), 1e-8, 1 - 1e-8)

            p_safe = np.clip(p_np, 1e-8, 1 - 1e-8)
            logp = np.where(acts == 1, np.log(p_safe), np.log(1.0 - p_safe))


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

        total_perf = np.array([sum(all_rews[i]) for i in range(N)], dtype=float)
        total_cost = np.array([sum(all_costs[i]) for i in range(N)], dtype=float)
        env.record_episode_fitness(total_perf, total_cost, beta=0.1)

        trajectories = []
        for i in range(N):
            trajectories.append({
                "obs": np.array(all_obs[i], dtype=np.float32),
                "acts": np.array(all_acts[i], dtype=np.int64),
                "perf_rews": np.array(all_rews[i], dtype=np.float32),
                "costs": np.array(all_costs[i], dtype=np.float32),
                "info": all_infos[i],
                "vals": np.array(all_vals[i], dtype=np.float32),
                "logps": np.array(all_logps[i], dtype=np.float32),
                "p_runs": np.array(all_p[i], dtype=np.float32),
            })
        return trajectories

    while ep_counter < total_episodes:
        batch_trajs = []
        for _ in range(batch_episodes):
            traj_list = run_episode(deterministic=False)
            batch_trajs.extend(traj_list)
            ep_counter += 1
            if ep_counter >= total_episodes:
                break

        flat_obs = []
        flat_acts = []
        flat_returns = []
        flat_oldvals = []
        flat_oldlogp = []
        flat_costs = []
        flat_p_runs = []
        per_episode_metrics = []

        for traj in batch_trajs:
            rews = traj["perf_rews"]
            costs = traj["costs"]
            vals = traj["vals"]
            p_runs = traj["p_runs"]
            infos = traj["info"]

            T = rews.shape[0]
            if T == 0:
                continue

            r_comb = rews - lambda_dual * costs

            v_next = np.zeros_like(vals)
            v_next[:-1] = vals[1:]
            deltas = r_comb + gamma_discount * v_next - vals

            advs = np.zeros_like(deltas)
            gae = 0.0
            for t in range(T - 1, -1, -1):
                gae = deltas[t] + gamma_discount * gae_lambda * gae
                advs[t] = gae
            returns = advs + vals

            flat_obs.append(traj["obs"])
            flat_acts.append(traj["acts"])
            flat_returns.append(returns)
            flat_oldvals.append(vals)
            flat_oldlogp.append(traj["logps"])
            flat_costs.append(costs)
            flat_p_runs.append(p_runs)

            total_perf = float(rews.sum())
            total_cost = float(costs.sum())
            total_entropy = float(np.sum([inf.get("entropy_approx", 0.0) for inf in infos]))
            run_fraction = float(np.mean(traj["acts"] == 1))

            A_vals = np.array([inf.get("A", np.nan) for inf in infos], dtype=np.float64)
            c_vals = np.array([inf.get("c_before", np.nan) for inf in infos], dtype=np.float64)
            fisher_vals = np.array([fisher_activity_est_numba(a) for a in A_vals if not np.isnan(a)], dtype=np.float64)
            mean_fisher = float(np.nanmean(fisher_vals)) if fisher_vals.size > 0 else 0.0

            logc = np.log(np.clip(c_vals, 1e-12, None))
            var_A = float(np.nanvar(A_vals)) if A_vals.size > 0 else 0.0
            eps_A = 0.01
            mi_A_logc = gaussian_MI_est_numba(var_A, eps_A ** 2)

            c_next = np.empty_like(c_vals)
            if c_vals.size > 0:
                c_next[:-1] = c_vals[1:]
                c_next[-1] = c_vals[-1]
            pred_info = mutual_info_pearson_est_numba(A_vals.astype(np.float64), c_next.astype(np.float64))

            kl_vals = []
            for i in range(1, p_runs.shape[0]):
                kl_vals.append(kl_bernoulli_numba(p_runs[i - 1], p_runs[i]))
            mean_policy_kl = float(np.mean(kl_vals)) if len(kl_vals) > 0 else 0.0

            logprob_ratios = []
            kl_actionwise = []
            for i in range(p_runs.shape[0] - 1):
                p_curr = float(p_runs[i])
                p_next = float(p_runs[i + 1])
                a = traj["acts"][i]
                if a == 1:
                    logp_curr = math.log(p_curr + 1e-8)
                    logp_next = math.log(p_next + 1e-8)
                else:
                    logp_curr = math.log(1.0 - p_curr + 1e-8)
                    logp_next = math.log(1.0 - p_next + 1e-8)
                logprob_ratios.append(logp_curr - logp_next)
                kl_actionwise.append(kl_bernoulli_numba(p_curr, p_next))

            mean_logprob_ratio = float(np.mean(logprob_ratios)) if len(logprob_ratios) > 0 else 0.0
            mean_kl_actionwise = float(np.mean(kl_actionwise)) if len(kl_actionwise) > 0 else 0.0
            efficiency = total_perf / (total_cost + 1e-12)

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

        obs_tensor = torch.from_numpy(np.concatenate(flat_obs, axis=0)).to(device)
        acts_tensor = torch.from_numpy(np.concatenate(flat_acts, axis=0).astype(np.float32)).to(device)
        returns_tensor = torch.from_numpy(np.concatenate(flat_returns, axis=0).astype(np.float32)).to(device)
        old_values_tensor = torch.from_numpy(np.concatenate(flat_oldvals, axis=0).astype(np.float32)).to(device)
        old_logps_tensor = torch.from_numpy(np.concatenate(flat_oldlogp, axis=0).astype(np.float32)).to(device)
        flat_costs_arr = np.concatenate(flat_costs, axis=0).astype(np.float32)

        advantages = returns_tensor - old_values_tensor
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        

        N_data = obs_tensor.shape[0]
        idxs = np.arange(N_data)

        last_loss = 0.0
        for epoch in range(ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, N_data, mini_batch_size):
                mb_idx = idxs[start:start + mini_batch_size].tolist()
                mb_obs = obs_tensor[mb_idx]
                mb_acts = acts_tensor[mb_idx]
                mb_returns = returns_tensor[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_oldlogp = old_logps_tensor[mb_idx]
                mb_oldvals = old_values_tensor[mb_idx]

                mb_p, mb_v, mb_g = net(mb_obs)
                mb_p = mb_p.clamp(1e-8, 1.0 - 1e-8)
                new_logp = mb_acts * torch.log(mb_p) + (1.0 - mb_acts) * torch.log(1.0 - mb_p)
                ratio = torch.exp(new_logp - mb_oldlogp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                if value_clip:
                    v_clipped = mb_oldvals + torch.clamp(mb_v - mb_oldvals, -clip_eps, clip_eps)
                    v_loss_unclipped = (mb_v - mb_returns) ** 2
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    value_loss = 0.5 * (mb_v - mb_returns).pow(2).mean()

                ent = -(mb_p * torch.log(mb_p) + (1.0 - mb_p) * torch.log(1.0 - mb_p))
                entropy_loss = ent.mean()

                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
                optimizer.step()
                last_loss = float(loss.item())

        per_episode_costs = [m["total_cost"] for m in per_episode_metrics] if len(per_episode_metrics) > 0 else [0.0]
        avg_cost = float(np.mean(per_episode_costs))

        if use_cost_constraint:
            lambda_dual = max(0.0, lambda_dual + lambda_lr * (avg_cost - cost_limit))
        else:
            lambda_dual = 0.0

        batch_total_perf = float(sum(m["total_perf"] for m in per_episode_metrics))
        batch_total_cost = float(sum(m["total_cost"] for m in per_episode_metrics))
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
            "loss": last_loss,
            "single_agent": env.num_agents == 1,
            "interactions": env.interactions_on,
            "evolution": env.evolution_on,
            "cost_constraint": use_cost_constraint,
            "grad_term": env.grad_term_on,
            "forward_term": env.forward_term_on
        })

        gamma_hist.append({
            "episode": ep_counter,
            "avg_gamma": float("nan"),
            "lambda": lambda_dual,
            "single_agent": env.num_agents == 1,
            "interactions": env.interactions_on,
            "evolution": env.evolution_on,
            "cost_constraint": use_cost_constraint,
            "grad_term": env.grad_term_on,
            "forward_term": env.forward_term_on
        })

        morph = env.morphology_snapshot()
        morph_records.append({
            "update": update_idx,
            "episode_up_to": ep_counter,
            "mean_size": morph["mean_size"],
            "mean_shape": morph["mean_shape"],
            "mean_speed": morph["mean_speed"],
            "single_agent": env.num_agents == 1,
            "interactions": env.interactions_on,
            "evolution": env.evolution_on,
            "cost_constraint": use_cost_constraint,
            "grad_term": env.grad_term_on,
            "forward_term": env.forward_term_on
        })

        print(f"[{datetime.now().isoformat()}] up-to-episode {ep_counter} "
              f"avg_cost={avg_cost:.4f} lambda={lambda_dual:.4f} batch_eff={batch_eff:.4f} "
              f"(single_agent={env.num_agents==1}, interactions={env.interactions_on}, "
              f"evolution={env.evolution_on}, cost_constraint={use_cost_constraint}, "
              f"grad_term={env.grad_term_on}, forward_term={env.forward_term_on})")

    df_metrics = pd.DataFrame(metrics_records)
    df_gamma = pd.DataFrame(gamma_hist)
    df_morph = pd.DataFrame(morph_records)

    df_metrics.to_csv(Path(outdir) / "biophys_metrics.csv", index=False)
    df_gamma.to_csv(Path(outdir) / "gamma_history.csv", index=False)
    df_morph.to_csv(Path(outdir) / "morphology_history.csv", index=False)

    try:
        if len(df_metrics) > 0:
            plt.figure(figsize=(8, 4))
            plt.plot(df_metrics["episode_up_to"], df_metrics["batch_total_perf"])
            plt.xlabel("Episode")
            plt.ylabel("Batch total performance")
            plt.title("Batch total performance")
            plt.tight_layout()
            plt.savefig(Path(outdir) / "batch_total_perf.png", dpi=150)
            plt.close()
    except Exception:
        pass

    try:
        if len(df_morph) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(df_morph["episode_up_to"], df_morph["mean_size"], label="mean size")
            plt.plot(df_morph["episode_up_to"], df_morph["mean_shape"], label="mean shape")
            plt.plot(df_morph["episode_up_to"], df_morph["mean_speed"], label="mean speed")
            plt.legend()
            plt.xlabel("Episode")
            plt.ylabel("Morphology")
            plt.title("Morphology evolution")
            plt.tight_layout()
            plt.savefig(Path(outdir) / "morphology_evolution.png", dpi=150)
            plt.close()
    except Exception:
        pass

    print("Training + metrics collection complete. Artifacts in:", outdir)
    return df_metrics, df_gamma, df_morph


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="output_biophys_metrics_numba")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--episode_len", type=int, default=100)
    parser.add_argument("--batch_episodes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--ppo_epochs", type=int, default=6)
    parser.add_argument("--mini_batch_size", type=int, default=512)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--cost_limit", type=float, default=25.0)
    parser.add_argument("--lambda_lr", type=float, default=1e-3)
    parser.add_argument("--save_stepwise", action="store_true")
    parser.add_argument("--train_gamma", action="store_true")

    # Evolution control for Strategy A
    parser.add_argument("--evo_interval", type=int, default=200,
                        help="(Strategy A) conservative evolution interval (episodes).")
    parser.add_argument("--evo_anneal_decay", type=float, default=0.98,
                        help="(Strategy A) multiplicative anneal decay for mutation stds per generation.")

    # Ablation flags
    parser.add_argument("--single_agent", action="store_true",
                        help="Use a single agent instead of 8.")
    parser.add_argument("--no_interactions", action="store_true",
                        help="Disable physical + social interactions.")
    parser.add_argument("--no_evolution", action="store_true",
                        help="Disable morphology evolution.")
    parser.add_argument("--no_cost_constraint", action="store_true",
                        help="Disable cost constraint (lambda=0).")
    parser.add_argument("--no_grad_term", action="store_true",
                        help="Remove log-gradient term from reward.")
    parser.add_argument("--no_forward_term", action="store_true",
                        help="Remove forward-alignment term from reward.")

    args = parser.parse_args()

    train_with_full_metrics(
        outdir=args.outdir,
        seed=args.seed,
        episodes=args.episodes,
        episode_len=args.episode_len,
        batch_episodes=args.batch_episodes,
        lr=args.lr,
        hidden=args.hidden,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch_size,
        clip_eps=args.clip_eps,
        cost_limit=args.cost_limit,
        lambda_lr=args.lambda_lr,
        save_stepwise=args.save_stepwise,
        train_gamma=args.train_gamma,
        single_agent=args.single_agent,
        no_interactions=args.no_interactions,
        no_evolution=args.no_evolution,
        no_cost_constraint=args.no_cost_constraint,
        no_grad_term=args.no_grad_term,
        no_forward_term=args.no_forward_term,
        evo_interval_override=args.evo_interval,
        evo_mutation_anneal_decay=args.evo_anneal_decay
    )


if __name__ == "__main__":
    main()
