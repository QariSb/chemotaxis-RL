# Chemotaxis-Evo-PPO

Multi-agent, biophysical chemotaxis simulator with:

- PPO + GAE + Lagrangian cost constraint
- Explicit thermodynamic & information-theoretic metrics
- Multi-agent world with shared concentration field
- Agents with evolving morphology (size / shape / speed)
- Physical interactions (soft collisions, density-based dissipation)
- Cooperative & competitive rewards (reward sharing, crowding penalties)

## Features

- **Biophysical environment**
  - Patchy chemoattractant field (`make_patchy_field`)
  - Single-agent `BiophysEnv` with:
    - Receptor activity `A`, methylation `m`
    - Run/tumble dynamics
    - Energetic dissipation, entropy production

- **Multi-agent world**
  - `MultiAgentBiophysEnv` manages multiple `BiophysEnv` agents
  - Shared field, independent bodies
  - Soft repulsive interactions
  - Density-based extra dissipation

- **Interactions**
  - Cooperative reward sharing (neighbors within a given radius)
  - Competitive crowding penalty (too many neighbors → extra cost)
  - Density-based thermodynamic penalty

- **Evolution**
  - Morphology parameters: size, shape, speed
  - Fitness: `total_perf - beta * total_cost`
  - Periodic evolution:
    - Rank agents by fitness
    - Pick elites
    - Mutate size/shape/speed with Gaussian noise
    - Clamp to biologically reasonable ranges

- **RL algorithm**
  - PPO (clipped surrogate)
  - GAE(λ) advantages
  - Lagrangian multiplier on cost (entropy / energy dissipation)
  - Single shared policy for all agents

- **Logging & outputs**
  - `biophys_metrics.csv` – batch-level reward/cost/info-theoretic metrics
  - `gamma_history.csv` – dual variable λ history
  - `morphology_history.csv` – mean size/shape/speed over training
  - `batch_total_perf.png` – performance vs episode
  - `morphology_evolution.png` – morphology vs episode

## Installation

```bash
git clone https://github.com/<your-username>/chemotaxis-evo-ppo.git
cd chemotaxis-evo-ppo
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
