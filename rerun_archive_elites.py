import os
import sys
import numpy as np
import pandas as pd

import warnings
from numba.core.errors import NumbaTypeSafetyWarning

from concurrent.futures import ProcessPoolExecutor

# The user may need to change the following paths to represent their correct directory
REPO_ROOT = r"C:\Users\mrdxy\RL-Epidemic-Benchmark"
ARCHIVE_CSV = (
    r"C:\Users\mrdxy\Documents\MLDS\Final Project\Notebooks\saved_archives"
    r"\COVID_A_200iter_archive.csv"
)
OUTPUT_CSV = (
    r"C:\Users\mrdxy\Documents\MLDS\Final Project\Notebooks\saved_archives"
    r"\COVID_A_200iter_archive_corrected.csv"
)

# Here the scenario ID can be chosen
SCENARIO_ID = "jsons/COVID_A"
MAX_STEPS = 52
MAX_WORKERS = 4
H1 = 64
H2 = 64

SEED = 42
np.random.seed(SEED)

sys.path.append(REPO_ROOT)
os.chdir(REPO_ROOT)

warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)

from sac_kernel import make_primal_env

tmp_env = make_primal_env(SCENARIO_ID, vac_starts=0)()
obs_dim = tmp_env.observation_space.shape[0]
action_dim = tmp_env.action_space.shape[0]
del tmp_env


def decode_policy(solution: np.ndarray):
    """Decode flattened policy parameters into a 2-hidden-layer MLP parameters."""
    solution = np.asarray(solution, dtype=np.float32)
    idx = 0

    # Layer 1: obs_dim -> H1
    w1_size = obs_dim * H1
    b1_size = H1
    W1 = solution[idx:idx + w1_size].reshape(H1, obs_dim)
    idx += w1_size
    b1 = solution[idx:idx + b1_size]
    idx += b1_size

    # Layer 2: H1 -> H2
    w2_size = H1 * H2
    b2_size = H2
    W2 = solution[idx:idx + w2_size].reshape(H2, H1)
    idx += w2_size
    b2 = solution[idx:idx + b2_size]
    idx += b2_size

    # Output layer: H2 -> action_dim
    w3_size = H2 * action_dim
    b3_size = action_dim
    W3 = solution[idx:idx + w3_size].reshape(action_dim, H2)
    idx += w3_size
    b3 = solution[idx:idx + b3_size]
    idx += b3_size

    return W1, b1, W2, b2, W3, b3


def policy_action(solution: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Two-hidden-layer MLP with Tanh activations, output within [0,1]."""
    W1, b1, W2, b2, W3, b3 = decode_policy(solution)
    obs = np.asarray(obs, dtype=np.float32)

    h1 = np.tanh(W1 @ obs + b1)
    h2 = np.tanh(W2 @ h1 + b2)
    raw = W3 @ h2 + b3

    # Bounded action [0,1]
    u_norm = 0.5 * (np.tanh(raw) + 1.0)
    return u_norm.astype(np.float32)

def policy_action_decoded(params: tuple, obs: np.ndarray) -> np.ndarray:
    """Two-hidden-layer MLP with Tanh activations, output within to [0,1]. Using parameters as input."""
    W1, b1, W2, b2, W3, b3 = params
    obs = np.asarray(obs, dtype=np.float32)

    h1 = np.tanh(W1 @ obs + b1)
    h2 = np.tanh(W2 @ h1 + b2)
    raw = W3 @ h2 + b3

    u_norm = 0.5 * (np.tanh(raw) + 1.0)
    return u_norm.astype(np.float32)


def rerun_solution(solution, env):
    """Rerun one elite and extract exact internal simulator costs."""

    env.time_passed = 0
    obs = env.reset()

    total_reward = 0.0
    done = False
    steps = 0

    params = decode_policy(solution)

    while not done and steps < MAX_STEPS:
        action = policy_action_decoded(params, obs)
        obs, reward, done, _ = env.step(action)

        total_reward += float(reward)
        steps += 1

    cumulative_cost = env.epi.get_current_state().obs.cumulative_cost

    component_names = [
        intervention.name
        for intervention in env.epi.static.interventions
    ]

    costs = {
        component_names[i]: float(cumulative_cost[i].sum())
        for i in range(len(component_names))
    }

    if SCENARIO_ID.startswith("jsons/COVID"):
        health_cost = (
            costs.get("Infectious_cost", 0.0)
            + costs.get("Hospitalized_cost", 0.0)
            + costs.get("Death_cost", 0.0)
        )
    else:
        health_cost = costs.get("Infection_Cost", 0.0)

    intervention_cost = (
        costs.get("Vaccination", 0.0)
        + costs.get("Masks", 0.0)
    )

    return total_reward, health_cost, intervention_cost, costs

_rerun_env = None

def rerun_solution_parallel(solution):
    global _rerun_env

    if _rerun_env is None:
        _rerun_env = make_primal_env(SCENARIO_ID, vac_starts=0)()

    return rerun_solution(solution, _rerun_env)

def main():
    
    archive_df = pd.read_csv(ARCHIVE_CSV)
    
    solution_cols = sorted(
        [c for c in archive_df.columns if c.startswith("solution_")],
        key=lambda c: int(c.split("_")[1])
    )
    
    results = []
    
    solutions = [
        row[solution_cols].to_numpy(dtype=np.float32)
        for _, row in archive_df.iterrows()
    ]
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        replay_results = list(
            executor.map(
                rerun_solution_parallel,
                solutions,
                chunksize=4,
            )
        )
    
    results = []
    
    for number, (reward, health, intervention, components) in enumerate(
        replay_results,
        start=1,
    ):
        results.append({
            "rerun_reward": reward,
            "health_cost": health,
            "inter_cost": intervention,
        })
    
    # Replace old manual values with internal simulator values.
    cost_df = pd.DataFrame(results)
    archive_df = archive_df.drop(
        columns=["health_cost", "inter_cost"]
    ).join(cost_df)
    
    archive_df["internal_total_cost"] = (
        archive_df["health_cost"] + archive_df["inter_cost"]
    )
    
    # Sanity test
    archive_df["reward_cost_residual"] = (
        archive_df["rerun_reward"] + archive_df["internal_total_cost"]
    )
    
    archive_df.to_csv(OUTPUT_CSV, index=False)
    
    print("\nSaved:", OUTPUT_CSV)
    print(
        "Maximum absolute reward/cost residual:",
        archive_df["reward_cost_residual"].abs().max()
    )

if __name__ == "__main__":
    main()