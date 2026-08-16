import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
import gym
from ribs.archives import GridArchive
from ribs.emitters import GaussianEmitter
from ribs.emitters import GeneticAlgorithmEmitter
from ribs.schedulers import Scheduler

import warnings
from numba.core.errors import NumbaTypeSafetyWarning

from datetime import datetime

run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# The user may need to change the following path to point to their respective directory
folder_path = r"C:\Users\mrdxy\Documents\MLDS\Final Project\Notebooks\saved_archives"
os.makedirs(folder_path, exist_ok=True)

# Name relevant for the saved files
# This variable can be set to control the name of the saved files
name = "COVID_A_200iter"

csv_path = os.path.join(folder_path, f"{name}_archive.csv")
schedule_csv_path = os.path.join(folder_path, f"{name}_best_schedule.csv")
iteration_log_csv_path = os.path.join(folder_path, f"{name}_iteration_log.csv")
run_info_json_path = os.path.join(folder_path, f"{name}_run_info.json")

warnings.filterwarnings("ignore", category=NumbaTypeSafetyWarning)

# Root repository of installed package
REPO_ROOT = r"C:\Users\mrdxy\RL-Epidemic-Benchmark"
sys.path.append(REPO_ROOT)
os.chdir(REPO_ROOT)

# Scenario selection and cost constants

# Here the model used can be set
SCENARIO_ID = "jsons/COVID_A"
COVID_MODEL_FLAG = SCENARIO_ID.startswith("jsons/COVID")
SCENARIO_JSON = SCENARIO_ID + ".json"

SEED = 42
np.random.seed(SEED)

with open(os.path.join(REPO_ROOT, SCENARIO_JSON), "r") as f:
    cfg = json.load(f)

# Vaccination intervention
vacc = next(i for i in cfg["interventions"] if i["name"] == "Vaccination")
vacc_cp = {cp["name"]: float(cp["default_value"]) for cp in vacc["control_params"]}
MAX_CAPACITY = vacc_cp["max_capacity"]
VACC_PRICE_PER_DOSE = vacc_cp["price_per_dose"]

# Masks intervention
mask = next(i for i in cfg["interventions"] if i["name"] == "Masks")
mask_cp = {cp["name"]: float(cp["default_value"]) for cp in mask["control_params"]}
MASK_COST_PER_DAY = mask_cp["cost_per_day"]

# Health cost
# Different constants for SIR/SIRV and COVID model
if COVID_MODEL_FLAG:
    infectious_cost = next(c for c in cfg["costs"] if c["name"] == "Infectious_cost")
    infct_cp = {cp["name"]: float(cp["default_value"]) for cp in infectious_cost["control_params"]}
    INFCT_COST_PER_DAY = infct_cp["cost_per_day"]

    hospitalized_cost = next(c for c in cfg["costs"] if c["name"] == "Hospitalized_cost")
    hosp_cp = {cp["name"]: float(cp["default_value"]) for cp in hospitalized_cost["control_params"]}
    HOSP_COST_PER_DAY = hosp_cp["cost_per_day"]

    death_cost = next(c for c in cfg["costs"] if c["name"] == "Death_cost")
    death_cp = {cp["name"]: float(cp["default_value"]) for cp in death_cost["control_params"]}
    DEATH_COST_PER_DEATH = death_cp["cost_per_death"]
else:
    infection_cost = next(c for c in cfg["costs"] if c["name"] == "Infection_Cost")
    inf_cp = {cp["name"]: float(cp["default_value"]) for cp in infection_cost["control_params"]}
    INF_COST_PER_DAY = inf_cp["cost_per_day"]

# Import EpiEnv helper from the benchmark repoository
from sac_kernel import make_primal_env

# Global configuration
MAX_STEPS = 52          # 52 weeks horizon
NUM_ITERATIONS = 200    # QD iterations
ARCHIVE_SIZE = 20       # archive dims: ARCHIVE_SIZE x ARCHIVE_SIZE
NUM_EMITTERS = 2        # logical number of emitter pairs (broad + coarse)
EMITTER_BATCH_SIZE = 50 # solutions per emitter per iteration
MAX_WORKERS = 8         # parallel processes (adjust to CPU cores)

# Policy representation

# Temporary env to determine obs/action dimensions
_tmp_env = make_primal_env(SCENARIO_ID, vac_starts=0)()
obs_dim = _tmp_env.observation_space.shape[0]
action_dim = _tmp_env.action_space.shape[0]

# Network like in paper for PPO
H1 = 64
H2 = 64
solution_dim = (
    obs_dim * H1 + H1 +  # W1, b1
    H1 * H2 + H2 +  # W2, b2
    H2 * action_dim + action_dim   # W3, b3
)

archive_range_1 = (0.0, 1.0)
archive_range_2 = (0.0, 1.0)

TOTAL_POP = float(_tmp_env.epi.static.default_state.obs.current_comp.sum())

del _tmp_env


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


# Per-process environment

_env = None


def evaluate_solution_parallel(solution: np.ndarray, features: str = "avg_actions"):
    """
    Worker function: each process lazily constructs its own EpiEnv and reuses it.

    Returns:
        total_reward: float
        feats: np.ndarray of shape (2,) with behavior descriptors
        phase: int (epidemic phase based on final prevalence)
        health_cost: float (sum infection cost over episode)
        inter_cost: float (sum intervention cost over episode)
    """
    global _env
    if _env is None:
        _env = make_primal_env(SCENARIO_ID, vac_starts=0)()

    env = _env
    obs = env.reset()
    done = False

    total_reward = 0.0
    steps = 0
    avg_actions = np.zeros(action_dim, dtype=np.float32)

    health_cost = 0.0
    inter_cost = 0.0

    last_obs = obs

    if COVID_MODEL_FLAG:
        D_t_prev = 0.0

    params = decode_policy(solution)
    
    while not done and steps < MAX_STEPS:
        action = policy_action_decoded(params, obs)
        obs, reward, done, info = env.step(action)

        total_reward += float(reward)
        steps += 1
        avg_actions += action
        last_obs = obs

        # Intervention cost, masks + vaccination
        m = float(action[0])  # mask compliance in [0,1]
        v_deg = float(action[1])  # vaccination degree in [0,1]

        # Masks cost
        mask_step = MASK_COST_PER_DAY * m * TOTAL_POP

        # Vaccination cost
        doses = v_deg * MAX_CAPACITY
        vacc_step = doses * VACC_PRICE_PER_DOSE

        inter_cost += mask_step + vacc_step

        # Health cost, different for SIR/SIRV and COVID model
        if COVID_MODEL_FLAG:
            I_t = float(obs[2]) + float(obs[3]) + float(obs[4]) + float(obs[5]) + float(obs[6]) + float(obs[11]) + float(obs[12]) + float(obs[13])
            health_step = INFCT_COST_PER_DAY * I_t

            H_t = float(obs[5]) + float(obs[6]) + float(obs[13])
            health_step += HOSP_COST_PER_DAY * H_t

            D_t = float(obs[7])
            delta_D = max(0.0, D_t - D_t_prev)
            D_t_prev = D_t
            health_step += DEATH_COST_PER_DEATH * delta_D
        else:
            I_t = float(obs[1])
            health_step = INF_COST_PER_DAY * I_t
            
        health_cost += health_step

    if steps > 0:
        avg_actions /= steps

    if features == "avg_actions":
        f1 = float(avg_actions[0])  # avg mask
        f2 = float(avg_actions[1] if action_dim > 1 else 0.0)  # avg vacc
    elif features == "cost_dimensions":
        f1 = float(inter_cost)
        f2 = float(health_cost)
    else:
        f1 = float(avg_actions[0])
        f2 = float(avg_actions[1] if action_dim > 1 else 0.0)

    feats = np.array([f1, f2], dtype=np.float32)

    # Phase classification based on final infected population
    if COVID_MODEL_FLAG:
        I_final = float(last_obs[2]) + float(last_obs[3]) + float(last_obs[4]) + float(last_obs[5]) + float(last_obs[6]) + float(last_obs[11]) + float(last_obs[12]) + float(last_obs[13])
    else:
        I_final = float(last_obs[1])
    prevalence = I_final / TOTAL_POP

    if I_final <= 0.5:  # extinct
        phase = 0
    elif prevalence <= 0.0005:  # near eradication
        phase = 1
    elif prevalence <= 0.005:  # 0.05%–0.5%
        phase = 2
    else:  # > 0.5% infectious
        phase = 3

    return total_reward, feats, phase, health_cost, inter_cost


# QD setup        

def setup_archive_scheduler(archive_size, range_1, range_2):
    archive = GridArchive(
        solution_dim=solution_dim,
        dims=[archive_size, archive_size],
        ranges=[range_1, range_2],
        extra_fields={
            "phase": ((), np.int32),
            "health_cost": ((), np.float32),
            "inter_cost": ((), np.float32),
        },
    )

    emitters = []

    for _ in range(NUM_EMITTERS):
        # Broad exploratory emitter
        emitters.append(
            GaussianEmitter(
                archive,
                x0=np.random.randn(solution_dim),
                sigma=1.0,
                batch_size=EMITTER_BATCH_SIZE,
            )
        )

        # Narrow exploitative emitter
        emitters.append(
            GaussianEmitter(
                archive,
                x0=np.random.randn(solution_dim),
                sigma=0.05,
                batch_size=EMITTER_BATCH_SIZE,
            )
        )

    scheduler = Scheduler(archive, emitters)
    return archive, scheduler


def run_optimization_loop_parallel(archive, scheduler, num_iterations,
                                   features="avg_actions", max_workers=4):
    all_rewards = []
    all_features = []
    iteration_log = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        eval_fn = partial(evaluate_solution_parallel, features=features)

        for itr in range(num_iterations):
            solutions = scheduler.ask()
            results = list(ex.map(eval_fn, solutions, chunksize=4))

            n = len(results)
            qualities = np.empty(n)
            features_batch = np.empty((n, 2), dtype=np.float32)
            phase_batch = np.empty(n, dtype=np.int32)
            health_batch = np.empty(n, dtype=np.float32)
            inter_batch = np.empty(n, dtype=np.float32)

            for i, (q, feats, phase, hc, ic) in enumerate(results):
                qualities[i] = q
                features_batch[i] = feats
                phase_batch[i] = phase
                health_batch[i] = hc
                inter_batch[i] = ic

            all_rewards.extend(list(qualities))
            all_features.extend(list(features_batch))

            scheduler.tell(
                qualities,
                features_batch,
                phase=phase_batch,
                health_cost=health_batch,
                inter_cost=inter_batch,
            )

            archive_df_iter = archive.data(return_type="pandas")

            iteration_log.append({
                "iteration": itr + 1,
                "archive_size": int(archive.stats.num_elites),
                "best_objective_archive": float(archive_df_iter["objective"].max()) if len(archive_df_iter) > 0 else np.nan,
                "mean_objective_archive": float(archive_df_iter["objective"].mean()) if len(archive_df_iter) > 0 else np.nan,
                "best_objective_batch": float(np.max(qualities)),
                "mean_objective_batch": float(np.mean(qualities)),
            })

            if (itr + 1) % 10 == 0:
                print(
                    f"Iteration {itr+1}/{num_iterations}, "
                    f"archive size = {archive.stats.num_elites}"
                )

    print("Final archive size:", archive.stats.num_elites)
    return all_features, all_rewards, iteration_log


# Seeding to fill archive

def make_constant_solution(a0_star: float, a1_star: float) -> np.ndarray:
    """Return a solution whose MLP policy always outputs [a0_star, a1_star] in [0,1]^2."""
    u_vec = np.array([a0_star, a1_star], dtype=np.float32)

    # Invert u = 0.5 * (tanh(raw) + 1)
    tanh_arg = 2.0 * u_vec - 1.0
    tanh_arg = np.clip(tanh_arg, -0.999999, 0.999999)
    b3 = np.arctanh(tanh_arg).astype(np.float32)   # output-layer bias

    # Zero all weights and hidden biases so the network output is constant.
    W1 = np.zeros((H1, obs_dim), dtype=np.float32)
    b1 = np.zeros(H1, dtype=np.float32)

    W2 = np.zeros((H2, H1), dtype=np.float32)
    b2 = np.zeros(H2, dtype=np.float32)

    W3 = np.zeros((action_dim, H2), dtype=np.float32)

    sol = np.concatenate([
        W1.flatten(), b1,
        W2.flatten(), b2,
        W3.flatten(), b3,
    ])
    return sol


def seed_archive_grid(archive, archive_range_1, archive_range_2):
    """Light seeding: one constant policy per archive cell center."""
    seed_solutions = []
    seed_qualities = []
    seed_features = []
    seed_phases = []
    seed_health = []
    seed_inter = []

    n0, n1 = archive.dims
    r0_min, r0_max = archive_range_1
    r1_min, r1_max = archive_range_2

    # Seeding loop
    for i in range(n0):
        for j in range(n1):
            a0_star = r0_min + (i + 0.5) * (r0_max - r0_min) / n0
            a1_star = r1_min + (j + 0.5) * (r1_max - r1_min) / n1

            sol = make_constant_solution(a0_star, a1_star)
            q, feats, ph, hc, ic = evaluate_solution_parallel(sol)

            seed_solutions.append(sol)
            seed_qualities.append(q)
            seed_features.append(feats)
            seed_phases.append(ph)
            seed_health.append(hc)
            seed_inter.append(ic)

    seed_solutions = np.stack(seed_solutions, axis=0)
    seed_qualities = np.array(seed_qualities, dtype=float)
    seed_features = np.array(seed_features, dtype=np.float32)
    seed_phases = np.array(seed_phases, dtype=np.int32)
    seed_health = np.array(seed_health, dtype=np.float32)
    seed_inter = np.array(seed_inter, dtype=np.float32)

    archive.add(
        seed_solutions,
        seed_qualities,
        seed_features,
        phase=seed_phases,
        health_cost=seed_health,
        inter_cost=seed_inter,
    )
    print("Seeded archive size:", archive.stats.num_elites)


# Run best elite and save intervention schedule

def rollout_intervention_schedule(solution, scenario_id=SCENARIO_ID, max_steps=MAX_STEPS):
    """Run one episode with the input solution and record intervention parameters over time."""
    env = make_primal_env(scenario_id, vac_starts=0)()
    obs = env.reset()
    done = False
    actions_over_time = []
    rewards_over_time = []
    states_over_time = [np.asarray(obs, dtype=np.float32).copy()]
    t = 0
    
    while not done and t < max_steps:
        action = policy_action(solution, obs)
        actions_over_time.append(np.asarray(action, dtype=np.float32).copy())
        obs, reward, done, info = env.step(action)
        rewards_over_time.append(float(reward))
        states_over_time.append(np.asarray(obs, dtype=np.float32).copy())
        t += 1
        
    actions_over_time = np.stack(actions_over_time, axis=0)  # (T, action_dim)
    rewards_over_time = np.asarray(rewards_over_time, dtype=np.float32)
    states_over_time = np.stack(states_over_time, axis=0)  # (T+1, obs_dim)
    
    return actions_over_time, rewards_over_time, states_over_time

def save_run_info(path):
    """Save run information in a meta file"""
    if COVID_MODEL_FLAG:
        cost_constants = {
            "max_capacity": float(MAX_CAPACITY),
            "vacc_price_per_dose": float(VACC_PRICE_PER_DOSE),
            "mask_cost_per_day": float(MASK_COST_PER_DAY),
            "infectious_cost_per_day": float(INFCT_COST_PER_DAY),
            "hospitalized_cost_per_day": float(HOSP_COST_PER_DAY),
            "death_cost_per_death": float(DEATH_COST_PER_DEATH),
        }
    else:
        cost_constants = {
            "max_capacity": float(MAX_CAPACITY),
            "vacc_price_per_dose": float(VACC_PRICE_PER_DOSE),
            "mask_cost_per_day": float(MASK_COST_PER_DAY),
            "infection_cost_per_day": float(INF_COST_PER_DAY),
        }
    
    run_info = {
        "name": name,
        "timestamp": run_timestamp,
        "scenario_id": SCENARIO_ID,
        "scenario_json": SCENARIO_JSON,
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "num_iterations": NUM_ITERATIONS,
        "archive_size": ARCHIVE_SIZE,
        "num_emitters": NUM_EMITTERS,
        "emitter_batch_size": EMITTER_BATCH_SIZE,
        "max_workers": MAX_WORKERS,
        "obs_dim": int(obs_dim),
        "action_dim": int(action_dim),
        "solution_dim": int(solution_dim),
        "total_population": float(TOTAL_POP),
        "feature_mode": "avg_actions",
        "archive_range_1": archive_range_1,
        "archive_range_2": archive_range_2,
        "cost_constants": cost_constants,
    }

    with open(path, "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Saved run info to: {path}")

# Main

if __name__ == "__main__":
    print(f"obs_dim = {obs_dim}, action_dim = {action_dim}, solution_dim = {solution_dim}")

    feature_name_1 = "avg intervention dim 0"
    feature_name_2 = "avg intervention dim 1"

    archive, scheduler = setup_archive_scheduler(ARCHIVE_SIZE, archive_range_1, archive_range_2)

    seed_archive_grid(archive, archive_range_1, archive_range_2)

    all_features, all_rewards, iteration_log = run_optimization_loop_parallel(
        archive, scheduler, NUM_ITERATIONS, features="avg_actions", max_workers=MAX_WORKERS
    )

    archive_df = archive.data(return_type="pandas")

    iteration_log_df = pd.DataFrame(iteration_log)
    iteration_log_df.to_csv(iteration_log_csv_path, index=False)

    archive_df.to_csv(csv_path, index=False)

    save_run_info(run_info_json_path)

    # Get best elite from archive
    sol_cols = [c for c in archive_df.columns if c.startswith("solution_")]
    best_idx = archive_df["objective"].idxmax()
    best_row = archive_df.loc[best_idx]
    best_solution = best_row[sol_cols].to_numpy(dtype=np.float32)

    print(f"Best elite objective: {best_row['objective']:.6f}")
    print(f"Best elite phase: {best_row['phase']}")

    # Roll out and calculate solution for best elite
    actions_over_time, rewards_over_time, states_over_time = rollout_intervention_schedule(best_solution)

    # Build schedule dataframe
    T = actions_over_time.shape[0]
    schedule_df = pd.DataFrame({
        "t": np.arange(T, dtype=int),
        "reward": rewards_over_time,
        "mask": actions_over_time[:, 0],
        "vaccination": actions_over_time[:, 1],
    })

    # State columns for SIR and SIRV, other interpretation for COVID model
    if states_over_time.shape[1] >= 3:
        schedule_df["S"] = states_over_time[:-1, 0]
        schedule_df["I"] = states_over_time[:-1, 1]
        schedule_df["R"] = states_over_time[:-1, 2]

        schedule_df["S_next"] = states_over_time[1:, 0]
        schedule_df["I_next"] = states_over_time[1:, 1]
        schedule_df["R_next"] = states_over_time[1:, 2]

    schedule_df.to_csv(schedule_csv_path, index=False)

    print(f"Saved archive to: {csv_path}")
    print(f"Saved iteration log to: {iteration_log_csv_path}")
    print(f"Saved best schedule to: {schedule_csv_path}")