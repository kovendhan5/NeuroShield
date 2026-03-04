"""Synthetic CI/CD pipeline simulator for PPO training.

Generates 52-dimensional state vectors and simulates action outcomes
matching the research paper specification (Table 1).

State vector layout (52D total):
  [ 0:10]  Build Metrics          (10D)
  [10:22]  Resource Metrics        (12D)
  [22:38]  Log Embeddings          (16D)  PCA-reduced DistilBERT
  [38:52]  Dependency Signals      (14D)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np

# --- Observation dimensions ---
OBS_DIM = 52
BUILD_METRICS_DIM = 10       # indices  0 ..  9
RESOURCE_METRICS_DIM = 12    # indices 10 .. 21
LOG_EMBEDDING_DIM = 16       # indices 22 .. 37
DEPENDENCY_SIGNALS_DIM = 14  # indices 38 .. 51

# --- Failure catalogue ---
FAILURE_TYPES: List[str] = [
    "OOM",
    "FlakyTest",
    "DependencyConflict",
    "NetworkLatency",
    "Healthy",
]

# --- Baseline & optimal MTTR (minutes) from paper Table 1 ---
BASELINE_MTTR_MINUTES: Dict[str, float] = {
    "OOM": 14.2,
    "FlakyTest": 8.5,
    "DependencyConflict": 15.1,
    "NetworkLatency": 11.3,
    "Healthy": 0.5,
}

OPTIMAL_MTTR_MINUTES: Dict[str, float] = {
    "OOM": 7.5,
    "FlakyTest": 4.3,
    "DependencyConflict": 9.8,
    "NetworkLatency": 6.2,
    "Healthy": 0.5,
}

# --- Action space (6 discrete actions) ---
# 0: retry_stage          – re-run the failed stage
# 1: clean_and_rerun      – wipe caches, rebuild
# 2: regenerate_config    – regenerate dependency/lock files
# 3: reallocate_resources – scale pods / increase limits
# 4: trigger_safe_rollback – roll back to last known-good
# 5: escalate_to_human    – no automated action (defer)
NUM_ACTIONS = 6

OPTIMAL_ACTION: Dict[str, int] = {
    "OOM": 3,                # reallocate_resources
    "FlakyTest": 0,          # retry_stage
    "DependencyConflict": 2, # regenerate_config
    "NetworkLatency": 4,     # trigger_safe_rollback
    "Healthy": 5,            # escalate_to_human (no-op)
}

# Resource cost per action (0.0 = free, 1.0 = expensive)
ACTION_RESOURCE_COST: Dict[int, float] = {
    0: 0.1,   # retry_stage          – cheap
    1: 0.4,   # clean_and_rerun      – moderate
    2: 0.3,   # regenerate_config    – moderate
    3: 0.6,   # reallocate_resources – expensive
    4: 0.5,   # trigger_safe_rollback – moderate-high
    5: 0.0,   # escalate_to_human    – free
}


@dataclass
class SimulationResult:
    """Results of a simulated pipeline action."""

    success: bool
    mttr: float
    cost: float
    resource_efficiency: float
    false_positive: float
    state: np.ndarray


# ---------------------------------------------------------------------------
# State sampling helpers – realistic value ranges per dimension group
# ---------------------------------------------------------------------------

def _sample_build_metrics(rng: random.Random, failure_type: str) -> List[float]:
    """Generate 10D build metrics with realistic ranges.

    Dimensions:
      0  build_duration       (seconds, 30-600)
      1  passed_tests         (count, 0-500)
      2  failed_tests         (count, 0-50)
      3  retry_count          (count, 0-5)
      4  stage_failure_rate   (ratio, 0.0-1.0)
      5  build_number         (count, 1-9999)
      6  queue_time           (seconds, 0-120)
      7  artifact_size        (MB, 1-500)
      8  test_coverage        (ratio, 0.0-1.0)
      9  change_set_size      (count, 0-200)
    """
    is_failing = failure_type != "Healthy"
    build_duration = rng.uniform(30, 600)
    total_tests = rng.randint(50, 500)
    failed_tests = rng.randint(1, 50) if is_failing else 0
    passed_tests = total_tests - failed_tests
    retry_count = rng.randint(1, 5) if is_failing else 0
    stage_failure_rate = rng.uniform(0.1, 0.8) if is_failing else rng.uniform(0.0, 0.05)
    build_number = rng.randint(1, 9999)
    queue_time = rng.uniform(0, 120)
    artifact_size = rng.uniform(1, 500)
    test_coverage = rng.uniform(0.4, 0.95)
    change_set_size = rng.randint(0, 200)
    return [
        build_duration, float(passed_tests), float(failed_tests),
        float(retry_count), stage_failure_rate, float(build_number),
        queue_time, artifact_size, test_coverage, float(change_set_size),
    ]


def _sample_resource_metrics(rng: random.Random, failure_type: str) -> List[float]:
    """Generate 12D resource metrics with realistic ranges.

    Dimensions:
      0  cpu_avg_5m          (ratio, 0.0-1.0)
      1  memory_avg_5m       (ratio, 0.0-1.0)
      2  memory_max          (ratio, 0.0-1.0)
      3  pod_restarts        (count, 0-20)
      4  throttle_events     (count, 0-100)
      5  network_latency     (ms, 0-500)
      6  disk_io             (MB/s, 0-200)
      7  cpu_limit_pct       (ratio, 0.0-1.0)
      8  memory_limit_pct    (ratio, 0.0-1.0)
      9  node_count          (count, 1-20)
     10  pending_pods        (count, 0-10)
     11  evicted_pods        (count, 0-5)
    """
    stressed = failure_type in ("OOM", "NetworkLatency")
    cpu_avg = rng.uniform(0.6, 0.98) if stressed else rng.uniform(0.1, 0.6)
    mem_avg = rng.uniform(0.7, 0.99) if failure_type == "OOM" else rng.uniform(0.2, 0.7)
    mem_max = min(1.0, mem_avg + rng.uniform(0.0, 0.15))
    pod_restarts = rng.randint(2, 20) if failure_type == "OOM" else rng.randint(0, 3)
    throttle_events = rng.randint(10, 100) if stressed else rng.randint(0, 10)
    net_latency = rng.uniform(100, 500) if failure_type == "NetworkLatency" else rng.uniform(1, 50)
    disk_io = rng.uniform(0, 200)
    cpu_limit_pct = rng.uniform(0.5, 1.0) if stressed else rng.uniform(0.1, 0.5)
    mem_limit_pct = rng.uniform(0.6, 1.0) if failure_type == "OOM" else rng.uniform(0.1, 0.6)
    node_count = rng.randint(1, 20)
    pending_pods = rng.randint(1, 10) if stressed else rng.randint(0, 2)
    evicted_pods = rng.randint(1, 5) if failure_type == "OOM" else 0
    return [
        cpu_avg, mem_avg, mem_max, float(pod_restarts), float(throttle_events),
        net_latency, disk_io, cpu_limit_pct, mem_limit_pct,
        float(node_count), float(pending_pods), float(evicted_pods),
    ]


def _sample_log_embeddings(rng: random.Random) -> List[float]:
    """Generate 16D PCA-reduced DistilBERT log embeddings.

    Values are normally distributed, mimicking PCA projection output.
    """
    return [rng.normalvariate(0.0, 1.0) for _ in range(LOG_EMBEDDING_DIM)]


def _sample_dependency_signals(rng: random.Random, failure_type: str) -> List[float]:
    """Generate 14D dependency signals with realistic ranges.

    Dimensions:
       0  dep_version_drifts    (count, 0-20)
       1  cache_hit_ratio       (ratio, 0.0-1.0)
       2  cache_miss_ratio      (ratio, 0.0-1.0)
       3  new_deps_count        (count, 0-10)
       4  outdated_deps         (count, 0-30)
       5  pkg_manager_npm       (binary, 0 or 1)
       6  pkg_manager_maven     (binary, 0 or 1)
       7  pkg_manager_pip       (binary, 0 or 1)
       8  dep_resolution_time   (seconds, 0-120)
       9  lock_file_changed     (binary, 0 or 1)
      10  transitive_dep_count  (count, 0-500)
      11  dep_conflict_count    (count, 0-10)
      12  registry_latency      (ms, 0-2000)
      13  dep_download_failures (count, 0-10)
    """
    dep_issue = failure_type == "DependencyConflict"
    drift = rng.randint(3, 20) if dep_issue else rng.randint(0, 5)
    cache_hit = rng.uniform(0.3, 0.6) if dep_issue else rng.uniform(0.7, 0.99)
    cache_miss = 1.0 - cache_hit
    new_deps = rng.randint(2, 10) if dep_issue else rng.randint(0, 3)
    outdated = rng.randint(5, 30) if dep_issue else rng.randint(0, 5)
    # one-hot-ish package manager flags
    pkg_npm = float(rng.random() < 0.33)
    pkg_maven = float(rng.random() < 0.33)
    pkg_pip = float(rng.random() < 0.33)
    dep_res_time = rng.uniform(30, 120) if dep_issue else rng.uniform(1, 30)
    lock_changed = float(rng.random() < 0.7) if dep_issue else float(rng.random() < 0.1)
    transitive = rng.randint(50, 500)
    conflicts = rng.randint(2, 10) if dep_issue else rng.randint(0, 1)
    reg_latency = rng.uniform(200, 2000) if dep_issue else rng.uniform(10, 200)
    dl_failures = rng.randint(1, 10) if dep_issue else 0
    return [
        float(drift), cache_hit, cache_miss, float(new_deps), float(outdated),
        pkg_npm, pkg_maven, pkg_pip, dep_res_time, lock_changed,
        float(transitive), float(conflicts), reg_latency, float(dl_failures),
    ]


def sample_state(rng: random.Random, failure_type: str = "Healthy") -> np.ndarray:
    """Generate a synthetic 52D state vector with realistic value ranges.

    Args:
        rng: Random generator.
        failure_type: Current failure category (affects metric distributions).

    Returns:
        52D float32 vector.
    """
    vec: List[float] = []
    vec.extend(_sample_build_metrics(rng, failure_type))       # 10D  [ 0:10]
    vec.extend(_sample_resource_metrics(rng, failure_type))     # 12D  [10:22]
    vec.extend(_sample_log_embeddings(rng))                     # 16D  [22:38]
    vec.extend(_sample_dependency_signals(rng, failure_type))   # 14D  [38:52]
    assert len(vec) == OBS_DIM, f"Expected {OBS_DIM}D, got {len(vec)}D"
    return np.array(vec, dtype=np.float32)


def simulate_mttr(failure_type: str, action: int) -> float:
    """Return MTTR (minutes) based on optimal action table.

    Args:
        failure_type: Failure category.
        action: Discrete action id (0-5).

    Returns:
        MTTR in minutes. Optimal action yields optimal MTTR; otherwise baseline.
    """
    baseline = BASELINE_MTTR_MINUTES.get(failure_type, 12.4)
    optimal = OPTIMAL_MTTR_MINUTES.get(failure_type, baseline)
    return optimal if action == OPTIMAL_ACTION.get(failure_type, 5) else baseline


def simulate_action(failure_type: str, action: int, rng: random.Random) -> SimulationResult:
    """Simulate action outcome for a given failure type.

    Deterministic MTTR values based on paper Table 1.
    Reward components:
      - resource_efficiency: 1.0 - action_resource_cost
      - false_positive:      1.0 if acting on a Healthy pipeline, else 0.0

    Args:
        failure_type: Failure category or Healthy.
        action: Discrete action id (0-5).
        rng: Random generator.

    Returns:
        SimulationResult with success flag, MTTR, cost, reward components, and state.
    """
    mttr = simulate_mttr(failure_type, action)
    baseline = BASELINE_MTTR_MINUTES.get(failure_type, 12.4)
    optimal = OPTIMAL_MTTR_MINUTES.get(failure_type, baseline)
    success = mttr <= (optimal + 1.0)

    action_cost = ACTION_RESOURCE_COST.get(action, 0.0)
    resource_efficiency = 1.0 - action_cost

    # False positive: taking a non-trivial action when the pipeline is healthy
    false_positive = 1.0 if (failure_type == "Healthy" and action != 5) else 0.0

    state = sample_state(rng, failure_type)

    return SimulationResult(
        success=success,
        mttr=mttr,
        cost=action_cost,
        resource_efficiency=resource_efficiency,
        false_positive=false_positive,
        state=state,
    )
