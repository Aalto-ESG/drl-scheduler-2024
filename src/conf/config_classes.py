import os
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.hydra_config import HydraConfig

""" 
Defines variable names and types to make autocomplete work with Hydra.
"""

@dataclass
class Paths:
    models_path: str = "models/"
    experiment_csv_path: str = ""
    tb_logs_path: str = "logs/tb_logs/"

@dataclass
class Simulator:
    dag: str = "C"  # Default DAG used in experiments
    max_batch: int = 1
    reward_scheme: int = 7
    history_length: int = 30  # Affects history state and rolling average reward
    state_scheme: int = 0
    state_representation: int = 0
    max_latency: float = 5000
    cache_hit_coefficient: float = 1
    streaming_tasks: int = 100
    batch_tasks: int = 2000
    eval_num_tasks: Optional[int] = 2000
    interval: int = 50
    reward_factor: float = 1
    min_idling_time: int = 10  # If no action, simulator idles for this duration at minimum
    invalid_action_penalty: float = 0  # Penalty on invalid actions (NOTE: use negative values)
    step_penalty: float = 0  # Penalty on each step to discourage idling (NOTE: use negative values)
    enable_idle_action: bool = True
    allow_idling: bool = True  # If disabled, simulator replaces illegal actions with valid actions
    tasks_can_expire: bool = True  # NOTE: If this is disabled and idling is allowed -> livelock possible
    done_on_expire: bool = True  # Returns done (finished simulation) when a task expires
    enable_reward_clipping: bool = False  # Only has effect if task expiration is disabled (clip to [-1, 1])
    use_deterministic_dag: bool = False  # Used to make evaluation deterministic - should be false during training

@dataclass
class TaskPoolState:
    num_tasks_boolean: bool = True
    one_hot_cache: bool = True
    num_tasks: bool = True
    avg_latencies: bool = True
    latencies: bool = True
    latencies_length: int = 3  # Take latency of N top tasks

@dataclass
class HistoryState:
    num_tasks: bool = False
    include_p3: bool = False
    approximate_p3: bool = False  # If true, use approximation of p3 instead of actual value in observation vector


@dataclass
class Training:
    ppo: bool = False  # Use PPO instead of DQN
    enabled: bool = True  # False disables training, in case you only want to run baseline policies
    threads: int = 1  # Number of parallel threads running the simulation (1 == no multithreading)
    training_steps: int = 100000
    learning_starts: int = 100
    batch_size: int = 32
    layer_size: int = 64
    num_layers: int = 2
    policy_kwargs: dict = field(default_factory=lambda: dict(net_arch=[32, 32, 32]))  # Maybe unused?


@dataclass
class Evaluation:
    iterations: int = 3  # Use lower numbers for quick testing - Use higher numbers for more accurate results


@dataclass
class PlotConfig:
    """ Perhaps make plots configurable from terminal? """
    title = None


@dataclass
class VisConfig:
    plot: bool = False


@dataclass
class SchedulerConfig:
    smoke_test: bool = False
    paths: Paths = Paths()
    sim: Simulator = Simulator()
    training: Training = Training()
    plots: PlotConfig = PlotConfig()
    vis: VisConfig = VisConfig()
    tpstate: TaskPoolState = TaskPoolState()
    hstate: HistoryState = HistoryState()
    eval: Evaluation = Evaluation()
    hydra_path: str = ""
    hyper_params_dict:  Optional[dict] = None


