import copy
import random
from typing import Dict, Optional

import numpy as np
# import ray
# from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
# from ray.rllib.algorithms.pg import PGConfig
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes
# from ray.rllib.examples.env.simple_corridor import SimpleCorridor
# from ray.rllib.utils.test_utils import check_learning_achieved

from src.experiments.dags import get_custom_env2
from src.experiments.evaluation import plot_baselines, eval_and_plot, plot_baselines2, cache_hits_to_percentage
# from scheduler.scheduler_policy import RayPolicy, FIFOPolicy, MRU


class CustomEvaluator:
    def __init__(self, env_func, cfg):
        self.env_func = env_func
        self.cfg = cfg
        self.baseline_cache: dict[int, float] = {}

    def dummy_eval_function(self, algorithm, eval_workers):
        print("DUMMY EVAL!!!")
        print("DUMMY EVAL!!!")
        print("DUMMY EVAL!!!")
        print("DUMMY EVAL!!!")
        return {"episode_reward_mean":1, "y":1}


    def custom_eval_function(self, algorithm, eval_workers):
        """Example of a custom evaluation function.
        Args:
            algorithm: Algorithm class to evaluate.
            eval_workers: Evaluation WorkerSet.
        Returns:
            metrics: Evaluation metrics dict.
        """
        verbose = False

        def verbose_print(x):
            if verbose:
                print(x)

        verbose_print("Custom eval start")

        # We configured 2 eval workers in the training config.
        def set_eval_mode(env, seed):
            env.set_seed(seed)
            # verbose_print(f"training mode in evaluation is{env.training_mode}")
            env.training_mode = False
            env.set_eval_mode()
            env.reset(seed=seed)
            # verbose_print(f"training mode in evaluation is{env.training_mode}")

        def set_baseline_mode(env, seed):
            env.set_seed(seed)
            # verbose_print(f"training mode in evaluation is{env.training_mode}")
            env.training_mode = False
            env.return_latencies_at_done = False  # No need when baselines
            # env.set_eval_mode()
            env.reset(seed=seed)
            # verbose_print(f"training mode in evaluation is{env.training_mode}")

        num_workers = len(eval_workers._remote_workers)

        diffs = []
        idle_actions = []
        illegal_actions = []
        makespans = []
        cache_hit_rates = []
        expireds = []
        exec_loads = {}
        task_loads = {}
        task_batch_loads = {}
        start_seed = 100  # Use different seeds than in the final evaluation of the model (to avoid overfitting)
        evaluations_per_worker = 2  # Each env gets its own seed
        for i in range(evaluations_per_worker):
            # reset_to_baseline_mode = []
            # reset_to_ppo_mode = []
            # measure_baselines = []
            # pending_result_seeds = []
            # for result_index, worker in enumerate(range(num_workers)):
            #     seed = start_seed * (worker + 1) + i
            #     # NOTE: Lambda default-value tricks are needed inside this for-loop,
            #     #       otherwise all seeds will have the same value
            #     reset_to_baseline_mode.append(lambda w, _seed=seed: w.foreach_env(lambda env, __seed=_seed: set_baseline_mode(env, __seed)))
            #     reset_to_ppo_mode.append(lambda w, _seed=seed: w.foreach_env(lambda env, __seed=_seed: set_eval_mode(env, __seed)))
            #     result_in_cache = seed in self.baseline_cache
            #     if result_in_cache:
            #         verbose_print(f"seed in cache {seed}")
            #         measure_baselines.append(lambda w: None)
            #     else:
            #         # TODO: Measure all baselines
            #         verbose_print(f"seed not in cache {seed}")
            #         measure_baselines.append(lambda w, _seed=seed: w.foreach_env(lambda env, __seed=_seed: plot_baselines2(env, self.cfg, f"Seed_{__seed}", seeds=[__seed], force_no_plotting=True)))
            #     pending_result_seeds.append((seed, result_in_cache))

            # reset_to_baseline_mode = lambda w: w.foreach_env(lambda env, __seed=w.worker_index: set_baseline_mode(env, __seed))
            # reset_to_ppo_mode = lambda w, _seed=seed: w.foreach_env(lambda env, __seed=_seed: set_eval_mode(env, __seed))
            # def test_return(w):
            #     return ("yep", "yeppers")
            def reset_to_baseline_mode(w):
                seed = start_seed * (w.worker_index + 1) + i
                w.foreach_env(lambda env, __seed=seed: set_baseline_mode(env, __seed))
            def reset_to_ppo_mode(w):
                seed = start_seed * (w.worker_index + 1) + i
                w.foreach_env(lambda env, __seed=seed: set_eval_mode(env, __seed))
            def measure_baselines(w):
                seed = start_seed * (w.worker_index + 1) + i
                result_in_cache = seed in self.baseline_cache
                if result_in_cache:
                    return (seed, [None])
                else:
                    return (seed, w.foreach_env(lambda env, __seed=seed: plot_baselines2(env, self.cfg, f"Seed_{__seed}", seeds=[__seed], force_no_plotting=True)))
                    # return (seed, w.foreach_env(lambda env: test_return(env)))
            attempts = 3
            baseline_results = []
            eval_workers.foreach_worker(func=reset_to_baseline_mode)
            baseline_results = eval_workers.foreach_worker(func=measure_baselines, local_worker=False)
            """
            while len(baseline_results) != len(measure_baselines) and attempts > 0:
                # NOTE: This loop is for debugging a rare case where one or more results were missing
                eval_workers.foreach_worker(func=reset_to_baseline_mode)
                baseline_results = eval_workers.foreach_worker(func=measure_baselines)
                attempts -= 1
                if len(baseline_results) != len(measure_baselines):
                    verbose_print(f"Eval returned invalid amount of results! Attempts remaining: {attempts}")
                    if attempts == 0:
                        return
            """

            for seed, result in baseline_results:
                result = result[0]  # Result is a list, but there is only one env -> only one element expected
                if result is None:
                    # This seed is already in cache! (or should be...?)
                    continue
                if len(result) < 2:
                    print(f"WTF {len(result)}, {seed}, {result}")
                    continue
                df, big_df = result
                df.reset_index(inplace=True, drop=True)  # TODO: df has index 0 on every row for some reason (this is why we need to reset here)
                min_idx = df["latency_avg"].idxmin()
                best_baseline_latency = df["latency_avg"][min_idx]
                if sum(df['expired_tasks'][min_idx]) > 0:
                    print("WARNING: best baseline policy has expired tasks!")
                verbose_print(
                    f"run_{seed} baseline executed: {best_baseline_latency}, {df['model_name'][min_idx]}, "
                    f"{df['expired_tasks'][min_idx]}")
                self.baseline_cache[seed] = best_baseline_latency

            # Reset envs
            eval_workers.foreach_worker(func=reset_to_ppo_mode)
            # Evaluate policy
            def get_infos(w):
                # w.sample() no longer returns final info with gymnasium
                # as a workaround, fetch the final info dict from the environment after sample()
                seed = start_seed * (w.worker_index + 1) + i
                return (seed, w.foreach_env(lambda env: env.older_info))
            policy_results = eval_workers.foreach_worker(func=lambda w: w.sample(), local_worker=False)
            policy_results = eval_workers.foreach_worker(func=get_infos, local_worker=False)

            for result_index, _ in enumerate(policy_results):
                seed, result = policy_results[result_index]
                verbose_print(f"run_{seed} reading cache...")
                best_baseline_latency = self.baseline_cache[seed]
                verbose_print(f"run_{seed} baseline from cache: {best_baseline_latency}")
                # print(result)
                last_info = result[0]  # TODO: currently assuming only one env per worker
                # print(last_info)
                idle_counter = last_info["idle_counter"]
                idle_actions.append(idle_counter)
                illegal_counter = last_info["illegal_counter"]
                illegal_actions.append(illegal_counter)
                exec_load_list = last_info["exec_loads"]
                for i in range(len(exec_load_list)):
                    if i not in exec_loads:
                        exec_loads[i] = []
                    exec_loads[i].append(exec_load_list[i])
                task_load_list = last_info["task_loads"]
                task_batch_load_list = last_info["task_batch_loads"]
                for i in range(len(task_batch_load_list)):
                    if i not in task_batch_loads:
                        task_batch_loads[i] = []
                    task_batch_loads[i].append(task_batch_load_list[i])
                    if i not in task_loads:
                        task_loads[i] = []
                    task_loads[i].append(task_load_list[i])
                latencies = last_info["latencies"]
                counts = last_info["task_counts"]
                makespan = last_info["makespan"]
                expired = last_info["expired"]
                expireds.append(expired)
                makespans.append(makespan)
                cache_hits = last_info["cache_hits"]
                cache_misses = last_info["cache_misses"]
                cache_hit_rate_per_task, cache_hit_rate = cache_hits_to_percentage(cache_hits, cache_misses)
                cache_hit_rates.append(cache_hit_rate)
                latencies_flat = np.array([latency for sublist in latencies for latency in sublist])
                avg_latency = latencies_flat.mean()
                diff = avg_latency / best_baseline_latency
                if expired > 0:
                    diff = 100  # Indicate poor evaluation results, as we cannot compute diff with expired tasks
                diffs.append(diff)
                verbose_print(f"run_{seed}, baseline: {best_baseline_latency}, policy_frac: {diff}, {counts}")

        episodes = collect_episodes(workers=eval_workers, timeout_seconds=99999)
        metrics = summarize_episodes(episodes)
        metrics["diff_mean"] = sum(diffs) / len(diffs)
        metrics["diff_min"] = min(diffs)
        metrics["diff_max"] = max(diffs)
        metrics["makespan_mean"] = sum(makespans) / len(makespans)
        metrics["makespan_min"] = min(makespans)
        metrics["makespan_max"] = max(makespans)
        metrics["cache_hit_rate_mean"] = sum(cache_hit_rates) / len(cache_hit_rates)
        metrics["cache_hit_rate_min"] = min(cache_hit_rates)
        metrics["cache_hit_rate_max"] = max(cache_hit_rates)
        metrics["expired_mean"] = sum(expireds) / len(expireds)
        metrics["idle_actions_mean"] = sum(idle_actions) / len(idle_actions)
        metrics["illegal_actions_mean"] = sum(illegal_actions) / len(illegal_actions)
        for i in range(len(exec_loads)):
            metrics[f"exec_load_{i}_mean"] = sum(exec_loads[i]) / len(exec_loads[i])
        for i in range(len(task_loads)):
            metrics[f"task_load_{i}_mean"] = sum(task_loads[i]) / len(task_loads[i])
        for i in range(len(task_batch_loads)):
            metrics[f"task_batch_load_{i}_mean"] = sum(task_batch_loads[i]) / len(task_batch_loads[i])
        return metrics


def get_envs(env_params=None, cfg=None):
    if env_params is None:
        # env_params = {'A': 4.722465546452744, 'B': 13.673180711407634, 'C': 14.554752981770758, 'D': 72.81689087235881,
        #               'E': 183.64120558651481, 'F': 1745.953323372009, 'cache_hit_coefficient': 0.9523525296649011,
        #               'interval': 45,
        #               'p2': 0.9954480821593636, 'p3': 0.5389942088288879}
        env_params = {'A': 5, 'B': 14, 'C': 13, 'D': 70,
                      'E': 180, 'F': 1700, 'cache_hit_coefficient': 0.9523525296649011,
                      'interval': 45,
                      'p2': 0.995, 'p3': 0.55}
    if cfg is not None:
        env_params["interval"] = cfg.sim.interval
        env_params["cache_hit_coefficient"] = cfg.sim.cache_hit_coefficient
    env_func = lambda cfg_x: get_custom_env2(cfg_x, env_params)
    def get_test_env(_cfg):
        _cfg = copy.copy(_cfg)
        # _cfg.sim.reward_scheme = 7
        _cfg.sim.streaming_tasks = _cfg.sim.eval_num_tasks
        env = env_func(_cfg)
        env.blind_mode = True
        return env
    def get_training_env(_cfg):
        x = "randomize_p3"
        if x in env_params and env_params[x] is True:
            env_params["p3"] = random.random()
        training_env = env_func(_cfg)
        training_env.set_training_mode()  # To generate random tasks and avoid problems with very low dag probabilities
        if x in env_params and env_params[x] is True:
            training_env.randomize_p3 = True
        return training_env
    # ActionMasker(training_env, mask_fn)
    return get_test_env, get_training_env


"""
Sets fixed seed for each training iteration in hopes of reducing training variance.
"""
class TrainingSeedCallback(DefaultCallbacks):
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict)
        self.iteration = 0

    def on_sub_environment_created(self, *, worker: "RolloutWorker", sub_environment: "EnvType",
                                   env_context: "EnvContext", env_index: Optional[int] = None, **kwargs) -> None:
        super().on_sub_environment_created(worker=worker, sub_environment=sub_environment, env_context=env_context,
                                           env_index=env_index, **kwargs)
        sub_environment.set_seed(9999 + self.iteration)

    def on_train_result(self, *, algorithm: "Algorithm", result: dict, **kwargs) -> None:
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        self.iteration += 1
        print(f"Traiiiiiiiiin {self.iteration}")
        algorithm.workers.foreach_worker(
            lambda ev: ev.foreach_env(
                lambda env: env.set_seed(9999 + self.iteration)))