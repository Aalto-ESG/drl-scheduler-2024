import random

import numpy as np
from matplotlib import pyplot as plt
# from stable_baselines3.common.vec_env import SubprocVecEnv

from src.conf.config_classes import SchedulerConfig
from src.scheduler.scheduler_policy import SchedulerPolicy, FIFOPolicy, RoundRobin, BottomTaskFirst, TopTaskFirst, \
    RandomPolicy, LongestJobFirstPolicy, MRU, ShortestJobFirstPolicy
from src.simulator.tile_simulator import TileSimulator
import pandas as pd
from statistics import mean
import copy



def evaluate(policy: "SchedulerPolicy", env: TileSimulator, max_steps=None, fixed_seed=None):
    """
    Evaluates the simulation for given amount of steps, while always choosing the least-recently queued task.

    If steps==None, execute until environment reports that it is done.
    """
    # if type(env) == SubprocVecEnv:
    #     # TODO: Add support for multithreaded evaluation?
    #     raise Exception("SubprocVecEnv not yet supported in Evaluate()!")
    env.reset()
    if fixed_seed is not None:
        # random.seed(fixed_seed)
        # np.random.seed(fixed_seed)
        # env = copy.deepcopy(env)
        # prev_env = env
        # cfg = copy.deepcopy(env.cfg)
        # cfg.sim.use_deterministic_dag = True
        # env = TileSimulator(env.cfg, prev_env.task_tree, use_deterministic_dag=True)
        env.set_seed(fixed_seed)
        # env.blind_mode = prev_env.blind_mode
    step = 0
    reward_sum = 0
    next_obs = env.reset()
    rewards = []
    while True:
        # Choose the next task to execute
        # actions = policy.choose_task(next_obs, env)
        actions = policy.choose_multitask(next_obs, env)

        # Step the simulation
        next_obs, reward, done, truncated, info = env.step(actions)
        rewards.append(reward)
        reward_sum += reward
        step += 1
        if done or step == max_steps:
            break
    rewards = np.array(rewards)
    df_dict = eval_results_to_dict(env, rewards)
    return env, rewards, df_dict

def eval_results_to_dict(env, rewards):
    """
    Useful for creating dataframes from the results
    """
    cache_hits = env.task_pool.cache_hit_counters
    cache_misses = env.task_pool.cache_miss_counters
    cache_hit_rate_per_task, cache_hit_rate = cache_hits_to_percentage(cache_hits, cache_misses)
    batch_occupancy_counters = env.task_pool.batch_occupancy_counters
    latencies = env.task_history.get_end_to_end_latencies()
    latencies_flat = np.array([latency for sublist in latencies for latency in sublist])
    avg_latency = latencies_flat.mean()
    sum_latency = latencies_flat.sum()
    batch_occupancy = batch_occupancy_counters_to_percentage(batch_occupancy_counters)
    row_dict = {"ep_latency_avg": avg_latency,
                "ep_latency_sum": sum_latency,
                "ep_rewards_sum": np.sum(rewards),
                "ep_rewards_mean": np.mean(rewards),
                "expired_tasks": env.task_pool.expired_task_counters,
                "cache_hits": cache_hits,
                "cache_misses": cache_misses,
                "cache_hit_rate": cache_hit_rate,
                "cache_hit_rate_per_task": cache_hit_rate_per_task,
                "batch_occupancy_counters": batch_occupancy_counters,
                "batch_occupancy": batch_occupancy,
                "latencies": latencies_flat,
                "rewards": rewards,
                }
    return row_dict

def batch_occupancy_counters_to_percentage(batch_occupancy_counters: list[int]):
    batch_occupancy_ticks = np.sum(batch_occupancy_counters)
    batch_occupancy_max_tasks = batch_occupancy_ticks * len(batch_occupancy_counters)
    batch_occupancy_executed_tasks = [batch_occupancy_counters[i] * (i+1) for i in range(len(batch_occupancy_counters))]
    batch_occupancy_executed_tasks = np.sum(batch_occupancy_executed_tasks)
    batch_occupancy = batch_occupancy_executed_tasks / (batch_occupancy_max_tasks+1)  # +1 to avoid zero divisions
    return batch_occupancy


def cache_hits_to_percentage(hits, misses):
    cache_hit_max_tasks = [hits[i] + misses[i] for i in range(len(hits))]
    cache_hit_rate_per_task = [hits[i] / (cache_hit_max_tasks[i]+1) for i in range(len(hits))]  # +1 to avoid zero divisions
    cache_hit_rate = np.sum(hits) / (np.sum(hits) + np.sum(misses) + 1)  # +1 to avoid zero divisions
    return cache_hit_rate_per_task, cache_hit_rate



def eval_and_plot(env, policy, model_name, case_name, cfg: SchedulerConfig, seeds=None, force_no_plotting=None):
    mean_lat = []
    rew_agg = []
    mean_occupancy = []
    df_dicts_list = []
    if seeds is None:
        seeds = [x for x in range(cfg.eval.iterations)]
    for seed in seeds:
        env, rewards, df_dict = evaluate(policy, env, fixed_seed=seed)
        rew_agg.append(rewards)
        header = {}
        cache_hits = env.task_pool.cache_hit_counters
        cache_misses = env.task_pool.cache_miss_counters
        cache_hit_rate_per_task, cache_hit_rate = cache_hits_to_percentage(cache_hits, cache_misses)
        batch_occupancy_counters = env.task_pool.batch_occupancy_counters
        latencies = env.task_history.get_end_to_end_latencies()
        latencies_flat = np.array([latency for sublist in latencies for latency in sublist])
        avg_latency = latencies_flat.mean()
        sum_latency = latencies_flat.sum()
        batch_occupancy = batch_occupancy_counters_to_percentage(batch_occupancy_counters)
        exec_loads = [load/env.current_time for load in env.exec_load]  # Avg tasks in execution over time
        task_loads = [load/env.current_time for load in env.task_load]  # Avg tasks in execution over time
        row_dict = {"model_name": model_name,
                    "case_name": case_name,
                    "seed": seed,
                    "ep_latency_avg": avg_latency,
                    "ep_latency_sum": sum_latency,
                    "ep_rewards_sum": np.sum(rewards),
                    "ep_rewards_mean": np.mean(rewards),
                    "expired_tasks": env.task_pool.expired_task_counters,
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "cache_hit_rate": cache_hit_rate,
                    "cache_hit_rate_per_task": cache_hit_rate_per_task,
                    "exec_loads": exec_loads,
                    "task_loads": task_loads,
                    "batch_occupancy_counters": batch_occupancy_counters,
                    "batch_occupancy": batch_occupancy,
                    "latencies": latencies_flat,
                    "idle_counter": env.idle_counter,
                    "latencies_0": latencies[0],  # Used for plotting lats for task 0
                    "latencies_1": latencies[1] if len(latencies) > 1 else [],  # Used for plotting lats for task 1
                    "latencies_2": latencies[2] if len(latencies) > 2 else [],  # Used for plotting lats for task 2
                    "rewards": rewards,
                    }
        row_dict.update({"model_name": model_name,
                         "case_name": case_name,
                         "seed": seed})
        # df_rewards_list.append(reward_dict)
        expired_tasks = env.task_pool.expired_task_counters

        df_dicts_list.append(row_dict)
        mean_lat.append(avg_latency)
    df = pd.DataFrame(df_dicts_list)
    df["rewards_sum"] = df["ep_rewards_sum"].mean()
    df["rewards_mean"] = df["ep_rewards_mean"].mean()
    df["latency_avg"] = df["ep_latency_avg"].mean()
    df["latency_sum"] = df["ep_latency_sum"].mean()
    df_simplified = df.drop(columns=["rewards", "latencies"])  # DF without extremely large elements
    avg_reward_step = np.mean([np.mean(x) for x in rew_agg])
    avg_reward_ep = np.mean([np.sum(x) for x in rew_agg])
    cache_hit_max_tasks = [cache_hits[i] + cache_misses[i] for i in range(len(cache_hits))]
    cache_hit_rate = [cache_hits[i] / (cache_hit_max_tasks[i]+1) for i in range(len(cache_hits))]  # +1 to avoid zero divisions
    cache_hit_rate_str = [f"{x:.2f}" for x in cache_hit_rate]


    if not force_no_plotting:
        print(f"model: {model_name}, case: {case_name}, counts: {env.task_history.get_counts()}  reward_step: {avg_reward_step:.2f}, "
              f"reward_ep: {avg_reward_ep:.2f}, mean_latency: {np.mean(mean_lat):.2f}, last_lat: {avg_latency:.2f}, "
              f"expired: {expired_tasks}, cache_hits: {cache_hits}, cache_misses: {cache_misses}, cache_rate: {cache_hit_rate_str}, "
              f"batch_occupancy: {batch_occupancy_counters}, {batch_occupancy*100:.2f} %")
    if cfg.vis.plot and not force_no_plotting:
        fig, ax = plt.subplots()
        plot(env, steps=10000)
        plt.title(f"{model_name}, {case_name}, i: {cfg.sim.interval}, t: {cfg.sim.streaming_tasks}, "
                  f"b: {cfg.sim.max_batch}, cp: {cfg.sim.cache_hit_coefficient}, "
                  f"\nmean_latency: {np.mean(mean_lat):.2f}, last_lat: {avg_latency:.2f}, "
                  f"\nexpired: {expired_tasks}, "
                  # f"\ncache_hits: {cache_hits}, "
                  # f"\ncache_misses: {cache_misses}, "
                  f"\ncache_hit_rate: {cache_hit_rate_str}, "
                  f"\nbatch_occupancy: {batch_occupancy_counters}, {batch_occupancy*100:.2f} %")
        plt.tight_layout()
        # plt.show(block=False)
        # return fig, ax
    # else:
    #     print(f"model: {model_name}, mean_latency: {mean(mean_lat)}, expired: {expired_tasks}")
    return df_simplified, df


def eval_latency(env, policy, model_name, case_name):
    raise Exception("If you want to use this old function, fix it first. For example, expired tasks are handled wrong.")
    lats = []
    expired_sums = 0
    for i in range(20):
        env, rewards = evaluate(policy, env, fixed_seed=i)
        expired_tasks = env.task_pool.expired_task_counters
        expired_sums += sum(expired_tasks)
        latencies = env.task_history.get_end_to_end_latencies()
        [latencies[0].append(2000) for _ in range(sum(expired_tasks))]
        avg_latency = np.array([latency for sublist in latencies for latency in sublist]).mean()
        lats.append(avg_latency)
    avg_latency = np.array(lats).mean()
    print(f"model: {model_name}, case: {case_name}, reward_avg: {rewards.mean():.2f}, reward_sum: {rewards.sum():.2f}, mean_latency: {avg_latency}, expired: {expired_sums/10}")


def plot(env, steps):
    return env.task_history.plot_distributions(steps)


def plot_baselines(get_test_env, cfg, test_case, save_img_path=None, seeds=None):
    df_list = []
    large_df_list = []
    for policy in [FIFOPolicy(),  #RoundRobin(),
                   BottomTaskFirst(),
                   TopTaskFirst(),
                   RandomPolicy(),
                   # LongestJobFirstPolicy(),
                   MRU(),
                   # ShortestJobFirstPolicy()
                   ]:
        model_name = str(policy)
        df, large_df = eval_and_plot(get_test_env(cfg), policy, model_name, test_case, cfg, seeds=seeds)
        df_list.append(df)
        large_df_list.append(large_df)
        if save_img_path is not None:
            # fig.savefig(f"{save_img_path}/{str(policy)}.pdf")
            # fig.savefig(f"{save_img_path}/{str(policy)}.png")
            plt.savefig(f"{save_img_path}/{str(policy)}.pdf")
            plt.savefig(f"{save_img_path}/{str(policy)}.png")
    df = pd.concat(df_list)
    large_df = pd.concat(large_df_list)
    return df, large_df


def plot_baselines2(env, cfg, test_case, save_img_path=None, seeds=None, force_no_plotting=None):
    df_list = []
    large_df_list = []
    for policy in [FIFOPolicy(),  #RoundRobin(),
                   BottomTaskFirst(),
                   TopTaskFirst(),
                   RandomPolicy(),
                   # LongestJobFirstPolicy(),
                   MRU(),
                   # ShortestJobFirstPolicy()
                   ]:
        model_name = str(policy)
        df, large_df = eval_and_plot(env, policy, model_name, test_case, cfg, seeds=seeds, force_no_plotting=force_no_plotting)
        df_list.append(df)
        if force_no_plotting:
            large_df = pd.DataFrame()  # Try to save memory by not collecting these large dfs
        large_df_list.append(large_df)
        if save_img_path is not None and not force_no_plotting:
            # fig.savefig(f"{save_img_path}/{str(policy)}.pdf")
            # fig.savefig(f"{save_img_path}/{str(policy)}.png")
            plt.savefig(f"{save_img_path}/{str(policy)}.pdf")
            plt.savefig(f"{save_img_path}/{str(policy)}.png")
    df = pd.concat(df_list)
    large_df = pd.concat(large_df_list)
    return df, large_df


def plot_reward():
    import os
    cwd = os.getcwd()
    print(cwd)
    df = pd.read_csv("./tmp/sb3_log/progress.csv")
    s = df["rollout/ep_rew_mean"]
    plt.figure(5)
    plt.title("Rewards of RL Model")
    lines = s.plot.line()

