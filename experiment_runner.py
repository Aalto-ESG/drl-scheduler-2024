import os
import sys

import numpy as np

from src.ray_utils.ray_train_pbt import ray_train_model

from ray.air import Checkpoint
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from ray.train.rl import RLCheckpoint
from ray.util.multiprocessing import Pool

from src.experiments.evaluation import eval_and_plot
from src.ray_utils.ray_eval import get_envs
from src.scheduler.scheduler_policy import RayPolicyLazy, ShortestJobFirstPolicy, FIFOPolicy, BatchFIFOPolicy, BatchMRU
from src.scheduler.scheduler_policy import BottomTaskFirst, TopTaskFirst, \
    RandomPolicy, MRU

from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore

sys.path.append(os.getcwd())
from src.conf.config_classes import SchedulerConfig
import ray
# from gymnasium.wrappers import EnvCompatibility
import hydra
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

cs = ConfigStore.instance()
cs.store(name="sim_conf", node=SchedulerConfig)


@hydra.main(version_base="1.2", config_path="src/conf", config_name="sim_conf")
def main(cfg: SchedulerConfig):
    ray.init()  # local_mode=False, ignore_reinit_error=True, num_cpus=30, object_store_memory=12*10**9,  _system_config={ 'automatic_object_spilling_enabled':False }, )
    global cfgi
    """
    EVAL PARAMS
    """
    print(f"Smoke_test: {cfg.smoke_test}")
    cfg.sim.eval_num_tasks = 10000 if not cfg.smoke_test else 500
    cfg.sim.done_on_expire = False  # False to make it easier to compare results between policies
    cfg.eval.iterations = 8  # Number of evaluation iterations
    cfg.training.layer_size = 64

    # cfg.tpstate.num_tasks_boolean = False
    # cfg.tpstate.one_hot_cache = False
    # cfg.tpstate.latencies_length = 1

    cfg.sim.cache_hit_coefficient = 1
    cfg.sim.interval = 8
    cfg.sim.max_batch = 10
    env_params = {'A': 10, 'B': 10, 'C': 10, 'D': 10,
                  'E': 10, 'F': 100,
                  # 'cache_hit_coefficient': 0.5,
                  # 'interval': 35,
                  'p2': 0.5, 'p3': 0.05,
                  "randomize_p3": True}

    """
    MULTIRUNNER PARAMS
    """
    iterations = 1  # Run multiple training and evaluation iterations
    prefer_checkpoint = True  # If true, load trained models from previous runs if they exist
    prefer_load_df_from_file = True  # If true, load evaluation results from previous runs if they exist
    training_only = False  # Skips evaluation if True
    use_unique_ppo_names = True  # False for "PPO"
    plot_immediately = False  # Old way of plotting
    base_run_name = "test_run_1"  # Name of this experiment - used as prefix for various output folders and files
    csv_output_path = "plotting/csv"
    normalize = True
    normalize_target_policy = "π_4"  # Normalize JCT values to this policy
    normalize_with_stored_vector = False  # Useful if the target policy is not included in the experiment
    include_executor_utilization = True
    include_baselines = True
    norm_vector = {0.0: 105.65317892469375, 0.1: 99.94897030542005, 0.2: 93.41521225899699,
                   0.30000000000000004: 87.21981363249438, 0.4: 80.83326113360424, 0.5: 74.58227782242304,
                   0.6000000000000001: 67.64866313822054, 0.7000000000000001: 60.88560885490026, 0.8: 52.59901379028196,
                   0.9: 43.23339078428654, 1.0: 34.68860044073374}

    def run_iteration(i):
        dfs = []
        exec_dfs = []
        for params in [
            (False, False, 0.1, "π_0"),
            # (False, False, 0.5, "π_1"),
            # (False, False, 0.9, "π_2"),
            # (False, True, 0.1, "π_3"),
            # (2, True, 0.1, "π_4"),   # Running average estimation for p(c) value
            (True, True, 0.1, "π_4"),  # Ground truth p(c) value
            # (2, True, 0.1, "π_4 (estimate p(c))"),

        ]:
            """
            TRAIN OR LOAD CHECKPOINT
            """
            final_plot_title = "fig_3"
            p3_included, randomize_p3, p3_training_value, policy_plot_name = params
            if p3_included == 2:
                p3_included = True
                cfg.hstate.approximate_p3 = True
                p3_name_suffix = "p3_approx"
            else:
                cfg.hstate.approximate_p3 = False
                p3_name_suffix = "p3_actual"

            env_params["randomize_p3"] = randomize_p3
            env_params["p3"] = p3_training_value
            cfg.hstate.include_p3 = p3_included
            run_name = base_run_name
            baseline_save_path = f"outputs_2/{run_name}"
            run_name = f"{run_name}_rand" if randomize_p3 else f"{run_name}_{p3_training_value}"
            run_name = f"{run_name}_{p3_name_suffix}" if p3_included else f"{run_name}_no_obs"
            wandb_group = f"{run_name}"  # Stack all iterations of same run in same group
            run_name += f"_{i}"
            checkpoint_path = "None"
            base_save_path = f"outputs_2/{run_name}"

            checkpoint_pointer_path = f"{base_save_path}/checkpoint.txt"
            os.makedirs(base_save_path, exist_ok=True)

            if prefer_checkpoint and os.path.exists(f"{os.getcwd()}/{checkpoint_pointer_path}"):
                with open(checkpoint_pointer_path, "r") as f:
                    checkpoint_path = f.readline()

            if os.path.exists(checkpoint_path):
                print(f"Got checkpoint from cache! ({checkpoint_path})")
                ModelCatalog.register_custom_model("TorchActionMaskModel", TorchActionMaskModel)
                c = Checkpoint(checkpoint_path)
                ppo_policy_lazy = RayPolicyLazy(RLCheckpoint.from_checkpoint(c))
                # ppo_policy_lazy.__str__ = lambda x: wandb_group
            else:
                ppo_policy, checkpoint_path = ray_train_model(cfg, wandb_project="drl-scheduler",
                                                              wandb_group=wandb_group, env_params=env_params)
                c = Checkpoint(checkpoint_path)
                ppo_policy_lazy = RayPolicyLazy(RLCheckpoint.from_checkpoint(c))
                # ppo_policy_lazy.__str__ = lambda x: wandb_group
                print(f"Saved in {checkpoint_path}")
                with open(f"{base_save_path}/checkpoint.txt", "w") as f:
                    f.write(checkpoint_path)

            # return
            if training_only:
                continue

            """
            EVALUATION
            """
            # for p3 in [0.1, 0.5, 0.9]:
            for p3 in np.linspace(0, 1, num=11):
                # if p3 < 0.8:
                #     continue
                lats = []
                env_params["p3"] = p3
                test_env_func, training_env_func = get_envs(env_params=env_params)
                dry_run_done = False
                policies = []
                if include_baselines:
                    # policies.append(BatchFIFOPolicy())
                    # policies.append(BatchMRU())
                    policies.append(MRU())
                    policies.append(BottomTaskFirst())
                    # policies.append(TopTaskFirst())
                    policies.append(RandomPolicy())
                    # policies.append(ShortestJobFirstPolicy())
                    policies.append(FIFOPolicy())
                policies.append(ppo_policy_lazy)
                for policy in policies:
                    if str(policy) == "PPO":
                        save_path = f"{base_save_path}/{policy}"
                    else:
                        save_path = f"{baseline_save_path}/{policy}"
                    os.makedirs(save_path, exist_ok=True)
                    _num_tasks = cfg.sim.eval_num_tasks
                    _iters = cfg.eval.iterations
                    eval_df_path_small = f"{save_path}/eval_small_{p3}_{_num_tasks}_{_iters}.feather"
                    eval_df_path_large = f"{save_path}/eval_large_{p3}_{_num_tasks}_{_iters}.feather"
                    if prefer_load_df_from_file and os.path.exists(eval_df_path_small) and os.path.exists(
                            eval_df_path_large):
                        # print(f"Loading...!")
                        small_df = pd.read_feather(eval_df_path_small)
                        large_df = pd.read_feather(eval_df_path_large)
                        print(f"Loaded eval from {eval_df_path_large}!")
                    else:
                        if not dry_run_done:
                            # NOTE: Dry run because first run uses wrong seed for no apparent reason
                            eval_and_plot(env=test_env_func(cfg), policy=MRU(),
                                          model_name=f"dry_run_because_first_seed_is_odd",
                                          case_name=cfg.sim.dag, cfg=cfg, force_no_plotting=True, seeds=[0])
                        multithreaded = True
                        if not multithreaded:
                            small_df, large_df = eval_and_plot(env=test_env_func(cfg), policy=policy,
                                                               model_name=f"{policy}",
                                                               case_name=cfg.sim.dag, cfg=cfg, force_no_plotting=True)
                        else:
                            with Pool() as pool:
                                f = lambda seed: eval_and_plot(env=test_env_func(cfg), policy=policy,
                                                               model_name=f"{policy}",
                                                               case_name=cfg.sim.dag, cfg=cfg, force_no_plotting=True,
                                                               seeds=[seed])
                                results = pool.map(f, range(cfg.eval.iterations))
                                small_dfs, large_dfs = zip(*results)
                                small_df = pd.concat(small_dfs, ignore_index=True)
                                large_df = pd.concat(large_dfs, ignore_index=True)
                        small_df.to_feather(eval_df_path_small)
                        large_df.to_feather(eval_df_path_large)
                        print(f"Saved eval to {eval_df_path_large}")

                    plt.show()

                    """
                    PLOTTING AND SAVING TO CSV
                    """
                    f = plt.figure(figsize=(4, 4))
                    all_iter_lats = []
                    for eval_iteration in range(len(large_df["latencies"].tolist())):
                        jct_df = pd.DataFrame({"latency": large_df["latencies"].tolist()[eval_iteration]})
                        # jct_df = pd.DataFrame({"latency": [large_df["latencies"].tolist()[eval_iteration].mean()]})
                        policy_name = str(policy)
                        if use_unique_ppo_names and policy_name == "PPO":
                            # policy_name = wandb_group
                            policy_name = policy_plot_name
                            pass

                        jct_df["policy"] = policy_name
                        jct_df["p3"] = p3
                        dfs.append(jct_df)
                    for eval_iteration in range(len(large_df["exec_loads"].tolist())):
                        exec_vals = large_df["exec_loads"].tolist()[eval_iteration]
                        exec_df = pd.DataFrame({"cpu_1": [exec_vals[0]],
                                                "cpu_2": [exec_vals[1]],
                                                "gpu_1": [exec_vals[2]],
                                                "gpu_2": [exec_vals[3]],
                                                "cpu_combined": [exec_vals[0] + exec_vals[1]],
                                                })
                        policy_name = str(policy)
                        if use_unique_ppo_names and policy_name == "PPO":
                            # policy_name = wandb_group
                            policy_name = policy_plot_name
                            pass

                        exec_df["policy"] = policy_name
                        exec_df["p3"] = p3
                        exec_dfs.append(exec_df)

                    print(f"Used checkpoint {checkpoint_path}")

                    # return
        dfs = pd.concat(dfs)
        data = dfs
        exec_dfs = pd.concat(exec_dfs)
        exec_dfs.to_csv(f"{csv_output_path}/fig_5.csv")
        print(f"Saved executor statistics to {csv_output_path}/fig_5.csv")

        if normalize:
            if normalize_with_stored_vector:
                ppo_p3_means = norm_vector
            else:
                name = normalize_target_policy
                data_mean = dfs.where(dfs["policy"] == name).groupby(["policy", "p3"], as_index=False).mean()
                ppo_p3_means = {}
                for p3 in data_mean["p3"].unique():
                    mean = data_mean.where(data_mean["p3"] == p3)["latency"].max()
                    ppo_p3_means[p3] = mean
                print(ppo_p3_means)
            data["normalized mean JCT"] = data.apply(lambda x: x["latency"] / ppo_p3_means[x["p3"]], axis=1)
            data.to_csv(f"{csv_output_path}/{final_plot_title}.csv")
            print(f"Saved eval results to {csv_output_path}/{final_plot_title}.csv")
            if plot_immediately:
                sns.lineplot(data, x="p3", y="normalized mean JCT", hue="policy")
        elif plot_immediately:
            sns.lineplot(data, x="p3", y="latency", hue="policy")

        if plot_immediately:
            plt.title(final_plot_title)
            fig: plt.Figure = plt.gcf()
            fig.set_size_inches(3.5, 3.0)
            plt.xlabel("p(c)")
            plt.ylabel("")
            plt.legend(loc='upper center')
            bottom, top = plt.ylim()
            # plt.ylim(0.90, 1.3)
            plt.tight_layout()
            # plt.autoscale()
            plt.show()

        # Exec loads plot:
        if include_executor_utilization and plot_immediately:
            for executor, plot_title in [("cpu_combined", "a) Average CPU utilization"),
                                         ("gpu_1", "b) Average GPU 1 utilization"),
                                         ("gpu_2", "c) Average GPU 2 utilization"),
                                         # ("cpu_1", "CPU 1"),
                                         # ("cpu_2", "CPU 2"),
                                         ]:
                colors = ["#229487", "#EDB732", "#A12864", "#A1A9AD", "#6BD650"]
                palette = sns.color_palette(colors)
                ax = sns.lineplot(exec_dfs, x="p3", y=executor, hue="policy", palette=palette)

                plt.title(plot_title, pad=30)

                fig: plt.Figure = plt.gcf()
                fig.set_size_inches(3.5, 3.0)
                plt.xlabel("p(c)")
                plt.ylabel("")
                # plt.legend(loc='upper center')
                leg = plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                                 mode="expand", borderaxespad=0, ncol=5,
                                 fontsize=10, handlelength=1)
                bottom, top = plt.ylim()
                top = 3 if "cpu" in executor else 6
                plt.ylim(0, top)

                # leg = ax.legend()
                leg_lines = leg.get_lines()
                leg_lines[0].set_linestyle("--")
                leg_lines[1].set_linestyle("--")
                leg_lines[2].set_linestyle("--")
                leg.get_frame().set_linewidth(0.0)
                ax.lines[0].set_linestyle(":")
                ax.lines[1].set_linestyle(":")
                ax.lines[2].set_linestyle(":")

                plt.grid(visible=True, axis="y")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                # ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                # ax.get_xaxis().set_ticks([])
                # ax.get_yaxis().set_ticks([])
                ax.tick_params(axis=u'both', which=u'both', length=0)
                # plt.yticks([i/2 for i in range(top*2, 1)])

                for i, text in enumerate(leg.get_texts()):
                    color = colors[i]
                    text.set_color(color)

                plt.tight_layout()
                # plt.autoscale()
                plt.show()

    for i in range(1, iterations + 1):  # Run multiple training and evaluation iterations
        run_iteration(i)


if __name__ == "__main__":
    print(os.getcwd())
    main()
