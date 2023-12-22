import math
import os
import time
import sys

from ray.tune.schedulers import PopulationBasedTraining

from ray.air import FailureConfig, Checkpoint
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.models.action_mask_model import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
from ray.train.rl import RLCheckpoint
from ray.tune.stopper import TrialPlateauStopper

from src.experiments.evaluation import eval_and_plot, plot_baselines
from src.ray_utils.ray_eval import CustomEvaluator, get_envs, TrainingSeedCallback
from src.ray_utils.ray_stoppers import MaxValueStopper, MinValueStopper
from src.scheduler.scheduler_policy import RayPolicy, RayPolicyLazy, MRU, ShortestJobFirstPolicy, FIFOPolicy
from src.scheduler.scheduler_policy import SchedulerPolicy, FIFOPolicy, RoundRobin, BottomTaskFirst, TopTaskFirst, \
    RandomPolicy, LongestJobFirstPolicy, MRU, ShortestJobFirstPolicy

# # os.environ['KMP_DUPLICATE_LIB_OK']='True'  # NOTE: This should not be needed, might cause other issues!

from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore

from ray import air
from ray import tune

sys.path.append(os.getcwd())
from src.conf.config_classes import SchedulerConfig
import ray
from ray.tune.registry import register_env
from ray.rllib.env.env_context import EnvContext
# from gymnasium.wrappers import EnvCompatibility
from munch import DefaultMunch
from ray.air.integrations.wandb import WandbLoggerCallback
import hydra
import wandb


def ray_train_model(cfg: SchedulerConfig, wandb_project: str, wandb_group: str, env_params):
    cfg = DefaultMunch.fromDict(cfg)

    test_env_func, training_env_func = get_envs(env_params=env_params)

    def ray_get_env(env_context: EnvContext):
        cfg = DefaultMunch.fromDict(env_context["cfg"])
        if "reward" in env_context:
            cfg.sim.reward_scheme = env_context["reward"]
        if "num_tasks" in env_context:
            cfg.sim.streaming_tasks = env_context["num_tasks"]
        if "training_tasks" in env_context:
            cfg.sim.batch_tasks = env_context["training_tasks"]
        if "streaming_ratio" in env_context:
            ratio = env_context["streaming_ratio"]
            cfg.sim.streaming_tasks = int(ratio * cfg.sim.batch_tasks) + 1
            cfg.sim.batch_tasks = int((1 - ratio) * cfg.sim.batch_tasks) + 1
        if "training" in env_context and env_context["training"] == True:
            env = training_env_func(cfg)
        else:
            cfg.sim.streaming_tasks = 1000
            cfg.sim.batch_tasks = 0
            cfg.eval.iterations = 1
            env = training_env_func(cfg)
            env.set_seed(0)
            env.training_mode = False

        if "blind" in env_context:
            env.blind_mode = env_context["blind"]
        return env

    ModelCatalog.register_custom_model("TorchActionMaskModel", TorchActionMaskModel)

    register_env("rl-scheduler", ray_get_env)

    from ray.rllib.algorithms.appo import APPOConfig
    from ray.rllib.algorithms.ppo import PPOConfig
    # for reward in [0, 1, 2, 3, 4, 5, 6, 7]:
    training_runs = []
    for reward in [6]:
        # config = APPOConfig()
        config = PPOConfig()
        config = config.resources(num_gpus=0, num_cpus_per_worker=0.1).framework("torch")
        config = config.rollouts(num_rollout_workers=20 if not cfg.smoke_test else 2, rollout_fragment_length=400)  # , rollout_fragment_length=300)
        config = config.environment("rl-scheduler")  # , disable_env_checking=True)
        k = 10 if not cfg.smoke_test else 1
        config = config.training(  # lr=tune.grid_search([ 0.005, 0.01, 0.025, 0.05]),
            lr=0.0005 * math.sqrt(k),
            num_sgd_iter=5,
            # train_batch_size=tune.grid_search([ 1000, 3000, 10000 ]),
            train_batch_size=8000 * k,
            sgd_minibatch_size=128 * k,
            model={"fcnet_hiddens": [cfg.training.layer_size for _ in range(cfg.training.num_layers)],
                   "use_attention": False},
        )
        config.env_config = {"cfg": cfg, "training": True,
                             # "reward": tune.grid_search([6, 7]),
                             "reward": 9,
                             # "interval": cfg.sim.interval,
                             # "cp": cfg.sim.cache_hit_coefficient,
                             # "training_tasks": cfg.sim.batch_tasks,
                             # "training_tasks":  tune.grid_search([ 20, 40]),
                             "training_tasks": 5000 if not cfg.smoke_test else 500,
                             # "streaming_ratio": tune.grid_search([ 0.1, 0.9]),
                             "streaming_ratio": 1,
                             }
        evaluator = CustomEvaluator(ray_get_env, cfg)
        config = config.evaluation(
            evaluation_num_workers=5,
            evaluation_interval=4,
            evaluation_duration=1,
            evaluation_parallel_to_training=False,
            evaluation_config={"env_config": {"cfg": cfg, "training": False}},
            custom_evaluation_function=evaluator.custom_eval_function,
        )
        config.model["custom_model"] = "TorchActionMaskModel"
        # config = config.callbacks(TrainingSeedCallback)
        print(config.to_dict())

        def explore(config):
            print(config)
            # ensure we collect enough timesteps to do sgd
            if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
                config["train_batch_size"] = config["sgd_minibatch_size"] * 2
            # ensure we run at least one sgd iter
            if config["num_sgd_iter"] < 1:
                config["num_sgd_iter"] = 1
            return config

        tuner = tune.Tuner(
            "PPO",
            run_config=air.RunConfig(
                stop=[
                    MaxValueStopper("timesteps_total", 5_000_000 if not cfg.smoke_test else 50000),
                ],
                checkpoint_config=air.CheckpointConfig(
                    # episode_reward_mean
                    checkpoint_score_attribute="episode_reward_mean",
                    checkpoint_score_order="max",
                    # checkpoint_score_attribute="evaluation/diff_mean",
                    # checkpoint_score_order="min",
                    # num_to_keep=1,
                    checkpoint_frequency=1, ),
                failure_config=FailureConfig(max_failures=1),
                verbose=2,
                callbacks=[
                    # WandbLoggerCallback(group="experiment-7-20_7_2023", project="indin-pbt", save_checkpoints=False),
                    # WandbLoggerCallback(group=wandb_group, project=wandb_project, save_checkpoints=False),
                ]
            ),
            param_space=config.to_dict(),
            # tune_config=tune.TuneConfig(
            #     scheduler=PopulationBasedTraining(
            #         # time_attr="num_agent_steps_sampled",
            #         time_attr="time_total_s",
            #         perturbation_interval=60*10,
            #         metric="episode_reward_mean",
            #         mode="max",
            #         hyperparam_mutations={
            #             # "lambda": tune.uniform(0.9, 1.0),
            #             # "clip_param": tune.uniform(0.01, 0.5),
            #             "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            #             "num_sgd_iter": tune.randint(1, 10),
            #             "sgd_minibatch_size": tune.randint(100, 1000),
            #             "train_batch_size": tune.randint(20000, 80000),
            #         },
            #         custom_explore_fn=explore,
            #     ),
            #     num_samples=1,
            # )
        )
        analysis = tuner.fit()
        training_runs.append(analysis)

    cfg.eval.iterations = 5
    best_checkpoint = None
    best_reward = None
    for analysis in training_runs:
        for result in analysis:
            for checkpoint in result.best_checkpoints:
                try:
                    reward = checkpoint[1]['episode_reward_mean']
                    if best_reward is None or reward > best_reward or math.isnan(best_reward):
                        best_reward = reward
                        best_checkpoint = checkpoint
                    print(f"Checkpoint reward: {reward}")
                except:
                    print("Checkpoint has no reward attribute!")
    print(f"Best checkpoint reward: {best_checkpoint[1]['episode_reward_mean']}")
    checkpoint = best_checkpoint[0]
    checkpoint_path = checkpoint.uri.replace("file://", "").replace("\\", "/")
    return RayPolicyLazy(RLCheckpoint.from_checkpoint(checkpoint)), checkpoint_path


cs = ConfigStore.instance()
cs.store(name="sim_conf", node=SchedulerConfig)


@hydra.main(version_base="1.2", config_path="../conf", config_name="sim_conf")
def main(cfg: SchedulerConfig):
    ray.init()  # local_mode=False, ignore_reinit_error=True, num_cpus=30, object_store_memory=12*10**9,  _system_config={ 'automatic_object_spilling_enabled':False }, )
    global cfgi
    env_params = {'A': 10, 'B': 10, 'C': 10, 'D': 10,
                  'E': 100, 'F': 100,
                  # 'cache_hit_coefficient': 0.5,
                  # 'interval': 35,
                  'p2': 0.5, 'p3': 0.05,
                  "randomize_p3": True}
    # cfg.tpstate.num_tasks_boolean = False
    # cfg.tpstate.num_tasks = True
    # cfg.tpstate.avg_latencies = True
    # cfg.tpstate.latencies = False
    # cfg.tpstate.latencies_length = 1
    cfgi = cfg
    print(cfg.sim)
    print(cfg)
    print(f"Working directory: {os.getcwd()}")
    print(f"Hydra generated directory: {os.getcwd()}/{HydraConfig.get().run.dir}")

    # full_training_experiment(cfg)
    run_name = f"test_{env_params['p3']}"
    checkpoint_path = "None"
    if os.path.exists(checkpoint_path):
        print("Got checkpoint from cache!")
        ModelCatalog.register_custom_model("TorchActionMaskModel", TorchActionMaskModel)
        c = Checkpoint(checkpoint_path)
        ppo_policy = RayPolicyLazy(RLCheckpoint.from_checkpoint(c))
    else:
        ppo_policy, checkpoint_path = ray_train_model(cfg, "plot-3", run_name, env_params)
        print(f"Saved in {checkpoint_path}")

    # return
    cfg.sim.eval_num_tasks = 5000
    cfg.sim.done_on_expire = False  # For more fair evaluation
    cfg.eval.iterations = 1

    import matplotlib.pyplot as plt
    dfs = []
    lats = []
    import seaborn as sns
    for x in [0.1, 0.5, 0.9]:
        env_params["p3"] = x
        test_env_func, training_env_func = get_envs(env_params=env_params)
        base_save_path = f"images/{run_name}"
        os.makedirs(base_save_path, exist_ok=True)
        with open(f"{base_save_path}/checkpoint.txt", "w") as f:
            f.write(checkpoint_path)
        for policy in [MRU(),
                       BottomTaskFirst(),
                       TopTaskFirst(),
                       RandomPolicy(),
                       # ShortestJobFirstPolicy(),
                       # FIFOPolicy(),
                       ppo_policy
                       ]:
            df, large_df = eval_and_plot(env=test_env_func(cfg), policy=policy, model_name=f"{policy}",
                                         case_name=cfg.sim.dag, cfg=cfg)
            plt.show()
            import pandas as pd
            a = pd.DataFrame({"latency": large_df["latencies"].tolist()[0]})
            a["policy"] = str(policy)
            lats_0 = pd.DataFrame({"latency": large_df["latencies_0"].tolist()[0]})
            lats_0["policy"] = str(policy)
            # lats_0["name"] = f"{policy}-ABC"
            lats_0["job"] = f"ABC"
            lats_1 = pd.DataFrame({"latency": large_df["latencies_2"].tolist()[0]})
            lats_1["policy"] = str(policy)
            # lats_1["name"] = f"{policy}-ABDE"
            lats_1["job"] = f"ABDE"
            lats_2 = pd.DataFrame({"latency": large_df["latencies_1"].tolist()[0]})
            lats_2["policy"] = str(policy)
            # lats_2["name"] = f"{policy}-ABDF"
            lats_2["job"] = f"ABDF"
            lats.append(lats_0)
            lats.append(lats_1)
            lats.append(lats_2)

            f = plt.figure(figsize=(4, 4))
            lat_df = pd.concat([lats_0, lats_1, lats_2])
            sns.ecdfplot(data=lat_df, x="latency", hue="job", palette=["#3182bd", "#31a354", "#de2d26"])
            dfs.append(a)
            save_path = f"{base_save_path}/{policy}"
            os.makedirs(save_path, exist_ok=True)

            plt.xlim([0, 500])

            plt.title(f"Job completion time - {policy} \n (average: {df['ep_latency_avg'][0]:.0f} ms)"
                      f"\n{run_name}, x: {x}")
            plt.ylim([0, 1.1])
            plt.savefig(f"{save_path}/cdf_{policy}_{x}.pdf")
            plt.savefig(f"{save_path}/cdf_{policy}_{x}.png")
            # plt.tight_layout()
            plt.show()
            print(f"Saved to {save_path}")
            print(f"Used checkpoint {checkpoint_path}")



if __name__ == "__main__":
    print(os.getcwd())
    main()
