import random
# from gymnasium.spaces import Dict

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.conf.config_classes import SchedulerConfig
from .reward_schemes import RewardSchemes
from .state_schemes import StateSchemes
from .task_history import TaskHistory
from .task_tree import TaskTree
from .task_pool import TaskPool
from .queued_task import QueuedTask, ArchivedTask, PooledTask

from munch import DefaultMunch

# from queue import PriorityQueue
from .utils import HeapPriorityQueue




class TileSimulator(gym.Env):
    """ Gym-interface based Fog4ML production monitoring simulator """
    metadata = {'render.modes': ['human']}  # TODO: Is this needed?

    def __init__(self, cfg: SchedulerConfig, DAG: TaskTree, use_deterministic_dag: bool = False):
        super(TileSimulator, self).__init__()
        self.task_to_execute = None
        self.cfg = cfg
        self.cfg = DefaultMunch.fromDict(cfg)  # Move disk-based cfg to memory (otherwise it is very slow)
        self.cache_hit_coefficient = cfg.sim.cache_hit_coefficient
        self.task_tree = DAG
        self.task_history = TaskHistory(self.task_tree, cfg)
        self.task_pool = TaskPool(self.task_tree, cfg)
        self.event_queue: "HeapPriorityQueue[QueuedTask]" = HeapPriorityQueue()
        self.current_time = -1
        self.prev_time = -1
        self.idle_counter = 0
        self.illegal_counter = 0
        self.task_in_cache = None
        self.info = {}
        self.older_info = {}
        self.system_load = 0

        self.executors = [1, 1, self.cfg.sim.max_batch//2, self.cfg.sim.max_batch]  # eg: cpu batch 1, gpu batch 5
        self.executor_speed = [0.8, 0.8, 1, 1]  # CPUs slightly faster than GPUs
        self.idle_per_exec = [-1] * len(self.executors)   # Time when this executor becomes idle
        self.batched_per_exec = [0] * len(self.executors)   # Number of tasks currently in batch execution
        self.cache_per_exec = [None] * len(self.executors)
        self.exec_load = [0] * len(self.executors)
        self.task_load = [0] * self.task_tree.get_number_of_tasks()
        self.task_batch_load = [0] * self.task_tree.get_number_of_tasks()
        self.task_batch_load_times = [0] * self.task_tree.get_number_of_tasks()

        # Action vector size depends on the given task tree
        action_pool_size = self.task_tree.get_number_of_tasks()
        if self.cfg.sim.enable_idle_action:
            self.action_space = spaces.Tuple(spaces.Discrete(action_pool_size+1) for _ in range(len(self.executors)))
        else:
            self.action_space = spaces.Tuple(spaces.Discrete(action_pool_size) for _ in range(len(self.executors)))
        self.action_space_without_idle = spaces.Discrete(action_pool_size)  # TODO: This represents number of task types, not actions?
        self.idle_action = self.action_space_without_idle.n

        # Observation vector size depends on the given task tree
        # self.observation_vector_size = self.get_observations().size
        self.observation_vector_size = self.get_observations()["observations"].size
        # self.observation_space = spaces.Box(low=0, high=1, shape=(self.observation_vector_size,), dtype=np.float64)
        self.observation_space = spaces.Dict({
            "action_mask": spaces.Box(0, 1, shape=((action_pool_size + int(self.cfg.sim.enable_idle_action)) * len(self.executors),)),
            # "action_mask": spaces.Box(0, 1, shape=(len(self.executors), (action_pool_size + int(self.cfg.sim.enable_idle_action)) ,)),
            "observations": spaces.Box(low=0, high=1, shape=(self.observation_vector_size,), dtype=np.float64),
        })
        self.use_deterministic_dag = use_deterministic_dag
        self.training_mode = False
        self.cfg_sim = cfg.sim
        self.cfg_sim_max_batch = self.cfg_sim.max_batch
        self.cfg_sim_max_latency = self.cfg_sim.max_latency
        self.cfg_sim_tasks_can_expire = self.cfg_sim.tasks_can_expire
        self.cfg_sim_min_idling_time = self.cfg_sim.min_idling_time
        self.cfg_sim_reward_scheme = self.cfg_sim.reward_scheme
        self.cfg_sim_invalid_action_penalty = self.cfg_sim.invalid_action_penalty
        self.cfg_sim_step_penalty = self.cfg_sim.step_penalty
        self.cfg_sim_allow_idling = self.cfg.sim.allow_idling
        self.number_initial_tasks = self.cfg.sim.streaming_tasks
        self.initial_task_interval = self.cfg.sim.interval
        self.use_deterministic_dag = self.cfg.sim.use_deterministic_dag
        self.blind_mode = False
        self.fixed_seed = None
        self.return_latencies_at_done = False
        self.randomize_p3 = False

        # print(self.action_space.shape)
        # print(self.observation_space.shape)

    def set_eval_mode(self):
        self.return_latencies_at_done = True

    def set_seed(self, seed):
        self.fixed_seed = seed
        self.use_deterministic_dag = True

    def set_training_mode(self):
        self.training_mode = True

    def reset(self, *, seed=None, options=None):
        # print(f"RESET SEED :{self.fixed_seed}")
        # print(f"RESET SEED :{self.fixed_seed}")
        # print(f"RESET SEED :{self.fixed_seed}")
        # print(f"RESET SEED :{self.fixed_seed}")
        # print(f"RESET SEED :{self.fixed_seed}")
        # print(f"RESET SEED :{self.fixed_seed}")
        # print(self.fixed_seed)
        if self.randomize_p3:
            old_p3 = self.task_tree.p3_node.probabilities
            self.task_tree.randomize_p3()
            new_p3 = self.task_tree.p3_node.probabilities
            print(f"Old probs: {old_p3}, New probs: {new_p3}")
        self.task_history = TaskHistory(self.task_tree, self.cfg)
        self.task_pool = TaskPool(self.task_tree, self.cfg)
        self.event_queue: "HeapPriorityQueue[QueuedTask]" = HeapPriorityQueue()
        self.idle_per_exec = [-1] * len(self.executors)
        self.batched_per_exec = [0] * len(self.executors)
        self.current_time = -1
        self.prev_time = -1
        self.idle_counter = 0
        self.system_load = 0
        self.illegal_counter = 0
        self.task_in_cache = None
        self.cache_per_exec = [None] * len(self.executors)
        self.exec_load = [0] * len(self.executors)
        self.task_load = [0] * self.task_tree.get_number_of_tasks()
        self.task_batch_load = [0] * self.task_tree.get_number_of_tasks()
        self.task_batch_load_times = [0] * self.task_tree.get_number_of_tasks()
        # print(random.seed)
        # super().reset(seed=self.fixed_seed)
        # self.seed(self.fixed_seed)
        random.seed(self.fixed_seed)
        self.generate_initial_tasks(self.fixed_seed)
        if self.training_mode:
            self.generate_batch_tasks(self.fixed_seed)
        # observations = np.zeros(self.observation_space.shape)
        observations = self.get_observations()
        old_info = self.info
        self.info = {}
        self.older_info = old_info
        return observations, old_info

    def generate_batch_tasks(self, seed=None):
        """ Helps training rarely seen situations by initializing the task pool to some values """

        if seed:
            random.seed(seed)
            np.random.seed(seed)
        # for i in range(0, int(random.random() * self.cfg.sim.batch_tasks)):
        for i in range(0, self.cfg.sim.batch_tasks):
            # print("yay")
            task = random.choice(self.task_tree.all_tasks)
            if self.use_deterministic_dag or seed:
                task = task.get_deterministic_copy()
            # item = QueuedTask(event_time=0, task=task, first_seen=-np.random.exponential(0.5) * self.cfg_sim_max_latency)
            item = QueuedTask(event_time=0, task=task, first_seen=0)
            self.event_queue.put(item)  # (Time of event, task, first seen)

    def generate_initial_tasks(self, seed=None):
        # Save variables to allow sb3 use reset function in a reasonable way
        # self.number_initial_tasks = self.cfg.sim.num_tasks
        # self.initial_task_interval = self.cfg.sim.interval
        # self.use_deterministic_dag = self.cfg.sim.use_deterministic_dag

        # Fill event queue with incoming tile tasks
        if seed:
            random.seed(seed)
            np.random.seed(seed)
        # for i in range(0, self.number_initial_tasks*self.initial_task_interval, self.initial_task_interval):
        t = 0
        for i in range(self.number_initial_tasks):
            # t += self.initial_task_interval
            t += np.random.normal(self.initial_task_interval, self.initial_task_interval/2)
            if self.use_deterministic_dag or seed:
                task = self.task_tree.initial_node.get_deterministic_copy()
            else:
                task = self.task_tree.initial_node
            item = QueuedTask(event_time=t, task=task, first_seen=t)
            self.event_queue.put(item)  # (Time of event, task, first seen)

        # Ensure that there is at least one task in the task pool
        while self.task_pool.empty() and self.event_queue.qsize() > 0:
            self.process_next_event()


    def process_next_event(self):
        if self.event_queue.empty():
            self.current_time = min(self.idle_per_exec)  # Make sure we don't get stuck
            return
        event: QueuedTask = self.event_queue.get()
        if event.task.is_leaf():
            item = ArchivedTask(task=event.task, first_seen=event.first_seen, time_completed=event.event_time)
            self.task_history.add_completed_task(item)
        else:
            item = PooledTask(task=event.task, first_seen=event.first_seen)
            self.task_pool.put_task(item)
            # new_task = event.task
        if self.current_time < event.event_time:
            self.current_time = event.event_time
        if event.event_time < event.first_seen:
            print("WTF first_seen- wrong")
        # if self.current_time > self.idle_at_time:
        #     self.idle_at_time = self.current_time

    
    def get_observations(self):
        """
        Creates an observation vector from the current state of the simulation.

        The observation can be used as an input to a policy.
        :return:
        """
        debug = False
        # observations = StateSchemes.get_state(self, self.cfg.sim.state_scheme)
        task_pool_observations = self.task_pool.get_state(self.current_time, debug=debug)
        # history_observations = self.task_history.get_state()
        if self.cfg.hstate.include_p3 and not self.cfg.hstate.approximate_p3:
            history_observations = [self.task_history.p3_actual]
        elif self.cfg.hstate.include_p3 and self.cfg.hstate.approximate_p3:
            history_observations = [self.task_history.get_p3_estimate()]
        else:
            history_observations = []
        one_hot_cache_vector = np.zeros(self.task_tree.get_number_of_tasks())
        if self.task_in_cache != None:
            one_hot_cache_vector[self.task_tree.get_task_index(self.task_in_cache)] = 1
        if self.cfg.tpstate.one_hot_cache:
            obs = np.concatenate([task_pool_observations, one_hot_cache_vector, history_observations])
        else:
            obs = np.concatenate([task_pool_observations, history_observations])
        if debug:
            for x in task_pool_observations:
                if x < 0 or x > 1:
                    print("task_pool_observations out of range!")
            for x in one_hot_cache_vector:
                if x < 0 or x > 1:
                    print("one_hot_cache_vector out of range!")
            for x in history_observations:
                if x < 0 or x > 1:
                    print("history_observations out of range!")
        # return obs
        return {"action_mask": self.action_masks(), "observations": obs}

    def action_masks_as_list(self):
        # True for valid actions, False for invalid actions
        # Python list is way faster than creating numpy array
        full_mask = []
        for exec_idle in self.idle_per_exec:
            busy = exec_idle > self.current_time
            if busy:
                if self.cfg.sim.enable_idle_action:
                    tasks = [False for q in self.task_pool.task_queues] + [True]
                else:
                    tasks = [False for q in self.task_pool.task_queues]  # NOTE: makes no sense to mask all actions as false
            else:
                if self.cfg.sim.enable_idle_action:
                    tasks = [not q.empty() for q in self.task_pool.task_queues] + [True]
                else:
                    tasks = [not q.empty() for q in self.task_pool.task_queues]
            full_mask += tasks
        return full_mask

    def action_masks(self):
        # NOTE: Name of this function matches the name defined in sb3-contrib maskable ppo
        # True for valid actions, False for invalid actions
        return np.array(self.action_masks_as_list())

    
    def step(self, actions: list[int]):
        # Validate action
        # SB3 DQN does not handle action-masking -- replace invalid actions with valid actions
        # -- another option would be to ignore invalid actions and make the system idle (penalty based on idling?)

        # Compute stats
        time_delta = self.current_time - self.prev_time
        for exec_index in range(len(self.exec_load)):
            self.exec_load[exec_index] += self.batched_per_exec[exec_index] * time_delta
            if self.cache_per_exec[exec_index] is not None and self.batched_per_exec[exec_index] > 0:
                task = self.cache_per_exec[exec_index]
                self.task_load[task] += time_delta
                self.task_batch_load[task] += self.batched_per_exec[exec_index] * time_delta
                self.task_batch_load_times[task] += time_delta
        current_load = sum(self.batched_per_exec)
        max_load = sum(self.executors)
        self.system_load += (current_load / max_load) * time_delta

        # Begin step
        self.prev_time = self.current_time
        reward = 0
        for exec_index in range(len(self.executors)):
            action = actions[exec_index]
            max_batch = self.executors[exec_index]
            cached_task = self.cache_per_exec[exec_index]
            exec_coeff = self.executor_speed[exec_index]
            exec_busy = self.idle_per_exec[exec_index] > self.current_time
            if exec_busy:
                if (action != len(self.task_pool.task_queues) and self.cfg.sim.enable_idle_action) and action != None:
                    print(f"Action chosen for an executor that is already executing something else!")  # Should not happen
                    print(f"mask: {self.action_masks_as_list()}, actions: {actions}, index: {exec_index}")
                continue
            self.batched_per_exec[exec_index] = 0
            if action == self.idle_action and self.cfg.sim.enable_idle_action:
                self.idle_counter += 1
                self.idle_per_exec[exec_index] = self.current_time + self.cfg_sim_min_idling_time
                continue
            elif not self.task_pool.is_legal(action):
                # print(f"Illegal!!!!!  actions: {actions}, exec_index = {exec_index}")
                reward += self.cfg_sim_invalid_action_penalty
                self.illegal_counter += 1
                if self.cfg_sim_allow_idling:
                    # This will cause the simulation to idle for one step
                    action = None
                else:
                    # This can result to FIFO-like behavior with untrained policies
                    action = self.task_pool.get_first_legal()
                # self.task_pool.add_illegal_action()

            # Execute action

            self.task_to_execute = None
            self.next_task = None
            self.current_task_make_span_latency = None
            if action is not None:
                self.task_to_execute = self.task_pool.get_task(action)

                cache_hit = self.task_to_execute.task == cached_task
                cache_bonus = self.cache_hit_coefficient if cache_hit else 1
                # self.idle_at_time += self.task_to_execute.task.processing_time * cache_bonus
                exec_idle_at = self.current_time + self.task_to_execute.task.processing_time * cache_bonus * exec_coeff
                self.idle_per_exec[exec_index] = exec_idle_at
                self.cache_per_exec[exec_index] = action

                self.next_task = self.task_to_execute.task.sample_next()
                item = QueuedTask(event_time=exec_idle_at, task=self.next_task, first_seen=self.task_to_execute.first_seen)
                self.event_queue.put(item)
                self.task_pool.register_cache_hit(action, cache_hit)
                self.task_in_cache = self.task_to_execute.task
                batched = 1
                self.current_task_make_span_latency = (exec_idle_at - self.task_to_execute.first_seen)
                self.task_history.add_executed_task(self.task_to_execute.task)
                self.task_history.add_encountered_task(self.next_task)  # Note: We are cheating here, we do not know the next task yet!

                for i in range(max_batch - 1):
                    if self.task_pool.is_legal(action):
                        pooled_task = self.task_pool.get_task(action)
                        next = pooled_task.task.sample_next()
                        item = QueuedTask(event_time=exec_idle_at, task=next, first_seen=pooled_task.first_seen)
                        self.event_queue.put(item)
                        self.task_pool.register_cache_hit(action, cache_hit)
                        batched += 1
                        self.task_history.add_executed_task(pooled_task.task)
                        self.task_history.add_encountered_task(next)  # Note: We are cheating here, we do not know the next task yet!
                self.task_pool.register_batch_occupancy(batched)
                self.batched_per_exec[exec_index] = batched






        # Process events
        time_at_least_one_exec_available = min(self.idle_per_exec)
        # if action is None:
        #     # Idle for one step to avoid deadlocks
        #     # TODO: Instead of this, we could always process at least one timestep
        #     self.current_time += self.cfg_sim_min_idling_time
        #     self.process_next_event()
        at_least_once = False
        while self.event_queue.qsize() > 0 and (time_at_least_one_exec_available > self.current_time or self.task_pool.empty()
                                                or self.event_queue.queue[0].event_time <= self.current_time):
            self.process_next_event()
            at_least_once = True
        if not at_least_once:
            self.process_next_event()

        while self.event_queue.qsize() > 0 and self.event_queue.queue[0].event_time < self.current_time:
            # In case of idling, make sure to add ALL tasks that happened during idling period
            self.process_next_event()

        # Process timeouts
        num_tasks_expired = 0
        terminated_on_expire = False
        if self.cfg_sim_tasks_can_expire:
            tasks_expired = self.task_pool.expire_tasks(self.current_time, self.cfg_sim_max_latency)
            for task in tasks_expired:
                num_tasks_expired += 1
                expiration_time = task.first_seen + self.cfg_sim_max_latency
                item = ArchivedTask(task=self.task_tree.expired_node, first_seen=task.first_seen,
                                    time_completed=expiration_time)
                self.task_history.add_completed_task(item)
                terminated_on_expire = self.cfg.sim.done_on_expire
                # print(f"Terminated!!!! {terminated_on_expire}")




        # Collect return values
        self.done = terminated_on_expire or (self.event_queue.qsize() == 0 and self.task_pool.empty())
        # self.info = {}
        if self.blind_mode:
            # For quick evaluation of policies that do not require these values
            reward = RewardSchemes.penalize_long_makespan(self)
            return [0], reward, self.done, False, self.info
        if self.done and self.return_latencies_at_done:
            exec_loads = [load/self.current_time for load in self.exec_load]  # Avg tasks in execution over time
            task_loads = [load/self.current_time for load in self.task_load]  # Avg tasks in execution over time
            task_batch_loads = [self.task_batch_load[i]/self.task_batch_load_times[i] for i in range(len(self.task_batch_load))]  # Avg task batch size in execution over time
            self.info = {"latencies": self.task_history.get_end_to_end_latencies(),
                         "idle_counter": self.idle_counter,
                         "illegal_counter": self.illegal_counter,
                         "task_counts": self.task_history.get_counts(),
                         "cache_hits": self.task_pool.cache_hit_counters,
                         "cache_misses": self.task_pool.cache_miss_counters,
                         "makespan": self.current_time,
                         "expired": sum(self.task_pool.expired_task_counters),
                         "exec_loads": exec_loads,
                         "task_loads": task_loads,
                         "task_batch_loads": task_batch_loads,
                         }
        observations = self.get_observations()
        reward += RewardSchemes.get_reward(self, self.cfg_sim_reward_scheme)
        # print(reward)
        reward += self.cfg_sim_step_penalty
        # reward += -num_tasks_expired * 1
        # NOTE: Dont penalize more for expiring multiple tasks in one timestep (important only if episode does not continue)
        reward += -1 if num_tasks_expired > 0 else 0
        if terminated_on_expire:
            # Add penalties for ending episode too early
            # TODO: Penalties may need scaling adjustments? Ending episode early should never be better than continuing
            # TODO: Might be especially good to scale these by expected task length to avoid expiring long tasks
            # reward += -self.task_pool.get_total_tasks_in_queue()
            # reward += -self.event_queue.qsize()
            pass
            # missed_tasks = 0
            # while self.event_queue.qsize() > 0:
            #     # Put all remaining tasks in task poolsq
            #     self.process_next_event()
            # while self.task_pool.get_total_tasks_in_queue() > 0:
            #     # Compute the number of steps that would be simulated if the simulation was not terminated
            #     for i in range(len(self.task_pool.task_queues)):
            #         q = self.task_pool.task_queues[i]
            #         while not q.empty():
            #             task = q.pop().task
            #             missed_tasks += 1
            #             while not task.is_leaf():
            #                 task = task.sample_next()
            #                 missed_tasks += 1
            # reward -= missed_tasks


        # print(self.task_pool.get_total_tasks_in_queue())



        if self.done:
            print(f"DEEEEBUG: p3_estimate: {self.task_history.get_p3_estimate()}, p3_actual: {self.task_history.get_p3_actual()}")
            print(f"DEEEEBUG: system load: {self.system_load / self.current_time}, (expired: {terminated_on_expire})")
        return observations, reward, self.done, False, self.info

    def render(self, **kwargs):
        # TODO: Is there any useful info to "render"?
        return None

    def close(self):
        # TODO: Implement me
        return None
