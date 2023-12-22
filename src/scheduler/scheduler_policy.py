import random
from typing import Optional

import gymnasium
import numpy as np


from src.simulator.tile_simulator import TileSimulator


class SchedulerPolicy:

    def __init__(self, batch_aware=False):
        self.actions = None
        self.batch_aware = batch_aware
        self.name_prefix = ""
        if batch_aware:
            self.name_prefix = "Batch-"

    def choose_batch_optimized_multitask(self, state: np.array, simulator: TileSimulator) -> list[Optional[int]]:
        num_execs = len(simulator.executors)
        actions = [None] * num_execs
        for i in range(num_execs):
            self.actions = actions
            if simulator.idle_per_exec[i] > simulator.current_time:
                # Executor is still busy
                continue
            selected_action = self.choose_task(state, simulator)
            if selected_action is None:
                break
            task_amounts, task_arrival_times = self.get_tasks_and_times(simulator)
            executors_remaining = [index for index, a in enumerate(actions) if (a is None and simulator.idle_per_exec[index] <= simulator.current_time)]
            exec = self.choose_executor(executors_remaining, task_amounts[selected_action])
            actions[exec] = selected_action
        for i in range(num_execs):
            if actions[i] == None and simulator.cfg.sim.enable_idle_action:
                # TODO: We should already return the idle action, instead of changing here
                actions[i] = simulator.action_space_without_idle.n
        return actions

    def choose_multitask(self, state: np.array, simulator: TileSimulator) -> list[Optional[int]]:
        if self.batch_aware:
            return self.choose_batch_optimized_multitask(state, simulator)
        num_execs = len(simulator.executors)
        actions = [None] * num_execs
        for i in range(num_execs):
            self.actions = actions
            if simulator.idle_per_exec[i] > simulator.current_time:
                # Executor is still busy
                continue
            actions[i] = self.choose_task(state, simulator)
            if actions[i] == None and simulator.cfg.sim.enable_idle_action:
                # TODO: We should already return the idle action, instead of changing here
                actions[i] = simulator.action_space_without_idle.n
                # print(self.get_tasks_and_times(simulator))
        return actions

    def choose_executor(self, executors_remaining, num_tasks):
        # TODO: This is hard-coded to assume very specific properties for the executors - generalize it!
        if num_tasks == 1:
            # Prioritize low batch sizes (at start of the executor list)
            exec = executors_remaining[0]
        elif num_tasks < 6 and 2 in executors_remaining:
            # Prioritize GPU 1 as it has lower batch size of GPU 2
            exec = 2
        else:
            # Prioritize high batch sizes (at end of the executor list)
            exec = executors_remaining[-1]
        return exec


    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        pass


    def get_executed_tasks(self, simulator: TileSimulator):
        executed = [0] * simulator.action_space_without_idle.n
        for executor_index, task_type in enumerate(self.actions):
            if task_type != None and task_type != simulator.idle_action:
                    executed[task_type] += simulator.executors[executor_index]
        return executed

    # @staticmethod
    
    def get_tasks_and_times(self, simulator: TileSimulator):
        pooled_tasks = []
        arrival_times = []
        executed = self.get_executed_tasks(simulator)
        for i, q in enumerate(simulator.task_pool.task_queues):
            offset = executed[i]  # Account for number of tasks executed by other accels
            size = q.qsize() - offset
            pooled_tasks.append(size)
            if size > 0:
                arrival_times.append(q.queue[offset].first_seen)
            else:
                arrival_times.append(0)
        return pooled_tasks, arrival_times

    @staticmethod
    
    def get_mask_and_times(simulator: TileSimulator):
        action_mask = simulator.action_masks_as_list()
        num_queues = simulator.task_tree.get_number_of_tasks()
        arrival_times = [0] * num_queues
        for i, mask in enumerate(action_mask[:num_queues]):
            if mask is True:
                arrival_times[i] = simulator.task_pool.task_queues[i].queue[0].first_seen
        return action_mask, arrival_times

    @staticmethod
    def get_task_index_sorted_by_execution_time(simulator: TileSimulator, reverse: bool):
        DAG = simulator.task_tree
        all_nodes = DAG.get_all_tasks()
        times = np.zeros_like(all_nodes)
        for node in all_nodes:
            times[DAG.get_task_index(node)] = node.processing_time
        index_prios = sorted(range(len(times)), key=lambda k: times[k], reverse=reverse)
        return index_prios


class FIFOPolicy(SchedulerPolicy):
    """
    First in, First out

    Always choose the task that has been queued for the longest.
    """
    
    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        task_amounts, task_arrival_times = self.get_tasks_and_times(simulator)
        # pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        earliest_task_index = None
        min_value = 0
        num_queues = simulator.task_tree.get_number_of_tasks()
        arrival_times = [0] * num_queues
        for i, num_tasks in enumerate(task_amounts):
            if num_tasks > 0:
                val = task_arrival_times[i]
                if earliest_task_index is None or min_value > val:
                    earliest_task_index = i
                    min_value = val

        # earliest_task_index = np.ma.array(task_arrival_times, mask=np.logical_not(pooled_tasks)).argmin()
        # if pooled_tasks[earliest_task_index] == 0:
        #     earliest_task_index = None
        return earliest_task_index

    @classmethod
    def __str__(cls):
        return "FIFO"

class BatchFIFOPolicy(SchedulerPolicy):
    """
    Batch-aware FIFO

    Prioritizes large batches on GPUs and small batches on CPUs
    """
    def choose_multitask(self, state: np.array, simulator: TileSimulator) -> list[Optional[int]]:
        num_execs = len(simulator.executors)
        actions = [None] * num_execs
        for i in range(num_execs):
            self.actions = actions
            if simulator.idle_per_exec[i] > simulator.current_time:
                # Executor is still busy
                continue
            selected_action = self.choose_task(state, simulator)
            if selected_action is None:
                break
            task_amounts, task_arrival_times = self.get_tasks_and_times(simulator)
            # print("11111BASFBAS")
            executors_remaining = [index for index, a in enumerate(actions) if (a is None and simulator.idle_per_exec[index] <= simulator.current_time)]
            # print("22222BASFBAS")
            exec = self.choose_executor(executors_remaining, task_amounts[selected_action])
            actions[exec] = selected_action
        for i in range(num_execs):
            if actions[i] == None and simulator.cfg.sim.enable_idle_action:
                # TODO: We should already return the idle action, instead of changing here
                actions[i] = simulator.action_space_without_idle.n
                # print(self.get_tasks_and_times(simulator))
        return actions



    
    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        task_amounts, task_arrival_times = self.get_tasks_and_times(simulator)
        # pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        earliest_task_index = None
        min_value = 0
        num_queues = simulator.task_tree.get_number_of_tasks()
        arrival_times = [0] * num_queues
        for i, num_tasks in enumerate(task_amounts):
            if num_tasks > 0:
                val = task_arrival_times[i]
                if earliest_task_index is None or min_value > val:
                    earliest_task_index = i
                    min_value = val

        # earliest_task_index = np.ma.array(task_arrival_times, mask=np.logical_not(pooled_tasks)).argmin()
        # if pooled_tasks[earliest_task_index] == 0:
        #     earliest_task_index = None
        return earliest_task_index

    @classmethod
    def __str__(cls):
        return "Batch-FIFO"


class TopTaskFirst(SchedulerPolicy):
    """
    Shallow Task First

    Choose task the earliest in the DAG
    """

    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        shortest_task = None
        pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        for i in reversed(range(len(pooled_tasks))):
            if pooled_tasks[i] > 0:
                shortest_task = i
        return shortest_task

    @classmethod
    def __str__(cls):
        return "Top-level task first"


class BottomTaskFirst(SchedulerPolicy):
    """
    Deep Task First - aka Complete Whole DAG first

    Choose task the furthest down in the DAG
    """

    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        longest_task = None
        pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        for i in range(len(pooled_tasks)):
            if pooled_tasks[i] > 0:
                longest_task = i
        return longest_task

    @classmethod
    def __str__(cls):
        return "Depth first"


class RandomPolicy(SchedulerPolicy):
    """
    Random Policy will randomly choose one of the valid actions.
    """

    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        valid_indices = []
        for i in range(len(pooled_tasks)):
            if pooled_tasks[i] > 0:
                valid_indices.append(i)
        if len(valid_indices) == 0:
            return None
        else:
            return random.choice(valid_indices)

    @classmethod
    def __str__(cls):
        return "Random"


class CompletelyRandomPolicy(SchedulerPolicy):
    """
    Completely Random Policy can select invalid actions.
    """

    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        return random.choice(range(len(pooled_tasks)))

    @classmethod
    def __str__(cls):
        return "CompletelyRandom"


class AlwaysIdlePolicy(SchedulerPolicy):
    """
    Policy that never chooses an action.
    Instead, it always returns None.
    This should cause the simulator to idle on every step (and eventually expire all tasks).

    WARNING: will livelock if tasks never expire
    """

    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        return None

    @classmethod
    def __str__(cls):
        return "Always Idle"


class RoundRobin(SchedulerPolicy):
    """
    Round Robin (sort of)

    Alternate between task types, without any concern about arrival times or execution times
    """

    def __init__(self):
        super().__init__()
        self.prev_index = 0

    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        for i in range(1, len(pooled_tasks)+1):
            index = (self.prev_index + i) % simulator.action_space_without_idle.n
            if pooled_tasks[index] > 0:
                self.prev_index = index
                return index
        return None

    @classmethod
    def __str__(cls):
        return "Round Robin"

class MRU(SchedulerPolicy):
    """
    Most recently used. Maximises cache hits in a naive way.

    Keep selecting the same task as long as possible. Then switch to the next task.
    """

    def __init__(self):
        super().__init__()
        self.prev_index = 0

    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        for i in range(0, len(pooled_tasks)):
            index = (self.prev_index + i) % simulator.action_space_without_idle.n
            if pooled_tasks[index] > 0:
                self.prev_index = index
                return index
        return None

    @classmethod
    def __str__(cls):
        # return "Most Recently Used (MRU)"
        return "MRU"

class BatchMRU(MRU):
    """
    Most recently used. Maximises cache hits in a naive way.

    Keep selecting the same task as long as possible. Then switch to the next task.
    """

    def choose_multitask(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        return self.choose_batch_optimized_multitask(state, simulator)

    @classmethod
    def __str__(cls):
        # return "Most Recently Used (MRU)"
        return "Batch-MRU"


class LongestJobFirstPolicy(SchedulerPolicy):
    """
    Always choose task with the longest execution time.
    """
    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        for i in SchedulerPolicy.get_task_index_sorted_by_execution_time(simulator, reverse=True):
            if pooled_tasks[i] > 0:
                return i
        return None

    @classmethod
    def __str__(cls):
        return "Longest job first"


class ShortestJobFirstPolicy(SchedulerPolicy):
    """
    Always choose task with the shortest execution time.
    """
    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        pooled_tasks, task_arrival_times = self.get_tasks_and_times(simulator)
        for i in SchedulerPolicy.get_task_index_sorted_by_execution_time(simulator, reverse=False):
            if pooled_tasks[i] > 0:
                return i
        return None

    @classmethod
    def __str__(cls):
        return "Shortest job first"



class RayPolicy(SchedulerPolicy):
    def __init__(self, algo):
        super().__init__()
        self.algo = algo

    def choose_multitask(self, state: np.array, simulator: TileSimulator) -> list[Optional[int]]:
        # Assume that our policy already returns multiple actions (we could also check policy action space)
        return self.choose_task(state, simulator)

    
    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        if simulator.blind_mode:
            state = simulator.get_observations()
        inference_results = self.algo.compute_single_action(state)
        action = inference_results[0]
        return action

    @classmethod
    def __str__(cls):
        return "PPO"

class RayPolicyLazy(SchedulerPolicy):
    """
    Ray policies initialize workers. Workers cannot be passed as pickled arguments (when multiprocessing).

    What we can do, is pickle the checkpoint and then initialize the workers only when actually needed.
    """
    def __init__(self, checkpoint):
        super().__init__()
        self.checkpoint = checkpoint
        self.policy = None

    def choose_multitask(self, state: np.array, simulator: TileSimulator) -> list[Optional[int]]:
        # Assume that our policy already returns multiple actions (we could also check policy action space)
        return self.choose_task(state, simulator)

    
    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        if self.policy is None:
            self.policy = self.checkpoint.get_policy()
        if simulator.blind_mode:
            state = simulator.get_observations()
        inference_results = self.policy.compute_single_action(state)
        action = inference_results[0]
        return action

    @classmethod
    def __str__(cls):
        return "PPO"


class SBPolicyEpsilon(SchedulerPolicy):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def choose_task(self, state: np.array, simulator: TileSimulator) -> Optional[int]:
        action, _states = self.model.predict(state)
        return action
