import types
from typing import Optional

import numpy as np

from src.conf.config_classes import SchedulerConfig
from .task_tree import TaskTree
from .queued_task import PooledTask



from .utils import HeapPriorityQueue


class TaskPool:
    """
    Stores tasks that have not been executed yet.
    """

    def __init__(self, task_tree: TaskTree, cfg: SchedulerConfig):
        self.task_tree = task_tree
        num_queues = task_tree.get_number_of_tasks()
        self.task_queues: "list[HeapPriorityQueue[PooledTask]]" = [HeapPriorityQueue() for _ in range(num_queues)]
        self.first_seen_sums: "list[int]" = [0 for _ in range(num_queues)]
        self.expired_task_counters: list[int] = [0 for _ in range(num_queues)]
        self.cache_hit_counters: list[int] = [0 for _ in range(num_queues)]
        self.cache_miss_counters: list[int] = [0 for _ in range(num_queues)]
        self.batch_occupancy_counters: list[int] = [0 for _ in range(cfg.sim.max_batch)]
        self.deepQ_eval = False
        self.illegal_actions: int = 0
        self.cfg = cfg
        self.speed_cfg = None

    def register_batch_occupancy(self, number_batched):
        # TODO: Account for multiple executors
        self.batch_occupancy_counters[number_batched-1] += 1

    def register_cache_hit(self, task_type: int, hit: bool):
        if hit:
            self.cache_hit_counters[task_type] += 1
        else:
            self.cache_miss_counters[task_type] += 1

    def expire_tasks(self, current_time, expiration_latency):
        min_first_seen = current_time - expiration_latency
        expired_tasks = []
        for i, queue in enumerate(self.task_queues):
            while queue.qsize() > 0:
                if queue.queue[0].first_seen < min_first_seen:
                    # task = queue.get()
                    task = self._get_task(i)
                    self.expired_task_counters[i] += 1
                    expired_tasks.append(task)
                else:
                    break
        return expired_tasks

    def get_total_tasks_in_queue(self):
        total_tasks = 0
        for q in self.task_queues:
            total_tasks += len(q.queue)
        return total_tasks

    def get_total_waiting_time(self, current_time):
        total_time = 0
        for q in self.task_queues:
            for task in q.queue:
                total_time += current_time - task.first_seen
        return total_time

    
    def get_avg_latencies_per_queue(self, current_time, max_latency):
        # avg_lats = np.zeros(len(self.task_queues))
        avg_lats = [0] * len(self.task_queues)
        for i in range(len(self.task_queues)):
            if not self.task_queues[i].empty():
                # TODO: Clamping to [0,1] hides bugs where first seen is in future (but fixes potential issues with float-precision)
                avg_first_seen = min(current_time, self.first_seen_sums[i] / self.task_queues[i].qsize())
                avg_lats[i] = max(0, (current_time - avg_first_seen) / max_latency)
                avg_lats[i] = min(1, avg_lats[i])
        return avg_lats

    
    def get_state(self, current_time: int, normalized=False, debug=False) -> np.array:
        """
        Get state of the pool as an array, that can be used as a neural network input.
        """
        if self.speed_cfg is None:
            # Accessing some hydra config values is extremely slow (~8x speedup)
            self.speed_cfg = types.SimpleNamespace()
            self.speed_cfg.num_tasks_boolean = self.cfg.tpstate.num_tasks_boolean
            self.speed_cfg.num_tasks = self.cfg.tpstate.num_tasks
            self.speed_cfg.max_latency = self.cfg.sim.max_latency
            self.speed_cfg.avg_latencies = self.cfg.tpstate.avg_latencies
            self.speed_cfg.latencies = self.cfg.tpstate.latencies
            self.speed_cfg.latencies_length = self.cfg.tpstate.latencies_length

        speed_cfg = self.speed_cfg
        # state_vec = np.zeros(shape=1)
        state_vec = []
        max_latency = self.speed_cfg.max_latency

        # Task amounts boolean
        if speed_cfg.num_tasks_boolean:
            num_tasks_boolean = [not q.empty() for q in self.task_queues]
            # state_vec = np.concatenate([state_vec, np.array(num_tasks_boolean)])
            state_vec += num_tasks_boolean
            if debug:
                for x in num_tasks_boolean:
                    if x < 0 or x > 1:
                        print("num_tasks_boolean out of range!")

        # Task amounts normalized
        if speed_cfg.num_tasks:
            max_tasks = 50
            num_tasks = [min(q.qsize() / max_tasks, 1) for q in self.task_queues]
            state_vec += num_tasks
            # state_vec = np.concatenate([state_vec, np.array(num_tasks)])
            if debug:
                for x in num_tasks:
                    if x < 0 or x > 1:
                        print("num_tasks out of range!")

        # Avg latency over all tasks in each queue
        if speed_cfg.avg_latencies:
            # avg_latencies = np.zeros(len(self.task_queues))
            # for i, q in enumerate(self.task_queues):
            #     avg_latency = 0
            #     if len(q.queue) > 0:
            #         latencies = []
            #         for task in q.queue:
            #             latency = current_time - task.first_seen
            #             latencies.append(latency)
            #         avg_latency = np.mean(latencies) / max_latency
            #     avg_latencies[i] = avg_latency
            avg_latencies = self.get_avg_latencies_per_queue(current_time, max_latency)
            # state_vec = np.concatenate([state_vec, np.array(avg_latencies)])
            state_vec += avg_latencies
            if debug:
                for x in avg_latencies:
                    if x < 0 or x > 1:
                        print("avg_latencies out of range!")

        # Task latency over N tasks in each queue
        if speed_cfg.latencies:
            num_latencies = speed_cfg.latencies_length
            # latencies = np.zeros(len(self.task_queues) * num_latencies)
            latencies = [0] * len(self.task_queues) * num_latencies
            for i, queue in enumerate(self.task_queues):
                for k in range(num_latencies):
                    index = i * num_latencies + k
                    if k < len(queue.queue):
                        latency = current_time - queue.queue[k].first_seen
                        latencies[index] = latency / max_latency
                    else:
                        # No more tasks in this queue
                        break
            # state_vec = np.concatenate([state_vec, latencies])
            state_vec += latencies
            if debug:
                for x in latencies:
                    if x < 0 or x > 1:
                        print("latencies out of range!")
        state_vec = np.array(state_vec)
        return state_vec




        # task_amounts = [q.qsize() for q in self.task_queues]
        # task_waiting_times = [current_time - q.queue[0].first_seen if q.qsize() > 0 else 0 for q in self.task_queues]
        # if normalized:  # Normalize to values between [0, 1]
        #     max_tasks = self.cfg.sim.num_tasks
        #     max_waiting_time = self.cfg.sim.max_latency
        #     task_amounts = np.array([x / max_tasks for x in task_amounts])
        #     task_waiting_times = np.array([t / max_waiting_time for t in task_waiting_times])
        # return task_amounts * task_waiting_times

    def _put_task(self, item: PooledTask, queue_index: int):
        self.task_queues[queue_index].push(item)
        self.first_seen_sums[queue_index] += item.first_seen

    def _get_task(self, queue_index: int):
        task = self.task_queues[queue_index].pop()
        self.first_seen_sums[queue_index] -= task.first_seen
        return task

    def put_task(self, item: PooledTask):
        """
        Put given task in the task pool.
        """
        index = self.task_tree.get_task_index(item.task)
        # self.task_queues[index].put(item)
        self._put_task(item, index)

    def get_task(self, task_type) -> PooledTask:
        """
        Get first task from queue of given type.
        """
        assert self.task_queues[task_type].qsize() > 0
        return self._get_task(task_type)

    def is_legal(self, task_type: Optional[int]) -> bool:
        if task_type is None or task_type < 0 or task_type >= len(self.task_queues):
            return False
        return self.task_queues[task_type].qsize() > 0

    def get_first_legal(self):
        for i in range(len(self.task_queues)):
            if self.task_queues[i].qsize() > 0:
                return i
        return None

    def empty(self) -> bool:
        """
        Return True if there are no tasks in the pool, otherwise False.
        """
        for q in self.task_queues:
            if q.qsize() > 0:
                return False
        return True

    def get_human_readable(self, current_time=0) -> str:
        """
        Return string representing the pool state in a human-readable form
        """
        result = "Tasks in pool:\n"
        for i in range(len(self.task_queues)):
            q = self.task_queues[i]
            result += f"\tTask {i}: {q.qsize()} "
            if q.qsize() > 0:
                result += f"(Queued for: {current_time - q.queue[0].first_seen})"
            result += "\n"
        return result

    # def add_illegal_action(self):
    #     self.illegal_actions += 1
