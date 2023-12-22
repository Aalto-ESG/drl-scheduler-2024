import typing
from queue import Queue

import matplotlib.pyplot as plt
import numpy as np

from src.conf.config_classes import SchedulerConfig
from .queued_task import ArchivedTask
from .task_tree import Task, TaskTree


class TaskHistory:
    """
    Holds historical data about recent tasks.
    Used as input to the scheduler.
    """

    def __init__(self, task_tree: TaskTree, cfg: SchedulerConfig):
        # history_length
        self.full_history: "dict[Task, list[ArchivedTask]]" = {}
        self.cfg = cfg

        self.history_length = cfg.sim.history_length
        self.partial_history: "Queue[ArchivedTask]" = Queue(maxsize=self.history_length)
        self.partial_history_lat_sum: int = 0
        # self.number_of_tasks = task_tree.get_number_of_tasks()
        self.task_tree = task_tree
        self.hstate_num_tasks = cfg.hstate.num_tasks
        self.p3_estimate = 0
        self.p3_actual = 0
        self.tasks_encountered = {}
        self.tasks_executed = {}
        for task in task_tree.all_tasks:
            name = task.name
            self.tasks_encountered[name] = 0
            self.tasks_executed[name] = 0
            if name == "D":
                self.p3_actual = task.probabilities[0]

    def add_encountered_task(self, task: Task):
        name = task.name
        if name in self.tasks_encountered:
            self.tasks_encountered[task.name] += 1

    def add_executed_task(self, task: Task):
        name = task.name
        if name in self.tasks_executed:
            self.tasks_executed[task.name] += 1

    def get_p3_estimate(self):
        # Add +1 to avoid div by zero - it should vanish soon enough as tasks keep piling up
        return self.tasks_encountered["E"] / (self.tasks_executed["D"] + 1)

    def get_p3_actual(self):
        return self.p3_actual

    def add_completed_task(self, archived_task: ArchivedTask):
        """
        Archive given task in the history.
        """
        task = archived_task.task
        if task not in self.full_history:
            self.full_history[task] = []
        self.full_history[task].append(archived_task)
        if self.partial_history.qsize() == self.history_length:
            old_task = self.partial_history.get()
            self.partial_history_lat_sum -= old_task.time_completed - old_task.first_seen
        self.partial_history_lat_sum += archived_task.time_completed - archived_task.first_seen
        self.partial_history.put(archived_task)

    def get_state(self, normalized=True) -> np.array:
        """
        Get state of the history as an array, that can be used as a neural network input.
        """
        # TODO: Implement more state representations
        if self.hstate_num_tasks:
            keys = list(self.full_history.keys())
            state = np.zeros(self.task_tree.get_number_of_leaf_nodes())
            # Count occurrences of each task
            # -> This is heavily dependent on the scheduling policy
            # -> Does this represent probabilities and execution times?
            for item in self.partial_history.queue:
                index = keys.index(item.task)
                state[index] += 1
            if normalized:
                # Normalize to [0, 1]
                state = [x / self.history_length for x in state]
            return state
        else:
            return np.zeros(1)  # TODO: This adds an useless 0 in state, but returning np.empty() sometimes becomes infinite

    def plot_distributions(self, max_steps: int):
        """
        Plot task distributions for the last max_steps.
        """
        import pandas as pd
        import seaborn as sns
        keys = []
        latencies = []
        times_completed = []
        for key in self.full_history.keys():
            for at in self.full_history[key]:
                keys.append(key)
                latencies.append(at.time_completed - at.first_seen)
                times_completed.append(at.time_completed)
        df = pd.DataFrame({"sequence": keys, "latency": latencies, "completion_time": times_completed})
        # sns.displot(data=df, x="latency", hue="sequence", stat="probability")
        sns.ecdfplot(data=df, x="latency", hue="sequence")
        return None

    def get_end_to_end_latencies(self):
        all_latencies = []
        # TODO: Below is a very hackish way to get latencies sorted by task name - should be refactored, but works
        key_task_pairs = {str(task): task for task in list(self.full_history.keys())}
        keys = sorted(list(key_task_pairs.keys()))
        for index, key in enumerate(keys):
            # key = keys[index]
            archived_tasks = self.full_history[key_task_pairs[key]]
            # index = task_tree.get_task_index(key)
            # print(f"{index}: {key}")
            row = []
            for task in archived_tasks:
                latency = task.time_completed - task.first_seen
                row.append(latency)
            all_latencies.append(row)
        return all_latencies

    def get_counts(self):
        counts = []
        for index, (key, archived_tasks) in enumerate(self.full_history.items()):
            c = len(archived_tasks)
            counts.append(c)
        return counts




    def plot_time_line(self):
        """
        Plot task execution timeline.
        """
        # TODO: Plot gant-style chart
        pass




