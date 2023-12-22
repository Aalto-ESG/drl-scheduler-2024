from dataclasses import dataclass, field
from .task_tree import Task


@dataclass(order=True)
class QueuedTask:
    event_time: float
    first_seen: float
    task: Task = field(compare=False)


@dataclass(order=True)
class PooledTask:
    first_seen: float
    task: Task = field(compare=False)


@dataclass(order=True)
class ArchivedTask:
    time_completed: float
    first_seen: float
    task: Task = field(compare=False)
