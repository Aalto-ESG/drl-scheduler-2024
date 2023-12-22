import random

import numpy as np
import copy


class Task:
    def __init__(self, name: str, processing_time: float):
        self.name = name
        self.processing_time = processing_time
        self.children = []
        self.probabilities = []
        self.weights = []
        self.deterministic = False
        self.deterministic_next = None

    def get_deterministic_copy(self):
        # Returns a node with pre-sampled children.
        #
        # Ensures that workload is identical when evaluating models.
        # Without determinism, the workload may vary depending on which task is chosen first.
        if self.is_leaf():
            return self
        copy_node = copy.copy(self)
        copy_node.deterministic_next = self.sample_next().get_deterministic_copy()
        copy_node.deterministic = True
        return copy_node

    def add_task(self, task, weight):
        # Add a task as a child
        self.children.append(task)
        self.weights.append(weight)
        self.recalculate_probabilities()
        return self  # Return self to allow chaining operations

    def add_tasks(self, tasks, weights):
        assert len(tasks) == len(weights)
        [self.children.append(task) for task in tasks]
        [self.weights.append(weight) for weight in weights]
        self.recalculate_probabilities()
        return self  # Return self to allow chaining operations

    def update_weights(self, new_weights: list):
        assert len(self.weights) == len(new_weights)
        self.weights = new_weights
        self.recalculate_probabilities()

    def recalculate_probabilities(self):
        weights_sum = np.sum(self.weights)
        self.probabilities = [w / weights_sum for w in self.weights]

    def is_leaf(self):
        return len(self.children) == 0

    def sample_next(self) -> "Task":
        if self.deterministic:
            return self.deterministic_next
        else:
            # return np.random.choice(self.children, p=self.probabilities)
            return random.choices(self.children, weights=self.probabilities, k=1)[0]

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Task):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)


class TaskTree:
    def __init__(self):
        self.initial_node: "Task" = None
        self.all_tasks = None
        self.expired_node: "Task" = Task("Expired", 0)
        self.p3_node: "Task" = None  # In experiments we are specifically randomizing p3 - TODO: Generalize me

    def randomize_p3(self):
        if self.p3_node is not None:
            p3 = random.random()
            self.p3_node.probabilities = [p3, 1-p3]

    def get_all_tasks(self):
        return self.all_tasks

    def get_task_index(self, task: Task):
        return self.all_tasks.index(task)

    def get_number_of_tasks(self):
        return len(self.all_tasks)

    def get_number_of_leaf_nodes(self):
        return len(self.all_tasks) + 1  # +1 for expired tasks

    def generate_example_tree(self):
        # TODO: Build leaf-nodes automatically (i.e., implement a smarter tree structure)
        leafA = Task("Leaf-A", 0)
        leafAB = Task("Leaf-AB", 0)
        leafABC = Task("Leaf-ABC", 0)
        leafABCD = Task("Leaf-ABCD", 0)
        A = Task("A", 56.75)
        B = Task("B", 30)
        C = Task("C", 20)
        D = Task("D", 10).add_task(leafABCD, 1)
        A.add_tasks([B, leafA], [0.5, 0.5])
        B.add_tasks([C, leafAB], [0.5, 0.5])
        C.add_tasks([D, leafABC], [0.5, 0.5])
        self.all_tasks = [A, B, C, D]
        self.initial_node = A

    def print_node(self, node: Task=None, depth=0):
        if node is None:
            node = self.initial_node
        if depth == 0:
            print(f"{node} 100%")
        for i in range(len(node.children)):
            n = node.children[i]
            p = node.probabilities[i]
            print(f"{'-'*(depth+1)} {n} (t: {n.processing_time}, p: {p*100}%)")
            self.print_node(n, depth+1)

    def print_node_combinations(self, node: Task=None, nodes=None, total_time=0, total_p=1):
        if node is None:
            node = self.initial_node
        if nodes is None:
            nodes = []
        if node.is_leaf():
            print(f"{nodes}, p: {total_p:.4f}, total_time: {total_time}")
        nodes.append(node)
        total_time += node.processing_time
        for i in range(len(node.children)):
            next_node = node.children[i]
            next_p = total_p * node.probabilities[i]
            self.print_node_combinations(next_node, nodes.copy(), total_time, next_p)

    def get_avg_execution_time(self, node: Task = None):
        if node is None:
            node = self.initial_node
        exec_time = 0
        for i in range(len(node.children)):
            t = self.get_avg_execution_time(node.children[i])
            p = node.probabilities[i]
            exec_time += t*p
        return exec_time + node.processing_time



    def __str__(self):
        pass



