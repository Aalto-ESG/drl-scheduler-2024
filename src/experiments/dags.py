import os

import hydra
from hydra.core.config_store import ConfigStore

from src.conf.config_classes import SchedulerConfig
from src.simulator.task_tree import TaskTree, Task
from src.simulator.tile_simulator import TileSimulator


class OldConfig:
    models_path = "../../models/"
    tb_logs_path = "../../logs/tb_logs/"

def get_env_from_settings(cfg: SchedulerConfig) -> TileSimulator:
    dag_name = cfg.sim.dag
    dag_name = dag_name.lower().strip()
    if dag_name == "a":
        return get_env_A(cfg)
    elif dag_name == "b":
        return get_env_B(cfg)
    elif dag_name == "a2":
        return get_env_A2(cfg)
    elif dag_name == "b2":
        return get_env_B2(cfg)
    elif dag_name == "c":
        return get_env_C(cfg)
    else:
        raise f"Requested dag in config: {dag_name}"

def get_env_A(cfg) -> TileSimulator:
    dag = TaskTree()
    "A,60,0.5B,0.6C"
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
    dag.all_tasks = [A, B, C, D]
    dag.initial_node = A

    # dag.print_node()
    # dag.print_node_combinations()
    # print(dag.get_avg_execution_time())

    env = TileSimulator(cfg, dag)
    env.generate_initial_tasks()
    return env

def get_env_A2(cfg) -> TileSimulator:
    dag = TaskTree()
    "A,60,0.5B,0.6C"
    leafA = Task("Leaf-A", 0)
    leafAB = Task("Leaf-AB", 0)
    leafABC = Task("Leaf-ABC", 0)
    leafABCD = Task("Leaf-ABCD", 0)
    A = Task("A", 56.75)
    B = Task("B", 30)
    C = Task("C", 20)
    D = Task("D", 10).add_task(leafABCD, 1)
    A.add_tasks([B, leafA], [0.9, 0.1])
    B.add_tasks([C, leafAB], [0.9, 0.1])
    C.add_tasks([D, leafABC], [0.9, 0.1])
    dag.all_tasks = [A, B, C, D]
    dag.initial_node = A

    # dag.print_node()
    # dag.print_node_combinations()
    # print(dag.get_avg_execution_time())

    env = TileSimulator(cfg, dag)
    env.generate_initial_tasks()
    return env


def get_env_B(cfg) -> TileSimulator:
    dag = TaskTree()
    leafA = Task("Leaf-A", 0)
    leafAB = Task("Leaf-AB", 0)
    leafABC = Task("Leaf-ABC", 0)
    leafABCD = Task("Leaf-ABCD", 0)
    A = Task("A", 10.5)
    B = Task("B", 20*2)
    C = Task("C", 30*3)
    D = Task("D", 50*4).add_task(leafABCD, 1)
    # A = Task("A", 30*3)
    # B = Task("B", 50*5)
    # C = Task("C", 20*2)
    # D = Task("D", 50.5*6).add_task(leafABCD, 1)
    A.add_tasks([B, leafA], [0.5, 0.5])
    B.add_tasks([C, leafAB], [0.5, 0.5])
    C.add_tasks([D, leafABC], [0.5, 0.5])
    dag.all_tasks = [A, B, C, D]
    dag.initial_node = A

    # dag.print_node()
    # dag.print_node_combinations()
    # print(dag.get_avg_execution_time())

    env = TileSimulator(cfg, dag)
    env.generate_initial_tasks()
    return env

def get_env_B2(cfg) -> TileSimulator:
    dag = TaskTree()
    leafA = Task("Leaf-A", 0)
    leafAB = Task("Leaf-AB", 0)
    leafABC = Task("Leaf-ABC", 0)
    leafABCD = Task("Leaf-ABCD", 0)
    A = Task("A", 10.5)
    B = Task("B", 20*2)
    C = Task("C", 30*3)
    D = Task("D", 50*4).add_task(leafABCD, 1)
    # A = Task("A", 30*3)
    # B = Task("B", 50*5)
    # C = Task("C", 20*2)
    # D = Task("D", 50.5*6).add_task(leafABCD, 1)
    A.add_tasks([B, leafA], [0.9, 0.1])
    B.add_tasks([C, leafAB], [0.9, 0.1])
    C.add_tasks([D, leafABC], [0.9, 0.1])
    dag.all_tasks = [A, B, C, D]
    dag.initial_node = A

    # dag.print_node()
    # dag.print_node_combinations()
    # print(dag.get_avg_execution_time())

    env = TileSimulator(cfg, dag)
    env.generate_initial_tasks()
    return env

def get_env_C(cfg: SchedulerConfig) -> TileSimulator:
    dag = TaskTree()
    budget = 40
    leafABC = Task("Leaf-ABC", 0)
    leafABDE = Task("Leaf-ABDE", 0)
    leafABDF = Task("Leaf-ABDF", 0)
    A = Task("A", 5)
    B = Task("B", 3)
    budget -= (5+3)
    C = Task("C", (budget/0.9)).add_task(leafABC, 1)
    D = Task("D", (budget/0.1/2))
    budget = budget*0.1/2
    E = Task("E", 60).add_task(leafABDE, 1)
    F = Task("F", 800).add_task(leafABDF, 1)
    A.add_tasks([B], [1.0])
    B.add_tasks([C, D], [0.9, 0.1])
    # D.add_tasks([E, F], [0.7, 0.3])
    D.add_tasks([E, F], [0.00001, 0.000001])
    dag.all_tasks = [A, B, C, D, E, F]
    dag.initial_node = A

    # dag.print_node()
    # dag.print_node_combinations()
    # print(dag.get_avg_execution_time())

    env = TileSimulator(cfg, dag)
    env.generate_initial_tasks()
    return env

def get_custom_env(cfg: SchedulerConfig, probs) -> TileSimulator:
    dag = TaskTree()
    # leafA = Task("Leaf-A", 0)
    leafABC = Task("Leaf-ABC", 0)
    leafABDE = Task("Leaf-ABDE", 0)
    leafABDF = Task("Leaf-ABDF", 0)
    A = Task("A", 10)
    B = Task("B", 5)
    C = Task("C", 40).add_task(leafABC, 1)
    D = Task("D", 40)
    E = Task("E", 30).add_task(leafABDE, 1)
    F = Task("F", 200).add_task(leafABDF, 1)
    # p1 = probs["p1"]
    p2 = probs["p2"]
    p3 = probs["p3"]
    # A.add_tasks([leafA, B], [p1, 1-p1])
    A.add_tasks([B], [1])
    B.add_tasks([C, D], [p2, 1-p2])
    D.add_tasks([E, F], [p3, 1-p3])
    dag.all_tasks = [A, B, C, D, E, F]
    dag.initial_node = A

    # dag.print_node()
    # dag.print_node_combinations()
    # print(dag.get_avg_execution_time())

    env = TileSimulator(cfg, dag)
    env.generate_initial_tasks()
    return env

def get_custom_env2(cfg: SchedulerConfig, probs) -> TileSimulator:
    dag = TaskTree()
    # leafA = Task("Leaf-A", 0)
    leafABC = Task("Leaf-ABC", 0)
    leafABDE = Task("Leaf-ABDE", 0)
    leafABDF = Task("Leaf-ABDF", 0)
    A = Task("A", probs["A"])
    B = Task("B", probs["B"])
    C = Task("C", probs["C"]).add_task(leafABC, 1)
    D = Task("D", probs["D"])
    E = Task("E", probs["E"]).add_task(leafABDE, 1)
    F = Task("F", probs["F"]).add_task(leafABDF, 1)
    # p1 = probs["p1"]
    p2 = probs["p2"]
    p3 = probs["p3"]
    # A.add_tasks([leafA, B], [p1, 1-p1])
    A.add_tasks([B], [1])
    B.add_tasks([C, D], [p2, 1-p2])
    D.add_tasks([E, F], [p3, 1-p3])
    dag.all_tasks = [A, B, C, D, E, F]
    dag.initial_node = A
    dag.p3_node = D  # TODO: Generalize for using other nodes

    # dag.print_node()
    # dag.print_node_combinations()
    # print(dag.get_avg_execution_time())

    env = TileSimulator(cfg, dag)
    env.generate_initial_tasks()
    return env

def get_custom_env3(cfg: SchedulerConfig, probs) -> TileSimulator:
    dag = TaskTree()
    # p1 = probs["p1"]
    p2 = probs["p2"]
    p3 = probs["p3"]
    budget = cfg.sim.interval
    # leafA = Task("Leaf-A", 0)
    leafABC = Task("Leaf-ABC", 0)
    leafABDE = Task("Leaf-ABDE", 0)
    leafABDF = Task("Leaf-ABDF", 0)

    A = Task("A", 10)
    B = Task("B", 5)
    budget -= 15
    C = Task("C", budget/p2).add_task(leafABC, 1)
    budget -= budget * p2
    D = Task("D", 40)
    E = Task("E", 30).add_task(leafABDE, 1)
    F = Task("F", 200).add_task(leafABDF, 1)

    # A.add_tasks([leafA, B], [p1, 1-p1])
    A.add_tasks([B], [1])
    B.add_tasks([C, D], [p2, 1-p2])
    D.add_tasks([E, F], [p3, 1-p3])
    dag.all_tasks = [A, B, C, D, E, F]
    dag.initial_node = A

    # dag.print_node()
    # dag.print_node_combinations()
    # print(dag.get_avg_execution_time())

    env = TileSimulator(cfg, dag)
    env.generate_initial_tasks()
    return env

cs = ConfigStore.instance()
cs.store(name="sim_conf", node=SchedulerConfig)
@hydra.main(version_base="1.2", config_path="../conf", config_name="sim_conf")
def main(cfg: SchedulerConfig):
    env = get_env_from_settings(cfg)
    print(f"Printing dag {cfg.sim.dag}")
    print("")
    env.task_tree.print_node()
    env.task_tree.print_node_combinations()
    print(env.task_tree.get_avg_execution_time())

if __name__ == "__main__":
    main()

