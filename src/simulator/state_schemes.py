from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Avoid circular import errors, but enable type hinting
    from tile_simulator import TileSimulator

import numpy as np

class StateSchemes:

    @staticmethod
    def get_state(state: TileSimulator, state_scheme):
        state_schemes = {
            0: StateSchemes.time_one_hot_cache,
            1: StateSchemes.time_no_cache,
        }
        if state_scheme in state_schemes:
            return state_schemes[state_scheme](state)
        else:
            assert "Invalid state scheme!"
    
    @staticmethod
    def get_state_representation(taskpool, current_time, normalized, state_scheme_representation):
        state_schemes = {
            0: StateSchemes.get_state_seperate,
            1: StateSchemes.get_state_upperbound_latency,
        }
        if state_scheme_representation in state_schemes:
            return state_schemes[state_scheme_representation](taskpool, current_time, normalized)
        else:
            assert "Invalid state scheme!"


    @staticmethod
    def time_one_hot_cache(state: TileSimulator):
        """
        Creates an observation vector from the current state of the simulation.

        The observation can be used as an input to a policy.
        :return:
        """

        # Get one-hot vector of the previously executed task   (e.g., [0,0,1,0])
        one_hot_cache_vector = np.zeros(state.task_tree.get_number_of_tasks())
        if state.task_in_cache != None:
            one_hot_cache_vector[state.task_tree.get_task_index(state.task_in_cache)] = 1

        # Task pool (preferably scaled to [0,1])
        task_pool_observations = StateSchemes.get_state_representation(state.task_pool, state.current_time, True, state.cfg.sim.state_representation)

        # Task history (preferably scaled to [0,1])
        # TODO: Re-enable history, when it actually does something
        #task_history_observations = self.task_history.get_state()
        task_history_observations = []

        # Concat all observations together
        observations = np.concatenate([task_pool_observations,
                                       one_hot_cache_vector,
                                       task_history_observations])
        return observations

    @staticmethod
    def time_no_cache(state: TileSimulator):
        """
        Creates an observation vector from the current state of the simulation.

        The observation can be used as an input to a policy.
        :return:
        """

        # Get one-hot vector of the previously executed task   (e.g., [0,0,1,0])
        one_hot_cache_vector = np.zeros(state.task_tree.get_number_of_tasks())
        if state.task_in_cache != None:
            one_hot_cache_vector[state.task_tree.get_task_index(state.task_in_cache)] = 1

        # Task pool (preferably scaled to [0,1])
        task_pool_observations = StateSchemes.get_state_representation(state.task_pool, state.current_time, True, state.cfg.sim.state_representation)

        # Task history (preferably scaled to [0,1])
        # TODO: Re-enable history, when it actually does something
        #task_history_observations = self.task_history.get_state()
        task_history_observations = []

        # Concat all observations together
        observations = np.concatenate([task_pool_observations,
                                       task_history_observations])
        return observations
    
    @staticmethod
    def get_state_seperate(task_pool, current_time, normalized=False) -> np.array:
        """
        Get state of the pool as an array, that can be used as a neural network input.
        """
        task_amounts = [q.qsize() for q in task_pool.task_queues]
        task_waiting_times = [current_time - q.queue[0].first_seen if q.qsize() > 0 else 0 for q in task_pool.task_queues]
        if normalized:  # Normalize to values between [0, 1]
            max_tasks = task_pool.cfg.sim.streaming_tasks
            max_waiting_time = task_pool.cfg.sim.max_latency 
            task_amounts = [x / max_tasks for x in task_amounts]
            task_waiting_times = [t / max_waiting_time for t in task_waiting_times]
        return np.array(task_amounts + task_waiting_times)

    @staticmethod
    def get_state_upperbound_latency(task_pool, current_time, normalized=False) -> np.array:
        """
        Get state of the pool as an array, that can be used as a neural network input.
        """
        task_amounts = [q.qsize() for q in task_pool.task_queues]
        task_waiting_times = [current_time - q.queue[0].first_seen if q.qsize() > 0 else 0 for q in task_pool.task_queues]
        if normalized:  # Normalize to values between [0, 1]
            max_tasks = task_pool.cfg.sim.streaming_tasks
            max_waiting_time = task_pool.cfg.sim.max_latency 
            task_amounts = np.array([x / max_tasks for x in task_amounts])
            task_waiting_times = np.array([t / max_waiting_time for t in task_waiting_times])
        return task_amounts * task_waiting_times