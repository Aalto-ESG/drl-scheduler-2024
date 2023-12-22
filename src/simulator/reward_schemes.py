from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Avoid circular import errors, but enable type hinting
    from tile_simulator import TileSimulator

import numpy as np


class RewardSchemes:

    @staticmethod
    def get_reward(state: TileSimulator, reward_scheme):
        reward_schemes = {
            0: RewardSchemes.reward_short_makespan,
            1: RewardSchemes.penalize_long_makespan,
            2: RewardSchemes.reduced_total_waiting_time,
            3: RewardSchemes.reduced_avg_waiting_time,
            4: RewardSchemes.penalize_total_waiting_time,
            5: RewardSchemes.penalize_avg_waiting_time,
            6: RewardSchemes.sparse_avg_latency,
            7: RewardSchemes.rolling_avg_latency,
            8: RewardSchemes.total_latency,
            9: RewardSchemes.jct,
            10: RewardSchemes.positive_jct,
            11: RewardSchemes.occupancy,

        }
        if reward_scheme in reward_schemes:
            return reward_schemes[reward_scheme](state)
        else:
            assert "Invalid reward scheme!"

    @staticmethod
    def normalize_to_max_latency(latency: float, state: TileSimulator):
        latency /= state.cfg_sim_max_latency
        if state.cfg.sim.enable_reward_clipping:
            latency = min(latency, 1)
            latency = max(latency, -1)
        return latency

    @staticmethod
    def reward_short_makespan(state: TileSimulator):
        """
        Hypothesis:
        - Prioritizes tasks that have been queued for a short time
        - Gets more penalty as avg waiting time rises towards the end
        - Initially learns to avoid executing tasks that have been waiting for a long time
            - But does this change when training for longer periods?
        """
        if not state.task_to_execute:
            return 0  # No tasks were executed this step
        reward = 0
        if state.next_task.is_leaf():
            latency = state.current_task_make_span_latency
            normalized = RewardSchemes.normalize_to_max_latency(latency, state)  # [0, 1]
            reward = 1 - normalized  # Reward low latency
        return reward * state.cfg.sim.reward_factor

    @staticmethod
    def penalize_long_makespan(state: TileSimulator):
        """
        Hypothesis:
        - Same as above (penalty instead of reward)
        - Learning-wise this should not change anything compared to above
        """
        # Give penalty based on how long the executed task was in the queue
        if not state.task_to_execute:
            return 0  # No tasks were executed this step
        reward = 0
        if state.next_task.is_leaf():
            reward = state.current_task_make_span_latency
            # reward = RewardSchemes.normalize_to_max_latency(reward, state)  # [0, 1]
            reward = -reward
        return reward * state.cfg.sim.reward_factor

    @staticmethod
    def reduced_total_waiting_time(state: TileSimulator):
        """
        Hypothesis:
        - Emulates Longest Job First
        - Gets higher rewards by allowing waiting time to pile up
            - Optimal is to idle for x seconds before starting to execute actions?
        - Good or bad?
            - Probably not good, emulates LJF and might learn malicious tactics
        """
        if not state.task_to_execute:
            return 0  # No tasks were executed this step
        reward = 0
        if state.next_task.is_leaf():
            latency = state.current_task_make_span_latency
            normalized = RewardSchemes.normalize_to_max_latency(latency, state)  # [0, 1]
            reward = normalized
        return reward * state.cfg.sim.reward_factor

    @staticmethod
    def reduced_avg_waiting_time(state: TileSimulator):
        """
        Hypothesis:
        - Similar to above BUT:
            - Reward is inverse scaled by the number of tasks in pool
            - Rewards get smaller as number of tasks piles up
            - Highest reward by having only one task in pool (and a long waiting time for this task)
        - Good or bad?
            - Encourages reducing amount of tasks and the waiting time
            - Maybe good?
        """
        if not state.task_to_execute:
            return 0  # No tasks were executed this step
        reward = 0
        if state.next_task.is_leaf():
            max_latency = state.cfg.sim.max_latency

            # Fetch current values
            total_time = state.task_pool.get_total_waiting_time(state.current_time)
            total_tasks = state.task_pool.get_total_tasks_in_queue()

            # Compute averages
            if total_tasks > 0:
                cur_avg_time = total_time / total_tasks
            else:
                cur_avg_time = 0
            prev_avg_time = (total_time + state.current_task_make_span_latency) / (total_tasks + 1)

            reward = prev_avg_time - cur_avg_time
            reward = RewardSchemes.normalize_to_max_latency(reward, state)  # [0, 1]
        return reward * state.cfg.sim.reward_factor

    @staticmethod
    def penalize_total_waiting_time(state: TileSimulator):
        """
        Hypothesis:
        - Penalties start piling up when too many tasks in queue
        - Small penalties at first
        - Large penalties later on
        - Good or bad?
            - No idea
        """
        if not state.task_to_execute:
            return 0  # No tasks were executed this step
        total_time = state.task_pool.get_total_waiting_time(state.current_time)
        normalized = RewardSchemes.normalize_to_max_latency(total_time, state)  # [0, 1]
        return -normalized * state.cfg.sim.reward_factor

    @staticmethod
    def penalize_avg_waiting_time(state: TileSimulator):
        """
        Hypothesis:
        - Same as above, BUT
            - Less extreme penalties when tasks are piling up
        - Good or bad?
            - No idea
        """
        if not state.task_to_execute:
            return 0  # No tasks were executed this step
        total_time = state.task_pool.get_total_waiting_time(state.current_time)
        total_tasks = state.task_pool.get_total_tasks_in_queue()
        if total_tasks > 0:
            avg_time = total_time / total_tasks
        else:
            avg_time = 0
        normalized = RewardSchemes.normalize_to_max_latency(avg_time, state)  # [0, 1]
        return -normalized * state.cfg.sim.reward_factor

    @staticmethod
    def sparse_avg_latency(state: TileSimulator):
        """
        Avg latency over the full experiment
        - Returns zero for everything else, except the final task

        Hypothesis:
        - Follows full experiment avg_latency perfectly
        - Gives very sparse rewards -> Can be hard to learn
        - Depends heavily on length of the experiment
        ---> Maybe a running average could be better?
        """
        if state.done:
            # Only give reward after full simulation is done
            all_latencies = state.task_history.get_end_to_end_latencies()
            avg_latency = np.array([latency for sublist in all_latencies for latency in sublist]).mean()
            normalized = RewardSchemes.normalize_to_max_latency(avg_latency, state)
            return -normalized * state.cfg.sim.reward_factor
        return 0






    @staticmethod
    # 
    def rolling_avg_latency(state: TileSimulator):
        """
        Rolling average over last n tasks

        Window length is configured in TaskHistory
        """
        # makespans = []
        # for task in state.task_history.partial_history.queue:
        #     # TODO: This causes serious amounts of repetitive computations - would be better to store these in TaskHistory
        #     makespan = task.time_completed - task.first_seen
        #     makespans.append(makespan)
        makespan_sum = 0
        num_tasks = state.task_history.partial_history.qsize()
        if state.done:
            tasks_counted = 0
            for i in range(state.task_history.history_length):
                if not state.task_history.partial_history.empty():
                    task = state.task_history.partial_history.get()
                    makespan_sum += (i+1) * (task.time_completed - task.first_seen)
                    tasks_counted += (i+1)
            avg_latency = makespan_sum / tasks_counted
            normalized = RewardSchemes.normalize_to_max_latency(avg_latency, state)
            return -normalized
        elif state.next_task is not None and state.next_task.is_leaf():
            makespan_sum = state.task_history.partial_history_lat_sum
            if num_tasks > 0:
                avg_latency = makespan_sum / num_tasks
                normalized = RewardSchemes.normalize_to_max_latency(avg_latency, state)
                return -normalized
                # normalized = RewardSchemes.normalize_to_max_latency(avg_latency, state)
                # normalized = avg_latency / state.cfg.sim.max_latency
                # return -normalized * state.cfg.sim.reward_factor
        return 0

    @staticmethod
    def total_latency(state: TileSimulator):
        """
        Avg latency over the full experiment
        - Returns zero for everything else, except the final task

        Hypothesis:
        - Follows full experiment avg_latency perfectly
        - Gives very sparse rewards -> Can be hard to learn
        - Depends heavily on length of the experiment
        ---> Maybe a running average could be better?
        """
        if state.done:
            # Only give reward after full simulation is done
            all_latencies = state.task_history.get_end_to_end_latencies()
            avg_latency = np.array([latency for sublist in all_latencies for latency in sublist]).sum()
            normalized = RewardSchemes.normalize_to_max_latency(avg_latency, state)
            return -normalized * state.cfg.sim.reward_factor
        return 0

    @staticmethod
    def jct(state: TileSimulator):
        """
        Reward as defined in DECIMA
        """
        # max_lat = state.cfg.sim.max_latency
        ms_to_s = 1/1000  # Scale down to avoid too large rewards
        scale = ms_to_s
        # scale = 1/max_lat
        reward = -(state.current_time - state.prev_time) * scale * state.task_pool.get_total_tasks_in_queue()
        return reward

    @staticmethod
    def positive_jct(state: TileSimulator):
        """
        Reward as defined in DECIMA, but positive to encourage not expiring tasks
        """
        # max_lat = state.cfg.sim.max_latency
        ms_to_s = 1/1000  # Scale down to avoid too large rewards
        scale = ms_to_s
        # scale = 1/max_lat
        reward = -(state.current_time - state.prev_time) * scale * state.task_pool.get_total_tasks_in_queue()
        return reward

    @staticmethod
    def occupancy(state: TileSimulator):
        """
        Reward for higher batch occupancy. Attempts to maximize resource utilization, and also throughput

        +1 if batch occupancy is maxed out
        0 if nothing is being executed
        """
        # max_lat = state.cfg.sim.max_latency
        ms_to_s = 1/1000  # Scale down to avoid too large rewards
        scale = ms_to_s
        # scale = 1/max_lat
        occupancy = sum(state.batched_per_exec) / sum(state.executors)

        reward = (state.current_time - state.prev_time) * scale * occupancy
        return reward

