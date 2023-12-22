from ray.tune import Stopper


class MaxValueStopper(Stopper):
    def __init__(self, metric, max_value):
        self._metric = metric
        self._max_value = max_value

    def __call__(self, trial_id, result):
        return result[self._metric] >= self._max_value

    def stop_all(self):
        return False


class MinValueStopper(Stopper):
    def __init__(self, metric, min_value, stop_all):
        self._metric = metric
        self._min_value = min_value
        self._stop_all = stop_all
        self._should_stop = False

    def __call__(self, trial_id, result):
        should_stop = result[self._metric] <= self._min_value
        if should_stop:  # For protecting against switching to false before stop_all() called?
            self._should_stop = True
        return should_stop

    def stop_all(self):
        return self._stop_all and self._should_stop

