from sklearn import metrics

from .base_trigger import BaseTrigger
from models.growth import next_size
import numpy as np


class IDTrigger(BaseTrigger):
    def __init__(self, start_dim, max_dim, growth_rate=1.7,
                 delta_threshold=0.3, patience=3,
                 min_epochs_per_stage=3, smooth_window=3, max_epochs_per_stage=11):
        print(f"IDTrigger: patience={patience}, min_eps={min_epochs_per_stage}, max_eps={getattr(self, 'max_epochs_per_stage', 'N/A')}")
        super().__init__(start_dim, max_dim, growth_rate, min_epochs_per_stage)
        self.delta_threshold = delta_threshold
        self.patience = patience
        self.smooth_window = smooth_window
        self.max_epochs_per_stage = max_epochs_per_stage
        self.id_history = []      # raw ID per epoch this stage
        self.bad_epochs = 0

    def _smoothed_id(self):
        w = self.id_history[-self.smooth_window:]
        return sum(w) / len(w)

    def should_grow(self, metrics):
        id_val = metrics.get("intrinsic_dim")
        if id_val is not None and not np.isnan(id_val):
            self.id_history.append(id_val)

        # failsafe first
        if self.can_grow() and self.epochs_since_growth >= self.max_epochs_per_stage:
            return True

        # skip the transient: don't even measure until min_epochs passed
        if self.epochs_since_growth < self.min_epochs_per_stage:
            return False

        if len(self.id_history) <= self.smooth_window:
            return False
        prev = sum(self.id_history[-self.smooth_window-1:-1]) / self.smooth_window
        curr = sum(self.id_history[-self.smooth_window:]) / self.smooth_window
        delta = curr - prev
        if delta < self.delta_threshold:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0
        return self.can_grow() and self.bad_epochs >= self.patience

    def next_dim(self, metrics):
        self.bad_epochs = 0
        self.id_history = []                      # reset signal for new stage
        return self.grow_to(next_size(self.current_dim, self.max_dim, self.growth_rate))