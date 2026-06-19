from .base_trigger import BaseTrigger
from models.growth import next_size
import numpy as np


class IDTrigger(BaseTrigger):
    """
    Adaptive growth trigger based on intrinsic-dimensionality convergence (v4).

    Grows the bottleneck when the smoothed latent ID stops rising — i.e. the
    current stage has converged. Uses an absolute per-epoch delta on a
    moving-averaged ID signal, with a patience counter to ride out noise, a
    min-epochs floor to skip the post-growth transient, and a max-epochs
    failsafe so a non-plateauing stage still grows in time to reach max_dim.

    Operating point (committed "fast" config, CIFAR-10 5-seed):
        ordering 0.814 ± 0.022, reaches dim=128 ~epoch 42, LP 0.4146.
    """

    def __init__(self, start_dim, max_dim, growth_rate=1.7,
                 delta_threshold=0.3, patience=3,
                 min_epochs_per_stage=3, smooth_window=3,
                 max_epochs_per_stage=11):
        super().__init__(start_dim, max_dim, growth_rate, min_epochs_per_stage)
        self.delta_threshold = delta_threshold
        self.patience = patience
        self.smooth_window = smooth_window
        self.max_epochs_per_stage = max_epochs_per_stage
        self.id_history = []     # raw ID per epoch, reset each stage
        self.bad_epochs = 0      # consecutive sub-threshold epochs

    def should_grow(self, metrics):
        idv = metrics.get("intrinsic_dim")
        if idv is not None and not np.isnan(idv):
            self.id_history.append(idv)

        # failsafe: force growth if the stage has run too long
        if self.can_grow() and self.epochs_since_growth >= self.max_epochs_per_stage:
            return True

        # skip the post-growth transient
        if self.epochs_since_growth < self.min_epochs_per_stage:
            return False

        # need enough history to compute a smoothed delta
        if len(self.id_history) <= self.smooth_window:
            return False

        prev = sum(self.id_history[-self.smooth_window - 1:-1]) / self.smooth_window
        curr = sum(self.id_history[-self.smooth_window:]) / self.smooth_window
        delta = curr - prev

        if delta < self.delta_threshold:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0

        return self.can_grow() and self.bad_epochs >= self.patience

    def next_dim(self, metrics):
        self.id_history = []        # reset signal for the new stage
        self.bad_epochs = 0
        self.epochs_since_growth = 0
        return self.grow_to(next_size(self.current_dim, self.max_dim, self.growth_rate))