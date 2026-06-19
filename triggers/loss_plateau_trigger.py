import math
from .base_trigger import BaseTrigger
from models.growth import next_size


class LossPlateauTrigger(BaseTrigger):
    def __init__(self, start_dim, max_dim, growth_rate=1.7,
                 patience=5, min_delta=0.001, min_epochs_per_stage=6):
        super().__init__(start_dim, max_dim, growth_rate, min_epochs_per_stage)
        self.patience = patience
        self.min_delta = min_delta   # relative threshold: 0.1% improvement rate
        self.best_loss = math.inf
        self.bad_epochs = 0

    def should_grow(self, metrics):
        loss = metrics.get("val_loss")
        if loss is None:
            return False

        if self.best_loss == math.inf:
            self.best_loss = loss
            return False

        # Relative improvement rate: how much did loss improve as % of best
        improvement_rate = (self.best_loss - loss) / self.best_loss

        if improvement_rate > self.min_delta:
            # Still improving meaningfully — reset patience
            self.best_loss = loss
            self.bad_epochs = 0
        else:
            # Not improving enough — count toward patience
            self.bad_epochs += 1
            if loss < self.best_loss:
                self.best_loss = loss  # still update best even if below threshold

        return self.can_grow() and self.bad_epochs >= self.patience

    def next_dim(self, metrics):
        self.bad_epochs = 0
        self.best_loss = math.inf   # reset for next stage
        return self.grow_to(next_size(self.current_dim, self.max_dim, self.growth_rate))