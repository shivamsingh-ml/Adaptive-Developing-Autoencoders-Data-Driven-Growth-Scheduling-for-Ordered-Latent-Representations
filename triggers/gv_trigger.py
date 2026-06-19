from .base_trigger import BaseTrigger
from models.growth import next_size


class GVTrigger(BaseTrigger):
    def __init__(
        self,
        start_dim,
        max_dim,
        growth_rate=1.7,
        relative_threshold=0.05,
        min_epochs_per_stage=3,
        max_epochs_per_stage=11,
        patience=3,
    ):
        super().__init__(start_dim, max_dim, growth_rate, min_epochs_per_stage)
        self.relative_threshold = relative_threshold
        self.reference_gradient_variance = None
        self.patience = patience
        self.bad_epochs = 0
        self.stage_max_gv = 0.0
        self.max_epochs_per_stage = max_epochs_per_stage
        self.min_epochs_per_stage = min_epochs_per_stage

    def should_grow(self, metrics):
        gv = metrics["gradient_variance"]
        self.stage_max_gv = max(self.stage_max_gv, gv)
        if self.stage_max_gv < 1e-8:          # still pre-learning noise floor
            return False
        if self.epochs_since_growth < self.min_epochs_per_stage:
            return False
        if gv < self.relative_threshold * self.stage_max_gv:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0
        return self.can_grow() and self.bad_epochs >= self.patience

    def next_dim(self, metrics):
        self.stage_max_gv = 0.0               # reset peak for new stage
        self.bad_epochs = 0
        return self.grow_to(next_size(self.current_dim, self.max_dim, self.growth_rate))