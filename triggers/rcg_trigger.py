from .base_trigger import BaseTrigger
from models.growth import next_size


class RCGTrigger(BaseTrigger):
    def __init__(
        self,
        start_dim,
        max_dim,
        growth_rate=1.7,
        gap_threshold=0.05,
        min_epochs_per_stage=3,
        patience=2,
    ):
        super().__init__(start_dim, max_dim, growth_rate, min_epochs_per_stage)
        self.gap_threshold = gap_threshold
        self.patience = patience
        self.bad_epochs = 0

    def should_grow(self, metrics: dict) -> bool:
        gap = metrics.get("reconstruction_classification_gap")

        if gap is None:
            return False

        if gap > self.gap_threshold:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0

        return self.can_grow() and self.bad_epochs >= self.patience

    def next_dim(self, metrics: dict) -> int:
        return self.grow_to(next_size(self.current_dim, self.max_dim, self.growth_rate))