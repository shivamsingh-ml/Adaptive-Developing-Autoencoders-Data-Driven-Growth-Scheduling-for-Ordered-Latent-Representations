from abc import ABC, abstractmethod


class BaseTrigger(ABC):
    def __init__(self, start_dim, max_dim, growth_rate=1.7, min_epochs_per_stage=1):
        self.current_dim = start_dim
        self.max_dim = max_dim
        self.growth_rate = growth_rate
        self.min_epochs_per_stage = min_epochs_per_stage
        self.epochs_since_growth = 0

    def step_epoch(self):
        self.epochs_since_growth += 1

    def can_grow(self):
        return (
            self.current_dim < self.max_dim
            and self.epochs_since_growth >= self.min_epochs_per_stage
        )

    def grow_to(self, new_dim):
        self.current_dim = min(new_dim, self.max_dim)
        self.epochs_since_growth = 0
        return self.current_dim

    @abstractmethod
    def should_grow(self, metrics: dict) -> bool:
        pass

    @abstractmethod
    def next_dim(self, metrics: dict) -> int:
        pass