from .base_trigger import BaseTrigger


class FixedTrigger(BaseTrigger):
    def __init__(self, schedule, start_dim=None, max_dim=None):
        start_dim = start_dim if start_dim is not None else schedule[0]
        max_dim   = max_dim   if max_dim   is not None else max(schedule)
        super().__init__(start_dim=start_dim, max_dim=max_dim)
        self.schedule = schedule
        self.epoch = 0   # always equals training epoch, incremented only by step_epoch

    def step_epoch(self):
        self.epoch += 1                    # sole place self.epoch is incremented
        self.epochs_since_growth += 1

    def should_grow(self, metrics: dict) -> bool:
        if self.epoch == 0:
            return False                   # no previous epoch to compare against
        if self.epoch >= len(self.schedule):
            return False
        return self.schedule[self.epoch] != self.schedule[self.epoch - 1]

    def next_dim(self, metrics: dict) -> int:
        # self.epoch already points to current training epoch (set by step_epoch)
        # schedule[self.epoch] is the new dimension for this epoch
        self.current_dim = self.schedule[self.epoch]
        self.epochs_since_growth = 0
        return self.current_dim