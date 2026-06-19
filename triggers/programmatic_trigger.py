"""
programmatic_trigger.py

Approach A for the Week-2 ablation: derive a per-epoch growth schedule
from two knobs — growth_rate and epochs_per_stage — and force it to
reach max_dim by the final epoch so that final capacity is held constant
across all ablation configs (a controlled comparison).

Internally it builds a full schedule list and then behaves exactly like
FixedTrigger, so it reuses the same step/should_grow/next_dim contract.
"""

from .base_trigger import BaseTrigger


def generate_programmatic_schedule(start_dim, max_dim, growth_rate,
                                   epochs_per_stage, total_epochs):
    """
    Returns (schedule, reached_naturally).
    schedule: list[int] of length total_epochs (latent dim per epoch).
    reached_naturally: True if max_dim was hit without force-topping the tail.
    """
    dims = [start_dim]
    while dims[-1] < max_dim:
        nxt = min(round(dims[-1] * growth_rate), max_dim)
        if nxt == dims[-1]:
            nxt = min(dims[-1] + 1, max_dim)
        dims.append(nxt)

    n_stages = len(dims)
    epochs_needed = n_stages * epochs_per_stage
    reached_naturally = (dims[-1] == max_dim and epochs_needed <= total_epochs)

    schedule = []
    if epochs_needed <= total_epochs:
        for i, d in enumerate(dims):
            n = epochs_per_stage if i < len(dims) - 1 else (total_epochs - len(schedule))
            schedule.extend([d] * n)
    else:
        base = max(1, total_epochs // n_stages)
        for d in dims:
            schedule.extend([d] * base)
        schedule = schedule[:total_epochs]
        if schedule and schedule[-1] != max_dim:
            n_force = max(1, total_epochs // n_stages)
            schedule[-n_force:] = [max_dim] * n_force

    while len(schedule) < total_epochs:
        schedule.append(max_dim)
    schedule = schedule[:total_epochs]
    schedule[-1] = max_dim   # force-reach guarantee
    return schedule, reached_naturally


class ProgrammaticTrigger(BaseTrigger):
    def __init__(self, start_dim, max_dim, growth_rate,
                 epochs_per_stage, total_epochs):
        super().__init__(start_dim=start_dim, max_dim=max_dim,
                         growth_rate=growth_rate)
        self.schedule, self.reached_naturally = generate_programmatic_schedule(
            start_dim, max_dim, growth_rate, epochs_per_stage, total_epochs
        )
        self.epoch = 0
        self.current_dim = self.schedule[0]

    def step_epoch(self):
        self.epoch += 1
        self.epochs_since_growth += 1

    def should_grow(self, metrics):
        if self.epoch == 0 or self.epoch >= len(self.schedule):
            return False
        return self.schedule[self.epoch] != self.schedule[self.epoch - 1]

    def next_dim(self, metrics):
        self.current_dim = self.schedule[self.epoch]
        self.epochs_since_growth = 0
        return self.current_dim
