from .base_trigger import BaseTrigger
from models.growth import next_size


class GVTrigger(BaseTrigger):
    """
    Adaptive growth trigger based on bottleneck gradient-variance convergence.

    Within each stage the gradient variance of the bottleneck weights rises
    from a near-zero noise floor as the model starts learning, peaks while it
    learns actively, then decays as the stage converges. We therefore track
    the running stage MAXIMUM and grow once the gradient variance has decayed
    to `relative_threshold` of that peak (the convergence signal).

    A `peak_floor` ignores the tiny early-stage noise (~1e-13 at dim=6) so a
    noise blip is never mistaken for a learning peak; the decay test only
    activates once a genuine peak above the floor has been observed. A
    `max_epochs_per_stage` failsafe guarantees the model still reaches max_dim.
    """

    def __init__(self, start_dim, max_dim, growth_rate=1.7,
                 relative_threshold=0.4, patience=2,
                 peak_floor=1e-9, min_epochs_per_stage=3,
                 max_epochs_per_stage=11):
        super().__init__(start_dim, max_dim, growth_rate, min_epochs_per_stage)
        self.relative_threshold = relative_threshold
        self.patience = patience
        self.peak_floor = peak_floor
        self.max_epochs_per_stage = max_epochs_per_stage
        self.stage_max_gv = 0.0     # running peak grad-var this stage
        self.bad_epochs = 0

    def should_grow(self, metrics):
        gv = metrics.get("gradient_variance")
        if gv is None:
            return False

        # track running stage peak
        if gv > self.stage_max_gv:
            self.stage_max_gv = gv

        # failsafe: force growth if the stage has run too long
        if self.can_grow() and self.epochs_since_growth >= self.max_epochs_per_stage:
            return True

        # skip the post-growth transient
        if self.epochs_since_growth < self.min_epochs_per_stage:
            return False

        # only evaluate decay once a genuine peak (above noise floor) was seen
        if self.stage_max_gv < self.peak_floor:
            return False

        # fire on decay to relative_threshold of the stage peak
        if gv < self.relative_threshold * self.stage_max_gv:
            self.bad_epochs += 1
        else:
            self.bad_epochs = 0

        return self.can_grow() and self.bad_epochs >= self.patience

    def next_dim(self, metrics):
        self.stage_max_gv = 0.0
        self.bad_epochs = 0
        self.epochs_since_growth = 0
        return self.grow_to(next_size(self.current_dim, self.max_dim, self.growth_rate))