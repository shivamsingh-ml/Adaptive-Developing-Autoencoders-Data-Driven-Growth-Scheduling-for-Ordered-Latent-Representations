from .fixed_trigger import FixedTrigger
from .loss_plateau_trigger import LossPlateauTrigger
from .id_trigger import IDTrigger
from .gv_trigger import GVTrigger
from .rcg_trigger import RCGTrigger
from .programmatic_trigger import ProgrammaticTrigger

def build_trigger(cfg):
    trigger_type = cfg["type"]

    if trigger_type == "fixed":
        return FixedTrigger(schedule=cfg["schedule"])

    if trigger_type == "loss_plateau":
        return LossPlateauTrigger(**{k: v for k, v in cfg.items() if k != "type"})

    if trigger_type == "id":
        return IDTrigger(**{k: v for k, v in cfg.items() if k != "type"})

    if trigger_type == "gv":
        return GVTrigger(**{k: v for k, v in cfg.items() if k != "type"})

    if trigger_type == "rcg":
        return RCGTrigger(**{k: v for k, v in cfg.items() if k != "type"})

    if trigger_type == "programmatic":
        return ProgrammaticTrigger(**{k: v for k, v in cfg.items() if k != "type"})

    raise ValueError(f"Unknown trigger type: {trigger_type}")