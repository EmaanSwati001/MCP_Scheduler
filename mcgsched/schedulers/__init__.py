from .base import Placement, Scheduler
from .baselines import FGD, Consolidate, MASTLike, RandomPlacement
from .fcfs import FCFSFirstFit
from .mcg import MCGNoQueue, MCGNoWindow, MCGPlain, MCGSched

# Register new policies here so run.py and the dashboard can find them by name.
REGISTRY = {
    FCFSFirstFit.name: FCFSFirstFit,
    RandomPlacement.name: RandomPlacement,
    Consolidate.name: Consolidate,
    FGD.name: FGD,
    MASTLike.name: MASTLike,
    MCGSched.name: MCGSched,
    MCGNoWindow.name: MCGNoWindow,
    MCGNoQueue.name: MCGNoQueue,
    MCGPlain.name: MCGPlain,
}


def get_scheduler(name: str, **kwargs) -> Scheduler:
    try:
        return REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"unknown scheduler '{name}'. Available: {sorted(REGISTRY)}"
        ) from None


__all__ = ["Placement", "Scheduler", "FCFSFirstFit", "REGISTRY", "get_scheduler"]