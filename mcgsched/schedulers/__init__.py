from .base import Placement, Scheduler
from .baselines import FGD, Consolidate, MASTLike, RandomPlacement
from .fcfs import FCFSFirstFit
from .mcg import MCGNoQueue, MCGNoWindow, MCGPlain, MCGSched
from .mcg_fair import MCGAging, MCGFairFull, MCGGuard, MCGNorm
from .mcg_hetero import MCGHetero, MCGHeteroNorm  # N2: capacity-aware fragmentation

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
    MCGAging.name: MCGAging,
    MCGGuard.name: MCGGuard,
    MCGFairFull.name: MCGFairFull,
    MCGNorm.name: MCGNorm,
    MCGHetero.name: MCGHetero,       # N2: capacity-aware F_C
    MCGHeteroNorm.name: MCGHeteroNorm,  # N2: capacity-aware F_C + renorm
}


def get_scheduler(name: str, **kwargs) -> Scheduler:
    try:
        return REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"unknown scheduler '{name}'. Available: {sorted(REGISTRY)}"
        ) from None


__all__ = ["Placement", "Scheduler", "FCFSFirstFit", "REGISTRY", "get_scheduler"]