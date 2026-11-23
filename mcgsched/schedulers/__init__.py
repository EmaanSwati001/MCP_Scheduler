from .base import Placement, Scheduler
from .fcfs import FCFSFirstFit

# Register new policies here so run.py and the dashboard can find them by name.
REGISTRY = {
    FCFSFirstFit.name: FCFSFirstFit,
}


def get_scheduler(name: str, **kwargs) -> Scheduler:
    try:
        return REGISTRY[name](**kwargs)
    except KeyError:
        raise ValueError(
            f"unknown scheduler '{name}'. Available: {sorted(REGISTRY)}"
        ) from None


__all__ = ["Placement", "Scheduler", "FCFSFirstFit", "REGISTRY", "get_scheduler"]
