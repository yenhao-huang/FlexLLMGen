"""Memory-budget search space for Hugging Face tensor placement."""

import dataclasses
from typing import Dict, List, Mapping, Sequence, Tuple, Union


GIB = 1 << 30


@dataclasses.dataclass(frozen=True)
class PlacementCandidate:
    """One reproducible Accelerate placement candidate."""

    name: str
    strategy: str
    gpu_memory_gib: Tuple[int, ...]
    cpu_memory_gib: int

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must not be empty")
        if self.strategy not in {"auto", "balanced", "balanced_low_0", "sequential", "cpu"}:
            raise ValueError("unsupported placement strategy: {}".format(self.strategy))
        if any(value < 0 for value in self.gpu_memory_gib):
            raise ValueError("GPU memory budgets must not be negative")
        if self.cpu_memory_gib < 1:
            raise ValueError("CPU memory budget must be positive")

    @property
    def max_memory(self) -> Mapping[Union[int, str], str]:
        result: Dict[Union[int, str], str] = {
            index: "{}GiB".format(value)
            for index, value in enumerate(self.gpu_memory_gib)
            if value > 0
        }
        result["cpu"] = "{}GiB".format(self.cpu_memory_gib)
        return result

    def estimated_weight_placement(self, model_bytes: int, bits: int = 8) -> Mapping[str, int]:
        """Estimate packed weight bytes by tier for cheap candidate pruning."""
        if model_bytes < 1:
            raise ValueError("model_bytes must be positive")
        if bits not in {4, 8, 16, 32}:
            raise ValueError("bits must be 4, 8, 16, or 32")
        packed_bytes = (model_bytes * bits + 15) // 16
        gpu_capacity = sum(self.gpu_memory_gib) * GIB
        gpu_bytes = min(packed_bytes, gpu_capacity)
        cpu_bytes = min(packed_bytes - gpu_bytes, self.cpu_memory_gib * GIB)
        return {
            "gpu": gpu_bytes,
            "cpu": cpu_bytes,
            "disk": packed_bytes - gpu_bytes - cpu_bytes,
        }

    def to_dict(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "strategy": self.strategy,
            "gpu_memory_gib": list(self.gpu_memory_gib),
            "cpu_memory_gib": self.cpu_memory_gib,
            "max_memory": {str(key): value for key, value in self.max_memory.items()},
        }


def generate_candidates(
    gpu_free_gib: Sequence[int],
    cpu_available_gib: int,
    gpu_reserve_gib: int = 2,
    cpu_reserve_gib: int = 8,
) -> List[PlacementCandidate]:
    """Generate a small ordered search space from currently free memory.

    Candidates deliberately use integer GiB budgets and fixed utilization
    levels so a result can be reproduced even when free-memory readings vary.
    """
    if not gpu_free_gib:
        raise ValueError("at least one GPU memory reading is required")
    if any(value < 1 for value in gpu_free_gib):
        raise ValueError("GPU free-memory readings must be positive")
    if cpu_available_gib < 1:
        raise ValueError("cpu_available_gib must be positive")
    if gpu_reserve_gib < 0 or cpu_reserve_gib < 0:
        raise ValueError("memory reserves must not be negative")

    usable_gpu = tuple(max(1, value - gpu_reserve_gib) for value in gpu_free_gib)
    usable_cpu = max(1, cpu_available_gib - cpu_reserve_gib)
    candidates: List[PlacementCandidate] = [
        PlacementCandidate("dram-baseline", "cpu", tuple(0 for _ in usable_gpu), usable_cpu)
    ]

    seen = set()
    for utilization in (50, 70, 85, 95):
        budgets = tuple(max(1, value * utilization // 100) for value in usable_gpu)
        if budgets in seen:
            continue
        seen.add(budgets)
        candidates.append(PlacementCandidate(
            "balanced-{}".format(utilization), "balanced", budgets, usable_cpu
        ))
        candidates.append(PlacementCandidate(
            "sequential-{}".format(utilization), "sequential", budgets, usable_cpu
        ))
    return candidates
