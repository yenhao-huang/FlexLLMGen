import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "exp" / "gpt_sol5_6" / "optimize_placement.py"
SPEC = importlib.util.spec_from_file_location("sol_optimizer", SCRIPT)
sol_optimizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sol_optimizer)


def valid_proposal():
    return {
        "analysis_summary": "Test progressively larger GPU budgets.",
        "candidates": [{
            "name": "balanced-test",
            "strategy": "balanced",
            "gpu_memory_gib": [18, 17],
            "cpu_memory_gib": 32,
            "batch_size": 1,
            "hypothesis": "Keeping more decoder blocks on GPU reduces transfers.",
        }],
    }


def test_sol_proposal_is_validated_against_local_limits():
    result = sol_optimizer.validate_proposal(valid_proposal(), [24, 24], 48)
    assert result["candidates"][0]["gpu_memory_gib"] == [18, 17]


def test_sol_proposal_cannot_consume_reserved_gpu_memory():
    proposal = valid_proposal()
    proposal["candidates"][0]["gpu_memory_gib"] = [23, 17]
    with pytest.raises(ValueError, match="GPU memory limit"):
        sol_optimizer.validate_proposal(proposal, [24, 24], 48)


def test_sol_proposal_rejects_unexpected_fields():
    proposal = valid_proposal()
    proposal["candidates"][0]["untrusted"] = True
    with pytest.raises(ValueError, match="unexpected fields"):
        sol_optimizer.validate_proposal(proposal, [24, 24], 48)
