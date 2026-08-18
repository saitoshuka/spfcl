from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoints import load_weights_checkpoint, state_dict_sha256


@dataclass(frozen=True)
class ForkAudit:
    """Evidence that a child stage starts at the parent's exact model state."""

    parent_checkpoint: Path
    parent_state_sha256: str
    initial_state_sha256: str
    optimizer_state_entries: int
    scheduler_type: str | None


@dataclass
class StageFork:
    model: Any
    optimizer: Any
    scheduler: Any | None
    audit: ForkAudit


def assert_fresh_optimizer(optimizer: Any) -> None:
    """Reject accidental optimizer-state continuation at a study boundary."""

    if not hasattr(optimizer, "state") or not hasattr(optimizer, "param_groups"):
        raise TypeError("optimizer_factory did not return a PyTorch-compatible optimizer")
    if len(optimizer.state) != 0:
        raise ValueError(
            "Stage optimizer already has state. Construct it after loading parent weights; "
            "never restore optimizer/scheduler/global-step state across studies."
        )
    if not optimizer.param_groups:
        raise ValueError("Fresh optimizer has no parameter groups")
    if not any(group.get("params") for group in optimizer.param_groups):
        raise ValueError("Fresh optimizer has no parameters")


def fork_stage(
    *,
    model_factory: Callable[[], Any],
    parent_checkpoint: str | Path,
    optimizer_factory: Callable[[Any], Any],
    scheduler_factory: Callable[[Any], Any] | None = None,
    device: str | Any = "cpu",
    strict: bool = True,
) -> StageFork:
    """Fork a new continual stage from weights only and fresh optimizer state.

    Call this independently for each child condition (for example B0 and B-lambda).
    Do not reuse the returned model or optimizer between sibling conditions.
    """

    model = model_factory()
    result = load_weights_checkpoint(model, parent_checkpoint, strict=strict)
    parent_sha256 = result.checkpoint.state_sha256
    initial_sha256 = state_dict_sha256(model.state_dict())
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            "A stage fork requires an exact model-state match; "
            f"missing={list(result.missing_keys)}, unexpected={list(result.unexpected_keys)}"
        )
    if initial_sha256 != parent_sha256:
        raise RuntimeError(
            f"Child model hash {initial_sha256} differs from parent hash {parent_sha256}"
        )

    model = model.to(device)
    optimizer = optimizer_factory(model.parameters())
    assert_fresh_optimizer(optimizer)
    scheduler = scheduler_factory(optimizer) if scheduler_factory is not None else None
    assert_fresh_optimizer(optimizer)

    audit = ForkAudit(
        parent_checkpoint=result.checkpoint.path,
        parent_state_sha256=parent_sha256,
        initial_state_sha256=initial_sha256,
        optimizer_state_entries=len(optimizer.state),
        scheduler_type=(
            f"{type(scheduler).__module__}.{type(scheduler).__qualname__}"
            if scheduler is not None
            else None
        ),
    )
    return StageFork(model=model, optimizer=optimizer, scheduler=scheduler, audit=audit)
