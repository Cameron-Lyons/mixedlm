from __future__ import annotations

from collections.abc import Iterable, Sequence


def _check_unique_coefficient_names(
    names: Sequence[str],
    requested: Iterable[str] | None = None,
    *,
    alternative: str,
) -> None:
    """Reject ambiguous labels used by a named selection or dictionary result."""
    unique = set(names)
    if len(unique) == len(names):
        return
    selected = unique if requested is None else set(requested)
    seen: set[str] = set()
    for name in names:
        if name in seen and name in selected:
            raise ValueError(
                f"Ambiguous coefficient name {name!r}: multiple fitted columns use this name. "
                f"{alternative}"
            )
        seen.add(name)
