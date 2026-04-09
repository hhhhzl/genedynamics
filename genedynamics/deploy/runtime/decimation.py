"""Decimation helper — keeps one event per N calls.

The deploy loop runs at the **control rate** but typically wants to do
some work less often (logging, plan re-solving, observer updates). A
:class:`Decimation` is a tiny stateful counter::

    plan_dec = Decimation(every=20)
    for step in range(N):
        ...
        if plan_dec.tick():
            replan(...)

That's it — no rate-limiting, no clock, no thread. The pattern is so
common that wrapping it in one place saves the ad-hoc ``if step % N == 0``
sprinkle that gets bug-prone when ``N`` becomes a config value.
"""

from __future__ import annotations

__all__ = ["Decimation"]


class Decimation:
    """Yield ``True`` once every ``every`` calls; ``False`` otherwise.

    Args:
        every: Period in number of calls. Must be ``>= 1``. ``every=1``
            means every call returns ``True`` (the trivial case).
        offset: Number of leading calls to skip before the first ``True``.
            Useful when you want the first replan after warmup. Defaults
            to ``0`` so the first call is the first ``True``.
    """

    def __init__(self, every: int, *, offset: int = 0) -> None:
        if every < 1:
            raise ValueError(f"Decimation 'every' must be >= 1, got {every}")
        if offset < 0:
            raise ValueError(f"Decimation 'offset' must be >= 0, got {offset}")
        self.every = int(every)
        self._counter = -int(offset)

    def reset(self, offset: int = 0) -> None:
        self._counter = -int(offset)

    def tick(self) -> bool:
        """Advance the counter and report whether the action should fire."""
        fire = self._counter % self.every == 0 and self._counter >= 0
        self._counter += 1
        return fire

    @property
    def calls(self) -> int:
        return max(self._counter, 0)
