from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np
Array = np.ndarray


@dataclass
class EnergyTerm:
    fn: Callable[[Array, Array, Dict[str, Any]], Array]
    weight: float = 1.0

    def __call__(self, x: Array, u: Array, info: Dict[str, Any] | None):
        return self.weight * self.fn(x, u, info)


class EnergyFunctional:
    def __init__(self, terms: Dict[str, EnergyTerm]):
        self.terms = dict(terms)
        self._term_items: Tuple[Tuple[str, EnergyTerm], ...] = tuple(self.terms.items())

    def __call__(self, x: Array, u: Array, info: Dict[str, Any] = None) -> Array:
        return self.compute(x, u, info)

    def compute(self, x: Array, u: Array, info: Dict[str, Any] | None = None) -> Array:
        total = 0.0
        for _, term in self._term_items:
            total = total + term(x, u, info)
        return total

    def compute_terms(self, x: Array, u: Array, info: Dict[str, Any] | None = None) -> Iterable[Tuple[str, Array]]:
        for name, term in self._term_items:
            yield name, term(x, u, info)

    def breakdown(self, x: Array, u: Array, info: Dict[str, Any] = None):
        out = {}
        info_local = info or {}
        for name, term in self.terms.items():
            out[name] = float(term(x, u, info_local))
        return out
