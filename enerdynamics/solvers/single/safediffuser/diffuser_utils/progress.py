from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from tqdm import tqdm


@dataclass
class Silent:
    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def close(self) -> None:
        return


class Progress:
    """
    Tiny progress helper compatible with SafeDiffuser's diffusion loop.
    """

    def __init__(self, total: int):
        self._pbar = tqdm(total=total)

    def update(self, info: Optional[Dict[str, Any]] = None) -> None:
        _ = info
        self._pbar.update(1)

    def close(self) -> None:
        self._pbar.close()

