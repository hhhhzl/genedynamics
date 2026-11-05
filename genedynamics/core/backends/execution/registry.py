"""Instance-owned factories avoid process-global backend selection."""


class ExecutionRegistry:
    def __init__(self):
        self._factories = {}

    def register(self, name, factory):
        if name in self._factories:
            raise ValueError(f"Execution backend already registered: {name}")
        self._factories[name] = factory

    def create(self, name, **configuration):
        if name not in self._factories:
            raise ValueError(
                f"Unknown execution backend {name!r}; available: {self.names()}"
            )
        return self._factories[name](**configuration)

    def names(self):
        return tuple(sorted(self._factories))
