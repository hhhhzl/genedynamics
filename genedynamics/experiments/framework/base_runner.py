"""Small reusable study lifecycle; applications own their experiment semantics."""

from abc import ABC, abstractmethod


class BaseRunner(ABC):
    def __init__(self, configuration):
        self.configuration = configuration
        self.failures = []

    @abstractmethod
    def run_single_experiment(self, *arguments, **keywords): ...

    def run_entries(self, entries, *, persist=None):
        results = []
        self.failures = []
        for entry in entries:
            try:
                result = self.run_single_experiment(entry)
                if persist is not None:
                    persist(entry, result)
                results.append(result)
            except Exception as error:  # noqa: BLE001 - study failures are persisted per entry
                self.failures.append(
                    {
                        "entry": str(entry),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
        return results
