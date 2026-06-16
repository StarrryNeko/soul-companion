"""Base tool interface."""

from abc import ABC, abstractmethod


class BaseTool(ABC):
    """Common interface for assistant tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def execute(self, **kwargs) -> dict:
        raise NotImplementedError

