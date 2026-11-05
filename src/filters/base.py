from abc import ABC, abstractmethod

from core.models import Signal


class Filter(ABC):
    """Interfaz base para filtros de señal"""
    @abstractmethod
    def __call__(self, signal: Signal) -> Signal | None:
        raise NotImplementedError("Implementa __call__ en tu filtro")