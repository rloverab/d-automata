# Interfaces o modelos base de los que heredan las estrategias
from abc import ABC, abstractmethod
from typing import Any, Literal

from derivpy import Candle, Tick

from core.models import Signal


class Strategy(ABC):
    @abstractmethod
    def get_signal(
            self,            
            data: list[Candle|Tick],
            duration: int = 10,
            duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
            **kwargs: Any  # ← Extensibilidad segura
    ) -> Signal | None:
        pass