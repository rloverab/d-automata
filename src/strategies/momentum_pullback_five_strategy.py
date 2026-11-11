from typing import Literal

from derivpy import Candle, Tick

from core.models import Signal
from strategies.base import Strategy


class MomentumPullbackFive(Strategy):
    """
    Estrategia de continuación tras pullback de 1 tick.
    
    Patrón: 5 ticks alcistas + 1 bajista → CALL
            5 ticks bajistas + 1 alcista → PUT
    
    La señal se genera en el tick siguiente (desfase de ejecución).
    Duración: 1 tick (entrada en k+7, salida en k+8).
    """

    PATTERN_LENGTH = 7

    def get_signal(
        self,
        data: list[Candle | Tick],
        duration: int = 1,
        duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
        **kwargs,
    ) -> Signal | None:
        if len(data) < self.PATTERN_LENGTH:
            return None

        # Tomar los últimos 7 ticks
        full_ticks = list(data)
        ticks = full_ticks[-self.PATTERN_LENGTH:]
        prices = [t.quote for t in ticks]

        # Calcular direcciones
        directions = []
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                directions.append(1)
            elif prices[i] < prices[i - 1]:
                directions.append(-1)
            else:
                directions.append(0)

        first_five = directions[:5]
        last_one = directions[5]

        # Patrón alcista: +++++ seguido de -
        if all(d == 1 for d in first_five) and last_one == -1:
            return Signal(
                data=full_ticks,
                direction="CALL",
                duration=duration,
                duration_unit=duration_unit,
                extras={
                    "strategy": "MomentumPullbackFive",
                    "pattern": "bullish_pullback",
                    "signal_epoch": ticks[-1].epoch,
                },
            )

        # Patrón bajista: ----- seguido de +
        if all(d == -1 for d in first_five) and last_one == 1:
            return Signal(
                data=full_ticks,
                direction="PUT",
                duration=duration,
                duration_unit=duration_unit,
                extras={
                    "strategy": "MomentumPullbackFive",
                    "pattern": "bearish_pullback",
                    "signal_epoch": ticks[-1].epoch,
                },
            )

        return None