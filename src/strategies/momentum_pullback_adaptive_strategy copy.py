from typing import Any, Literal

from derivpy import Candle, Tick

from core.models import Signal
from strategies.base import Strategy


class MomentumPullbackAdaptiveStrategy(Strategy):
    """
    Estrategia experimental de pullback de 5 ticks con decisión adaptativa.

    Patrón base:
        - Cinco ticks alcistas seguidos de un tick bajista:
            [1, 1, 1, 1, 1, -1]

        - Cinco ticks bajistas seguidos de un tick alcista:
            [-1, -1, -1, -1, -1, 1]

    Variación respecto a una estrategia de pullback clásica:
        La dirección final no siempre es la de los cinco ticks iniciales.

        Se compara la fuerza del impulso reciente contra la fuerza del
        último tick, que actúa como pullback.

        Definiciones:
            impulse  = |precio[-2] - precio[-4]|
            pullback = |precio[-1] - precio[-2]|

        Si impulse > pullback:
            - Patrón +++++- opera CALL.
            - Patrón -----+ opera PUT.

        Si impulse <= pullback:
            - Patrón +++++- opera PUT.
            - Patrón -----+ opera CALL.

    Notas:
        - Esta estrategia está pensada para datos tick a tick.
        - La señal incluye `data` con la lista completa de ticks para que
          filtros posteriores, como filtros de régimen o volatilidad, puedan
          acceder al contexto de mercado.
        - La duración por defecto es 1 tick, coherente con la evaluación:
            señal en k+6, entrada en k+7, salida en k+8.
    """

    PATTERN_LENGTH: int = 7
    DIRECTIONS_LENGTH: int = PATTERN_LENGTH - 1
    MOMENTUM_LENGTH: int = 5

    def get_signal(
        self,
        data: list[Candle | Tick],
        duration: int = 1,
        duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
        **kwargs: Any,
    ) -> Signal | None:
        """
        Evalúa los últimos ticks y genera una señal si se cumple el patrón.

        Returns:
            Signal: si existe patrón válido.
            None: si no hay datos suficientes o no hay patrón.
        """
        ticks = list(data)

        if len(ticks) < self.PATTERN_LENGTH:
            return None

        window = ticks[-self.PATTERN_LENGTH :]
        prices = [float(bar.quote) for bar in window]

        directions = self._calculate_directions(prices)

        if len(directions) != self.DIRECTIONS_LENGTH:
            return None

        first_five = directions[: self.MOMENTUM_LENGTH]
        last_direction = directions[-1]

        if not self._is_valid_pattern(first_five, last_direction):
            return None

        impulse, pullback = self._calculate_impulse_and_pullback(prices)

        if impulse == pullback:
            return None

        impulse_dominates = impulse > pullback

        if first_five[0] == 1:
            direction: Literal["CALL", "PUT"] = (
                "CALL" if impulse_dominates else "PUT"
            )
            pattern = "bullish_pullback"
        else:
            direction = "PUT" if impulse_dominates else "CALL"
            pattern = "bearish_pullback"

        return Signal(
            data=ticks,
            direction=direction,
            duration=duration,
            duration_unit=duration_unit,
            extras=self._build_extras(
                pattern=pattern,
                directions=directions,
                impulse=impulse,
                pullback=pullback,
                impulse_dominates=impulse_dominates,
                signal_epoch=getattr(window[-1], "epoch", None),
            ),
        )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_directions(prices: list[float]) -> list[int]:
        """
        Convierte una serie de precios en direcciones tick a tick.

        Returns:
            Lista de direcciones:
                1  → precio sube
               -1  → precio baja
                0  → precio sin cambio
        """
        directions: list[int] = []

        for previous_price, current_price in zip(prices, prices[1:]):
            if current_price > previous_price:
                directions.append(1)
            elif current_price < previous_price:
                directions.append(-1)
            else:
                directions.append(0)

        return directions

    @staticmethod
    def _is_valid_pattern(first_five: list[int], last_direction: int) -> bool:
        """
        Valida que los primeros cinco movimientos sean iguales y no nulos,
        y que el último movimiento sea contrario.
        """
        if len(first_five) != 5:
            return False

        base_direction = first_five[0]

        if base_direction == 0:
            return False

        if not all(direction == base_direction for direction in first_five):
            return False

        return last_direction == -base_direction

    @staticmethod
    def _calculate_impulse_and_pullback(
        prices: list[float],
    ) -> tuple[float, float]:
        """
        Calcula el impulso reciente y el pullback final.

        Para una ventana de 7 precios:

            p0, p1, p2, p3, p4, p5, p6

        Se usa:

            impulse  = |p5 - p3|
            pullback = |p6 - p5|

        Esta es la comparación exacta de la versión experimental original.
        """
        impulse = abs(prices[-2] - prices[-4])
        pullback = abs(prices[-1] - prices[-2])

        return impulse, pullback

    @staticmethod
    def _build_extras(
        *,
        pattern: str,
        directions: list[int],
        impulse: float,
        pullback: float,
        impulse_dominates: bool,
        signal_epoch: int | None,
    ) -> dict[str, Any]:
        """
        Construye metadatos útiles para debugging, logging y filtros.
        """
        ratio: float | None = None

        if pullback != 0:
            ratio = impulse / pullback

        return {
            "strategy": "MomentumPullbackAdaptiveStrategy",
            "pattern": pattern,
            "directions": directions,
            "impulse": impulse,
            "pullback": pullback,
            "impulse_pullback_ratio": ratio,
            "impulse_dominates": impulse_dominates,
            "signal_epoch": signal_epoch,
        }