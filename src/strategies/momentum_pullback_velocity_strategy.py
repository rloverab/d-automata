from typing import Any, Literal

from derivpy import Candle, Tick

from core.models import Signal
from strategies.base import Strategy


class MomentumPullbackVelocityStrategy(Strategy):
    """
    Variante de MomentumPullbackAdaptiveStrategy que compara VELOCIDADES
    (pips por tick) en lugar de distancias acumuladas de precio.

    Patrón base (sin cambios):
        - Cinco ticks alcistas seguidos de un tick bajista:
            [1, 1, 1, 1, 1, -1]
        - Cinco ticks bajistas seguidos de un tick alcista:
            [-1, -1, -1, -1, -1, 1]

    Decisión adaptativa:
        impulse_velocity  = |p[-2] - p[-2 - impulse_ticks]| / pip / impulse_ticks
        pullback_velocity = |p[-1] - p[-2]| / pip / 1

        ratio = impulse_velocity / pullback_velocity

        ratio > 1  -> domina el impulso  -> continuación
        ratio < 1  -> domina el pullback -> reversión
        ratio == 1 -> empate             -> no operar

    Zona muerta (min_velocity_ratio):
        Con min_velocity_ratio = 1.0 no hay zona muerta (solo se descarta
        el empate exacto).

        Con min_velocity_ratio = 1.25 se descarta toda señal cuyo ratio
        caiga entre 0.80 y 1.25, por considerarse ambigua.

    Diferencia importante con la versión por distancia:
        Al dividir el impulso por su cantidad de ticks, el pullback
        (1 solo tick) domina con mucha más frecuencia. Esta versión
        genera más señales de reversión que la versión por distancia.

    Notas:
        - La velocidad se mide en pips por TICK, no por segundo.
          Si hay gaps de epoch, ticks transcurridos != segundos.
        - Se adjunta `data` completa a la Signal para que los filtros
          posteriores puedan calcular régimen.
    """

    PATTERN_LENGTH: int = 7
    DIRECTIONS_LENGTH: int = PATTERN_LENGTH - 1
    MOMENTUM_LENGTH: int = 5
    DEFAULT_IMPULSE_TICKS: int = 2
    DEFAULT_PIP_SIZE: int = 2

    def __init__(
        self,
        impulse_ticks: int = DEFAULT_IMPULSE_TICKS,
        min_velocity_ratio: float = 1.0,
        pip_size: int | None = None,
    ):
        """
        Args:
            impulse_ticks: cuántos ticks del impulso se promedian (1 a 5).
            min_velocity_ratio: umbral de la zona muerta. Debe ser >= 1.0.
                1.0 = sin zona muerta. 1.25 = exige 25% de diferencia.
            pip_size: decimales del pip. Si es None, se toma del Tick
                y, si no viene, se usa DEFAULT_PIP_SIZE.

        Raises:
            ValueError: si algún parámetro está fuera de rango.
        """
        if not 1 <= impulse_ticks <= self.MOMENTUM_LENGTH:
            raise ValueError(
                f"impulse_ticks debe estar entre 1 y {self.MOMENTUM_LENGTH}, "
                f"recibido: {impulse_ticks}"
            )

        if min_velocity_ratio < 1.0:
            raise ValueError(
                f"min_velocity_ratio debe ser >= 1.0, recibido: {min_velocity_ratio}"
            )

        self.impulse_ticks = impulse_ticks
        self.min_velocity_ratio = min_velocity_ratio
        self.pip_size = pip_size

    def get_signal(
        self,
        data: list[Candle | Tick],
        duration: int = 1,
        duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
        **kwargs: Any,
    ) -> Signal | None:
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

        pip_size = self._resolve_pip_size(window[-1])
        pip = 10.0 ** (-pip_size)

        impulse_velocity, pullback_velocity = self._calculate_velocities(prices, pip)

        # Con patrón válido el pullback nunca es cero, pero protegemos igual.
        if pullback_velocity == 0:
            return None

        ratio = impulse_velocity / pullback_velocity

        # Empate exacto -> no operar (tu criterio).
        if ratio == 1.0:
            return None

        # Zona muerta alrededor de 1.0 -> señal ambigua, no operar.
        upper = self.min_velocity_ratio
        lower = 1.0 / upper
        if lower < ratio < upper:
            return None

        impulse_dominates = ratio > 1.0

        if first_five[0] == 1:
            direction: Literal["CALL", "PUT"] = "CALL" if impulse_dominates else "PUT"
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
                impulse_velocity=impulse_velocity,
                pullback_velocity=pullback_velocity,
                velocity_ratio=ratio,
                impulse_dominates=impulse_dominates,
                pip_size=pip_size,
                signal_epoch=getattr(window[-1], "epoch", None),
            ),
        )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_directions(prices: list[float]) -> list[int]:
        """1 si sube, -1 si baja, 0 si no cambia."""
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
        """Cinco movimientos iguales y no nulos, más uno opuesto al final."""
        if len(first_five) != 5:
            return False

        base_direction = first_five[0]

        if base_direction == 0:
            return False

        if not all(d == base_direction for d in first_five):
            return False

        return last_direction == -base_direction

    def _resolve_pip_size(self, bar: Candle | Tick) -> int:
        """Prioridad: parámetro explícito > atributo del tick > default."""
        if self.pip_size is not None:
            return self.pip_size

        value = getattr(bar, "pip_size", None)
        return int(value) if value is not None else self.DEFAULT_PIP_SIZE

    def _calculate_velocities(
        self, prices: list[float], pip: float
    ) -> tuple[float, float]:
        """
        Velocidades en pips por tick.

        Para la ventana p0..p6:
            impulse  = |p5 - p(5 - impulse_ticks)| / pip / impulse_ticks
            pullback = |p6 - p5| / pip / 1
        """
        impulse_distance = abs(prices[-2] - prices[-2 - self.impulse_ticks])
        pullback_distance = abs(prices[-1] - prices[-2])

        impulse_velocity = impulse_distance / pip / self.impulse_ticks
        pullback_velocity = pullback_distance / pip

        return impulse_velocity, pullback_velocity

    def _build_extras(
        self,
        *,
        pattern: str,
        directions: list[int],
        impulse_velocity: float,
        pullback_velocity: float,
        velocity_ratio: float,
        impulse_dominates: bool,
        pip_size: int,
        signal_epoch: int | None,
    ) -> dict[str, Any]:
        return {
            "strategy": self.__class__.__name__,
            "mode": "velocity",
            "impulse_ticks": self.impulse_ticks,
            "min_velocity_ratio": self.min_velocity_ratio,
            "pip_size": pip_size,
            "pattern": pattern,
            "directions": directions,
            "impulse_velocity": impulse_velocity,
            "pullback_velocity": pullback_velocity,
            "velocity_ratio": velocity_ratio,
            "impulse_dominates": impulse_dominates,
            "signal_epoch": signal_epoch,
        }