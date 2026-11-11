import math
from collections import deque

from derivpy import Candle, Tick

from core.models import Signal
from filters.base import Filter


class VolatilityRegimeFilter(Filter):
    """
    Filtro que bloquea señales cuando el mercado está en régimen
    desfavorable (choppy/errático o volatilidad extrema).
    
    Métricas:
    - efficiency_ratio: hipotenusa (desplazamiento neto) / camino recorrido.
      Cerca de 1.0 → tendencia limpia. Cerca de 0.0 → ruido lateral.
    - rolling_volatility: desviación estándar de los cambios tick a tick.
    - max_abs_delta: mayor salto absoluto en la ventana (detecta "propulsores").
    """

    def __init__(
        self,
        window: int = 20,
        min_efficiency: float = 0.25,
        max_volatility_percentile: float = 95.0,
        max_abs_delta_threshold: float | None = None,
        block_on_choppy: bool = True,
        block_on_spike: bool = True,
    ):
        """
        Args:
            window: Número de ticks para calcular el régimen.
            min_efficiency: Umbral mínimo de eficiencia (hipotenusa/path).
                            Debajo de esto → choppy → bloquear.
            max_volatility_percentile: Percentil histórico de volatilidad
                                       por encima del cual se bloquea.
            max_abs_delta_threshold: Si un delta absoluto supera este valor,
                                     se considera "spike" y se bloquea.
                                     None = auto-calibrado con la media.
            block_on_choppy: Bloquear en régimen lateral/ruido.
            block_on_spike: Bloquear en régimen de salto extremo.
        """
        self.window = window
        self.min_efficiency = min_efficiency
        self.max_volatility_percentile = max_volatility_percentile
        self.max_abs_delta_threshold = max_abs_delta_threshold
        self.block_on_choppy = block_on_choppy
        self.block_on_spike = block_on_spike

        # Buffer circular para mantener historial de volatilidades
        # y poder calcular percentiles de forma rolling.
        self._vol_history: deque[float] = deque(maxlen=200)

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def __call__(
        self,
        signal: Signal
    ) -> Signal | None:
        """
        Evalúa si la señal debe pasar o ser bloqueada según el régimen.
        
        Returns:
            La misma Signal si el régimen es favorable.
            None si debe bloquearse.
        """
        data: list[Tick] | None = signal.data
        if data is None or len(data) < self.window:
            # Sin datos suficientes → dejar pasar (o bloquear, según criterio)
            return None

        # Extraer precios
        prices = [t.quote for t in data[-self.window:]]

        # Calcular métricas de régimen
        efficiency = self._efficiency_ratio(prices)
        volatility = self._rolling_volatility(prices)
        max_delta = self._max_abs_delta(prices)

        # Actualizar historial para percentiles
        self._vol_history.append(volatility)

        # Determinar umbrales
        vol_threshold = self._percentile(
            list(self._vol_history), self.max_volatility_percentile
        )

        # Calibración automática del umbral de spike
        if self.max_abs_delta_threshold is None:
            mean_abs = self._mean_abs_delta(prices)
            spike_threshold = mean_abs * 3.0  # 3x la media = spike
        else:
            spike_threshold = self.max_abs_delta_threshold

        # ----------------------------------------------------------
        # Lógica de bloqueo
        # ----------------------------------------------------------

        # 1. Régimen choppy: eficiencia baja → el mercado no va a ningún lado
        if self.block_on_choppy and efficiency < self.min_efficiency:
            return None

        # 2. Régimen de spike: volatilidad extrema o salto anómalo
        if self.block_on_spike:
            if volatility > vol_threshold:
                return None
            if max_delta > spike_threshold:
                return None

        # 3. Si pasa todos los checks → señal válida
        return signal

    # ------------------------------------------------------------------
    # Cálculos internos
    # ------------------------------------------------------------------

    @staticmethod
    def _efficiency_ratio(prices: list[float]) -> float:
        """
        Ratio de eficiencia (Kaufman adaptado a ticks).
        
        efficiency = |P_n - P_0| / Σ|P_i - P_{i-1}|
        
        - 1.0 → tendencia perfecta (hipotenusa = camino)
        - 0.0 → ruido puro (ida y vuelta, desplazamiento neto = 0)
        """
        if len(prices) < 2:
            return 1.0

        path_length = sum(
            abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))
        )

        if path_length == 0:
            return 1.0  # Sin movimiento → no hay señal posible de todos modos

        net_displacement = abs(prices[-1] - prices[0])
        return net_displacement / path_length

    @staticmethod
    def _rolling_volatility(prices: list[float]) -> float:
        """Desviación estándar de los deltas tick a tick."""
        if len(prices) < 3:
            return 0.0

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        n = len(deltas)
        mean = sum(deltas) / n
        variance = sum((d - mean) ** 2 for d in deltas) / n
        return math.sqrt(variance)

    @staticmethod
    def _max_abs_delta(prices: list[float]) -> float:
        """Mayor salto absoluto entre ticks consecutivos."""
        if len(prices) < 2:
            return 0.0
        return max(
            abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))
        )

    @staticmethod
    def _mean_abs_delta(prices: list[float]) -> float:
        """Media de los valores absolutos de los deltas."""
        if len(prices) < 2:
            return 0.0
        deltas = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        return sum(deltas) / len(deltas)

    @staticmethod
    def _percentile(data: list[float], pct: float) -> float:
        """Percentil simple sin dependencias externas."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        idx = (pct / 100.0) * (len(sorted_data) - 1)
        lower = int(math.floor(idx))
        upper = int(math.ceil(idx))
        if lower == upper:
            return sorted_data[lower]
        frac = idx - lower
        return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac