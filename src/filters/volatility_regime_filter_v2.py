import math

from derivpy import Tick

from core.models import Signal
from filters.base import Filter


class VolatilityRegimeFilterV2(Filter):
    """
    Bloquea señales cuando el mercado está en un régimen desfavorable.

    Dos condiciones de bloqueo:
      1. CHOPPY: el precio se mueve mucho pero no va a ningún lado
         (eficiencia baja → ruido lateral).
      2. SPIKE: hubo un salto de precio anormalmente grande
         (propulsor → la señal probablemente falle).

    Si ninguna condición se activa, la señal pasa.
    """

    def __init__(
        self,
        window: int = 20,
        min_efficiency: float = 0.15,
        spike_multiplier: float = 4.0,
        warmup_ticks: int = 50,
    ):
        """
        Args:
            window: cuántos ticks miro hacia atrás para medir el régimen.
            min_efficiency: debajo de este valor → mercado "ruidoso" → bloquear.
                            0.0 = totalmente lateral, 1.0 = tendencia perfecta.
                            0.15 es permisivo; 0.30 es estricto.
            spike_multiplier: si algún delta es mayor a (media × este número)
                              → hubo un "propulsor" → bloquear.
                              4.0 es permisivo; 2.5 es estricto.
            warmup_ticks: mínimo de ticks en signal.data antes de que el
                          filtro empiece a bloquear. Antes de eso, deja pasar.
        """
        self.window = window
        self.min_efficiency = min_efficiency
        self.spike_multiplier = spike_multiplier
        self.warmup_ticks = warmup_ticks

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def __call__(self, signal: Signal) -> Signal | None:
        """
        Decide si la señal pasa o se bloquea.

        Returns:
            signal  → el régimen es favorable, operar.
            None    → régimen desfavorable, no operar.
        """
        data = list(signal.data)

        # Sin datos o insuficientes → dejar pasar (no podemos medir)
        if data is None or len(data) < self.window:
            return signal  # ← CAMBIO: antes retornaba None (bloqueaba)

        # Tomar los últimos `window` precios
        prices = [t.quote for t in data[-self.window:]]

        # Calcular las dos métricas
        efficiency = self._efficiency(prices)
        has_spike = self._has_spike(prices)

        # --- Decisión ---

        # Condición 1: mercado lateral/ruidoso
        if efficiency < self.min_efficiency:
            return None

        # Condición 2: salto anómalo reciente
        if has_spike:
            return None

        # Ninguna condición se activó → señal válida
        return signal

    # ------------------------------------------------------------------
    # Métricas internas
    # ------------------------------------------------------------------

    @staticmethod
    def _efficiency(prices: list[float]) -> float:
        """
        Mide qué tan "directo" fue el movimiento del precio.

        Fórmula:  |precio_final - precio_inicial| / suma_de_todos_los_movimientos

        Ejemplo:
          Precios: [100, 101, 102, 103, 104]
          Desplazamiento neto: |104 - 100| = 4
          Camino recorrido: 1+1+1+1 = 4
          Eficiencia: 4/4 = 1.0  → tendencia perfecta

          Precios: [100, 101, 100, 101, 100]
          Desplazamiento neto: |100 - 100| = 0
          Camino recorrido: 1+1+1+1 = 4
          Eficiencia: 0/4 = 0.0  → ruido puro
        """
        if len(prices) < 2:
            return 1.0

        net = abs(prices[-1] - prices[0])
        path = sum(abs(prices[i] - prices[i - 1]) for i in range(1, len(prices)))

        if path == 0:
            return 1.0  # precio completamente quieto

        return net / path

    def _has_spike(self, prices: list[float]) -> bool:
        """
        Detecta si hubo un salto anormalmente grande en la ventana.

        Un "spike" es un delta individual que supera la media de todos
        los deltas multiplicada por `spike_multiplier`.

        Ejemplo con spike_multiplier=4:
          Deltas: [0.1, 0.2, 0.1, 0.15, 0.9]
          Media:  0.29
          Umbral: 0.29 × 4 = 1.16
          ¿0.9 > 1.16? → No → no hay spike

          Deltas: [0.1, 0.1, 0.1, 0.1, 0.6]
          Media:  0.2
          Umbral: 0.2 × 4 = 0.8
          ¿0.6 > 0.8? → No → no hay spike

          Deltas: [0.05, 0.05, 0.05, 0.05, 0.5]
          Media:  0.14
          Umbral: 0.14 × 4 = 0.56
          ¿0.5 > 0.56? → No → no hay spike
        """
        if len(prices) < 3:
            return False

        deltas = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
        mean_delta = sum(deltas) / len(deltas)

        if mean_delta == 0:
            return False  # precio quieto, no puede haber spike

        threshold = mean_delta * self.spike_multiplier
        return any(d > threshold for d in deltas)