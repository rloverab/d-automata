from collections import deque
import logging

import numpy as np
from typing import Any, Literal
from derivpy import Tick, Candle
from config.strategies_config import RiseFallMontecarloParams
from core.models import Signal
from strategies.base import Strategy

logger = logging.getLogger(__name__)

class RiseFallMontecarloStrategy(Strategy):
    """
    Estrategia Rise/Fall basada en Montecarlo Empírico.
    Modela la distribución de los cambios (diffs) de los últimos N ticks 
    para predecir la suma de los cambios en los próximos 'duration' ticks.
    """
        
    def __init__(self, params: RiseFallMontecarloParams):
        self.window_size = params.window_size
        self.simulations = params.simulations
        self.threshold = params.threshold
        self.min_data_points = params.min_data_points

    def get_signal(
        self,
        data: list[Candle | Tick],
        duration: int = 5,
        duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
        **kwargs: Any
    ) -> Signal | None:
        
        # 1. Validación de contrato (esta estrategia es exclusiva para ticks)
        if duration_unit != "t":
            raise ValueError("Esta estrategia está diseñada exclusivamente para duración en ticks ('t').")
            
        if len(data) < self.min_data_points:
            logger.debug(f"📉 Datos insuficientes: {len(data)} < {self.min_data_points}")
            return None  # Datos insuficientes para inferir una distribución

        # 2. Extracción de la ventana de datos (últimos N ticks)
        # Asumimos que la lista 'data' viene ordenada cronológicamente por el motor
        # window_data = data[-self.window_size:] if len(data) >= self.window_size else data
        window_data = list(deque(data, maxlen=self.window_size))
        
        # Filtramos solo Ticks (por seguridad de tipos, aunque el stream sea homogéneo)
        ticks = [d for d in window_data if isinstance(d, Tick)]
        if len(ticks) < self.min_data_points:
            logger.debug(f"📉 Ticks insuficientes: {len(ticks)} < {self.min_data_points}")
            return None

        # 3. Cálculo de retornos (diferencias de precio tick a tick)
        prices = np.array([t.quote for t in ticks])
        diffs = np.diff(prices)  
        
        # Filtro de mercado "muerto" (desviación estándar cero)
        if np.std(diffs) == 0:
            logger.debug(f"📉 Volatilidad cero: std={np.std(diffs)}")
            return None

        # 4. Simulación de Montecarlo Empírico (Vectorizado)
        # Extraemos 'duration' cambios, 'simulations' veces con reemplazo.
        # Esto construye 10,000 caminos posibles de 'duration' ticks hacia adelante.
        # Nota: En producción, evita fijar la seed (np.random.seed) para garantizar 
        # la estocasticidad real en cada evaluación del motor.
        samples = np.random.choice(diffs, size=(self.simulations, duration), replace=True)
        future_paths = np.sum(samples, axis=1)
        
        # 5. Cálculo de Probabilidades de Dirección
        prob_rise = np.mean(future_paths > 0)
        prob_fall = np.mean(future_paths < 0)

        # 🔍 LOG DE DIAGNÓSTICO CRÍTICO
        logger.debug(
            f"🎲 Montecarlo -> prob_rise: {prob_rise:.3f} | "
            f"prob_fall: {prob_fall:.3f} | threshold: {self.threshold} | "
            f"std_dev: {np.std(diffs):.6f}"
        )
        
        # 6. Evaluación contra el Umbral (Threshold)
        direction = None
        if prob_rise >= self.threshold:
            direction = "CALL"
            # direction = "PUT"
        elif prob_fall >= self.threshold:
            direction = "PUT"
            # direction = "CALL"
            
        # 7. Emisión de la Señal
        if direction:
            return Signal(
                data=window_data,
                direction=direction,
                duration=duration,
                duration_unit="t",
                extras={
                    "prob_rise": float(prob_rise),
                    "prob_fall": float(prob_fall),
                    "model": "EmpiricalMontecarlo",
                    "window_size": len(ticks),
                    "std_dev": float(np.std(diffs))
                }
            )

        # Log cuando NO hay señal
        logger.debug(
            f"❌ Sin señal: prob_rise={prob_rise:.3f} < {self.threshold} | "
            f"prob_fall={prob_fall:.3f} < {self.threshold}"
        )
        return None