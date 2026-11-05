import logging

import numpy as np
from core.models import Signal
from derivpy import Tick

from filters.base import Filter

logger = logging.getLogger(__name__)

class VolatilityFilter(Filter):
    """
    Filtro de rechazo que descarta señales si la volatilidad reciente 
    (desviación estándar de los diffs) está por debajo de un umbral mínimo.
    """
    def __init__(self, min_volatility: float = 0.0001, lookback: int = 100):
        self.min_volatility = min_volatility
        self.lookback = lookback

    def __call__(self, signal: Signal) -> Signal | None:
        if not signal.data:
            logger.debug("🚫 Filtro: Sin datos en señal")
            return None
            
        # Tomamos los últimos 'lookback' ticks de los datos que trae la señal
        recent_data = signal.data[-self.lookback:]
        ticks = [d for d in recent_data if isinstance(d, Tick)]
        
        if len(ticks) < 2:
            logger.debug(f"🚫 Filtro: Ticks insuficientes ({len(ticks)} < 2)")
            return None
            
        prices = np.array([t.quote for t in ticks])
        diffs = np.diff(prices)
        volatility = np.std(diffs)

        logger.debug(f"🔍 Filtro volatilidad: {volatility:.6f} vs min: {self.min_volatility}")
        
        # Si la volatilidad es menor al mínimo requerido, rechazamos la señal (None)
        if volatility < self.min_volatility:
            logger.debug(f"🚫 Filtro rechazó: volatilidad {volatility:.6f} < {self.min_volatility}")
            return None
            
        # Si pasa el filtro, enriquecemos los metadatos y devolvemos la señal
        signal.extras["filtered_volatility"] = float(volatility)
        logger.debug(f"✅ Filtro aprobó señal con volatilidad: {volatility:.6f}")
        return signal