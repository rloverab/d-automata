from derivpy import Tick
from typing import Optional
import numpy as np

from core.models import Signal
from filters.base import Filter


class SpreadTrendFilter(Filter):
    """
    Filtra señales cuando el spread está en una tendencia fuerte (expandiéndose o contrayéndose).
    
    Útil para detectar cambios de régimen de liquidez. Un spread en tendencia alcista
    indica deterioro de liquidez, mientras que uno en tendencia bajista indica mejora.
    
    Args:
        window_size: Ventana de ticks para calcular la tendencia. Por defecto 100.
        min_slope_threshold: Pendiente mínima (en unidades de spread por tick) para considerar
                            que hay tendencia. Por defecto 0.0001.
        min_ticks: Mínimo de ticks necesarios. Por defecto 150.
    """
    
    def __init__(self, window_size: int = 100, min_slope_threshold: float = 0.0001, min_ticks: int = 150):
        self.window_size = window_size
        self.min_slope_threshold = min_slope_threshold
        self.min_ticks = min_ticks
    
    def __call__(self, signal: Signal) -> Optional[Signal]:
        if not signal.data or len(signal.data) < self.min_ticks:
            return None
        
        # Extraer ticks válidos
        ticks = [t for t in signal.data if isinstance(t, Tick) and t.bid is not None and t.ask is not None]
        if len(ticks) < self.min_ticks:
            return None
        
        # Tomar los últimos window_size ticks
        recent_ticks = ticks[-self.window_size:]
        spreads = np.array([t.ask - t.bid for t in recent_ticks])
        
        # Calcular tendencia usando regresión lineal simple
        x = np.arange(len(spreads))
        
        # Pendiente de la regresión lineal
        slope = np.polyfit(x, spreads, 1)[0]
        
        # Si la pendiente es significativa (tendencia fuerte), filtrar
        if abs(slope) > self.min_slope_threshold:
            return None
        
        return signal