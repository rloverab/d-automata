import time

from derivpy import Tick
import numpy as np

from config.strategies_config import MarkovTickParams
from core.models import Signal
from strategies.base import Strategy


class MarkovTickStrategy(Strategy):
    # def __init__(self, estimated_window: int = 2000):
    def __init__(self, params: MarkovTickParams):
        # Estados
        self.estimated_window = params.estimated_window
        self.ESTADOS = [
            'UP_LOW', 'UP_MED', 'UP_HIGH',
            'FLAT_LOW', 'FLAT_MED', 'FLAT_HIGH',
            'DOWN_LOW', 'DOWN_MED', 'DOWN_HIGH'
        ]
        self._mapa_estados = {s: i for i, s in enumerate(self.ESTADOS)}

        # Matriz de transición
        self.P = np.ones((9, 9)) / 9

        # Contador para reestimación (FIX: no usar len(buffer))
        self.ticks_desde_ultima_estimacion = 0
        # self.intervalo_reestimacion = 200  # ticks
        self.intervalo_reestimacion = params.intervalo_reestimacion  # ticks

        # Timestamp para reestimación temporal (backup)
        self.ultima_reestimacion_time = time.time()
        self.intervalo_reestimacion_time = 300  # segundos

        # Umbrales (AJUSTADOS)
        # self.UMBRAL_PROB = 0.50  # más realista para 9 estados
        self.UMBRAL_PROB = params.umbral_prob  # más realista para 9 estados
        self.UMBRAL_HIT_RATE_MIN = 0.52  # apenas mejor que aleatorio

        # Ventana de análisis para características
        self.ventana_caracteristicas = params.ventana_caracteristicas  # REDUCIDA para coherencia con duración 10
        # self.ventana_caracteristicas = 15  # REDUCIDA para coherencia con duración 10
        # self.ventana_caracteristicas = 1500  # REDUCIDA para coherencia con duración 10

        self._ultima_actualizacion: Tick | None = None

    def calcular_caracteristicas(self, ticks_recientes: list[Tick], ventana=None):
        """
        Calcula estado de régimen basado en dirección y volatilidad.
        """
        if ventana is None:
            ventana = self.ventana_caracteristicas
        
        precios = np.array([t.quote for t in ticks_recientes[-ventana:]])
        
        if len(precios) < ventana:
            return 'FLAT_LOW', 0.0
        
        # 1. Dirección: comparación con media móvil
        media = np.mean(precios)
        precio_actual = precios[-1]
        
        # Umbrales dinámicos basados en volatilidad
        vol_base = np.std(np.diff(precios) / precios[:-1])
        umbral_dir = max(vol_base * 2, 0.00005)  # al menos 0.5 pips
        
        if precio_actual > media * (1 + umbral_dir):
            direccion = 'UP'
        elif precio_actual < media * (1 - umbral_dir):
            direccion = 'DOWN'
        else:
            direccion = 'FLAT'
        
        # 2. Volatilidad: desviación estándar de retornos
        retornos = np.diff(precios) / precios[:-1]
        vol = np.std(retornos)
        
        # Umbrales dinámicos basados en percentiles históricos
        # (idealmente calcular sobre ventana más larga)
        if vol < 0.00015:
            volatilidad = 'LOW'
        elif vol < 0.0004:
            volatilidad = 'MED'
        else:
            volatilidad = 'HIGH'
        
        # 3. Momentum: pendiente de regresión lineal
        x = np.arange(len(precios))
        slope = np.polyfit(x, precios, 1)[0]
        
        return f"{direccion}_{volatilidad}", slope

    def actualizar_matriz(self, ticks_buffer: list[Tick], verbose: bool = False):
        """Reestima P usando toda la ventana de ticks disponible."""
        if len(ticks_buffer) < 500:
            return  # no hay suficientes datos
        
        # Calcular secuencia de estados
        estados_secuencia = []
        ticks_list = list(ticks_buffer)
        
        for i in range(len(ticks_list) - self.ventana_caracteristicas):
            ticks_ventana = ticks_list[i:i + self.ventana_caracteristicas]
            estado, _ = self.calcular_caracteristicas(ticks_ventana)
            estados_secuencia.append(estado)
        
        # Construir matriz de conteos
        conteos = np.zeros((9, 9))
        for t in range(len(estados_secuencia) - 1):
            i = self._mapa_estados[estados_secuencia[t]]
            j = self._mapa_estados[estados_secuencia[t + 1]]
            conteos[i, j] += 1
        
        # Normalizar con suavizado Laplace
        self.P = (conteos + 1) / (conteos.sum(axis=1, keepdims=True) + 9)
        
        # Resetear contadores
        self.ticks_desde_ultima_estimacion = 0
        self.ultima_reestimacion_time = time.time()

        if verbose:
            print(f"🔄 Matriz P reestimada con {len(estados_secuencia)} transiciones")

    def get_signal(self, data, duration = 10, duration_unit = "t", **kwargs):
        """Genera señal CALL/PUT/HOLD basada en estado actual."""
        ticks_buffer: list[Tick] = data
        if len(ticks_buffer) < self.estimated_window:
            return None

        # Calcular estado actual
        ticks_ventana = list(ticks_buffer)[-self.ventana_caracteristicas:]
        estado_actual, momentum = self.calcular_caracteristicas(ticks_ventana)

        i = self._mapa_estados[estado_actual]
        probs = self.P[i]

        # Buscar estado siguiente más probable
        j_max = np.argmax(probs)
        estado_siguiente = self.ESTADOS[j_max]
        prob_max = probs[j_max]

        # Lógica de decisión
        if prob_max < self.UMBRAL_PROB:
            return None

        # Extraer dirección del estado siguiente
        if estado_siguiente.startswith('UP'):
            if momentum > 0:
                return Signal(direction='CALL', data=ticks_buffer, duration=duration, duration_unit=duration_unit, extras={"momentum": momentum, "estado": estado_siguiente})
        elif estado_siguiente.startswith('DOWN'):
            if momentum < 0:
                return Signal(direction='PUT', data=ticks_buffer, duration=duration, duration_unit=duration_unit, extras={"momentum": momentum, "estado": estado_siguiente})
        else:
            return None
