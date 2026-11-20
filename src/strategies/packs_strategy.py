from collections import Counter
from typing import Any, Literal

import numpy as np
from derivpy import Candle, Tick
from pydantic import BaseModel, ConfigDict, Field

from core.models import Signal
from strategies.base import Strategy  # Ajusta la ruta de importación según tu proyecto


# =====================================================================
# 1. REVERSIÓN POR AGOTAMIENTO DE RACHA (Z-Score + Desaceleración)
# =====================================================================

class ZScoreReversionConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    short_window: int = Field(default=5, ge=3, description="Ventana de impulso reciente")
    long_window: int = Field(default=50, ge=20, description="Ventana base de volatilidad")
    z_threshold: float = Field(default=2.5, gt=1.0, description="Umbral de desviación estándar")
    require_deceleration: bool = Field(
        default=True,
        description="Exige que el último tick sea menor al penúltimo (vital para entrada en T+1)",
    )
    default_duration: int = Field(default=4, ge=1, le=10)


class ZScoreReversionStrategy(Strategy):
    def __init__(self, config: ZScoreReversionConfig | None = None) -> None:
        self.config = config or ZScoreReversionConfig()

    def get_signal(
        self,
        data: list[Candle | Tick],
        duration: int = 4,
        duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
        **kwargs: Any,
    ) -> Signal | None:
        if duration_unit != "t":
            return None

        min_required = self.config.long_window + 1
        if len(data) < min_required or not isinstance(data[-1], Tick):
            return None

        # Extraer cotizaciones recientes necesarias
        quotes = np.array([t.quote for t in data[-min_required:]], dtype=np.float64)
        deltas = np.diff(quotes)

        # Retorno acumulado ventana corta y volatilidad ventana larga
        short_deltas = deltas[-self.config.short_window :]
        r_short = np.sum(short_deltas)
        sigma_long = np.std(deltas, ddof=1)

        if sigma_long == 0:
            return None

        z_score = float(r_short / (sigma_long * np.sqrt(self.config.short_window)))

        # Filtro de desaceleración para proteger la entrada en T+1
        if self.config.require_deceleration:
            if abs(deltas[-1]) >= abs(deltas[-2]):
                return None

        direction: Literal["CALL", "PUT"] | None = None
        if z_score > self.config.z_threshold:
            direction = "PUT"
        elif z_score < -self.config.z_threshold:
            direction = "CALL"

        if not direction:
            return None

        target_duration = duration if 1 <= duration <= 10 else self.config.default_duration

        return Signal(
            data=data[-min_required:],
            direction=direction,
            duration=target_duration,
            duration_unit="t",
            extras={
                "strategy": "ZScoreReversion",
                "z_score": round(z_score, 4),
                "sigma_long": round(float(sigma_long), 6),
                "last_delta": round(float(deltas[-1]), 6),
            },
        )


# =====================================================================
# 2. RUPTURA DE COMPRESIÓN DE VOLATILIDAD (Squeeze Breakout)
# =====================================================================

class VolatilitySqueezeConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    short_atr_window: int = Field(default=5, ge=3)
    long_atr_window: int = Field(default=30, ge=15)
    compression_ratio: float = Field(default=0.5, gt=0.1, lt=1.0)
    squeeze_ticks: int = Field(default=4, ge=3, description="Ticks consecutivos en compresión")
    breakout_multiplier: float = Field(
        default=2.2,
        gt=1.0,
        description="Multiplicador de ATR largo para confirmar ignición antes de T+1",
    )
    default_duration: int = Field(default=7, ge=1, le=10)


class VolatilitySqueezeStrategy(Strategy):
    def __init__(self, config: VolatilitySqueezeConfig | None = None) -> None:
        self.config = config or VolatilitySqueezeConfig()

    def get_signal(
        self,
        data: list[Candle | Tick],
        duration: int = 7,
        duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
        **kwargs: Any,
    ) -> Signal | None:
        if duration_unit != "t":
            return None

        min_required = self.config.long_atr_window + self.config.squeeze_ticks + 2
        if len(data) < min_required or not isinstance(data[-1], Tick):
            return None

        quotes = np.array([t.quote for t in data[-min_required:]], dtype=np.float64)
        abs_deltas = np.abs(np.diff(quotes))

        # Evaluar estado previo al tick actual (hasta t-1)
        prev_abs_deltas = abs_deltas[:-1]
        atr_long = np.mean(prev_abs_deltas[-self.config.long_atr_window :])

        if atr_long == 0:
            return None

        # Verificar que los últimos `squeeze_ticks` anteriores tuvieron baja volatilidad
        recent_atr_short = np.mean(prev_abs_deltas[-self.config.squeeze_ticks :])
        if (recent_atr_short / atr_long) >= self.config.compression_ratio:
            return None

        # Caja de consolidación de los últimos N ticks antes de la ruptura
        box_quotes = quotes[-(self.config.squeeze_ticks + 1) : -1]
        box_high = np.max(box_quotes)
        box_low = np.min(box_quotes)

        current_quote = quotes[-1]
        current_delta = quotes[-1] - quotes[-2]

        direction: Literal["CALL", "PUT"] | None = None
        if current_quote > box_high and current_delta > (self.config.breakout_multiplier * atr_long):
            direction = "CALL"
        elif current_quote < box_low and current_delta < -(self.config.breakout_multiplier * atr_long):
            direction = "PUT"

        if not direction:
            return None

        target_duration = duration if 1 <= duration <= 10 else self.config.default_duration

        return Signal(
            data=data[-min_required:],
            direction=direction,
            duration=target_duration,
            duration_unit="t",
            extras={
                "strategy": "VolatilitySqueeze",
                "compression_ratio": round(float(recent_atr_short / atr_long), 4),
                "breakout_strength": round(float(abs(current_delta) / atr_long), 4),
                "box_high": float(box_high),
                "box_low": float(box_low),
            },
        )


# =====================================================================
# 3. ASIMETRÍA DE MICRO-VELAS (Tick Imbalance & Absorción)
# =====================================================================

class TickImbalanceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: int = Field(default=10, ge=5, le=30)
    imbalance_threshold: float = Field(default=0.8, ge=0.6, le=1.0)
    max_efficiency_ratio: float = Field(default=0.4, gt=0.05, lt=0.8)
    default_duration: int = Field(default=2, ge=1, le=5)


class TickImbalanceStrategy(Strategy):
    def __init__(self, config: TickImbalanceConfig | None = None) -> None:
        self.config = config or TickImbalanceConfig()

    def get_signal(
        self,
        data: list[Candle | Tick],
        duration: int = 2,
        duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
        **kwargs: Any,
    ) -> Signal | None:
        if duration_unit != "t":
            return None

        min_required = self.config.window + 1
        if len(data) < min_required or not isinstance(data[-1], Tick):
            return None

        quotes = np.array([t.quote for t in data[-min_required:]], dtype=np.float64)
        deltas = np.diff(quotes)

        up_ticks = np.sum(deltas > 0)
        down_ticks = np.sum(deltas < 0)

        imbalance = float((up_ticks - down_ticks) / self.config.window)
        total_path = float(np.sum(np.abs(deltas)))

        if total_path == 0:
            return None

        net_displacement = abs(float(quotes[-1] - quotes[0]))
        efficiency_ratio = net_displacement / total_path

        # Confirmación para entrada en T+1: el último tick debe ser el que generó la absorción
        # (es decir, ir en contra de la mayoría de ticks débiles)
        direction: Literal["CALL", "PUT"] | None = None
        if imbalance >= self.config.imbalance_threshold and efficiency_ratio < self.config.max_efficiency_ratio:
            if deltas[-1] < 0:  # Absorción bajista confirmada en el tick de disparo
                direction = "PUT"
        elif imbalance <= -self.config.imbalance_threshold and efficiency_ratio < self.config.max_efficiency_ratio:
            if deltas[-1] > 0:  # Absorción alcista confirmada en el tick de disparo
                direction = "CALL"

        if not direction:
            return None

        target_duration = duration if 1 <= duration <= 5 else self.config.default_duration

        return Signal(
            data=data[-min_required:],
            direction=direction,
            duration=target_duration,
            duration_unit="t",
            extras={
                "strategy": "TickImbalance",
                "imbalance": round(imbalance, 4),
                "efficiency_ratio": round(efficiency_ratio, 4),
            },
        )


# =====================================================================
# 4. FILTRO DE ENTROPÍA DE SHANNON (Regímenes Deterministas)
# =====================================================================

class ShannonEntropyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: int = Field(default=24, ge=15, le=60, description="Ventana de ticks para calcular frecuencias")
    word_length: int = Field(default=3, ge=2, le=4, description="Longitud del patrón binario")
    entropy_threshold: float = Field(
        default=2.2,
        gt=0.5,
        description="Entropía máxima permitida (para k=3 el máximo teórico es 3.0 bits)",
    )
    min_conditional_prob: float = Field(
        default=0.70,
        ge=0.55,
        le=1.0,
        description="Probabilidad mínima del siguiente bit dado el prefijo actual",
    )
    default_duration: int = Field(default=3, ge=1, le=5)


class ShannonEntropyStrategy(Strategy):
    def __init__(self, config: ShannonEntropyConfig | None = None) -> None:
        self.config = config or ShannonEntropyConfig()

    def get_signal(
        self,
        data: list[Candle | Tick],
        duration: int = 3,
        duration_unit: Literal["d", "m", "s", "h", "t"] = "t",
        **kwargs: Any,
    ) -> Signal | None:
        if duration_unit != "t":
            return None

        min_required = self.config.window + 1
        if len(data) < min_required or not isinstance(data[-1], Tick):
            return None

        quotes = np.array([t.quote for t in data[-min_required:]], dtype=np.float64)
        deltas = np.diff(quotes)

        # Ignorar ticks sin cambio de precio (0) para mantener alfabeto binario puro {0, 1}
        non_zero_deltas = deltas[deltas != 0]
        k = self.config.word_length

        if len(non_zero_deltas) < (k + 8):
            return None

        bits = "".join("1" if d > 0 else "0" for d in non_zero_deltas)

        # Extraer palabras superpuestas de longitud k
        words = [bits[i : i + k] for i in range(len(bits) - k + 1)]
        total_words = len(words)
        counts = Counter(words)

        # Cálculo de Entropía de Shannon: H = -sum(p_i * log2(p_i))
        probs = np.array([count / total_words for count in counts.values()], dtype=np.float64)
        entropy = float(-np.sum(probs * np.log2(probs)))

        if entropy >= self.config.entropy_threshold:
            return None

        # Evaluar el prefijo actual de longitud (k - 1) para predecir el bit en T+1
        prefix = bits[-(k - 1) :]
        count_up = counts.get(prefix + "1", 0)
        count_down = counts.get(prefix + "0", 0)
        total_prefix = count_up + count_down

        if total_prefix < 3:  # Muestra mínima para evitar sobreajuste
            return None

        prob_up = count_up / total_prefix
        prob_down = count_down / total_prefix

        direction: Literal["CALL", "PUT"] | None = None
        win_prob = 0.0

        if prob_up >= self.config.min_conditional_prob:
            direction = "CALL"
            win_prob = prob_up
        elif prob_down >= self.config.min_conditional_prob:
            direction = "PUT"
            win_prob = prob_down

        if not direction:
            return None

        target_duration = duration if 1 <= duration <= 5 else self.config.default_duration

        return Signal(
            data=data[-min_required:],
            direction=direction,
            duration=target_duration,
            duration_unit="t",
            extras={
                "strategy": "ShannonEntropy",
                "entropy_bits": round(entropy, 4),
                "matched_prefix": prefix,
                "conditional_prob": round(win_prob, 4),
            },
        )