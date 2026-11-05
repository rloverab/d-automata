from __future__ import annotations

import time
from typing import Any

import numpy as np
from derivpy import Tick

from core.models import Signal
from strategies.base import Strategy


class MarkovRegimeStrategy(Strategy):
    """
    Estrategia basada en una cadena de Markov sobre estados de régimen.

    Estados:
        UP_LOW, UP_MED, UP_HIGH
        FLAT_LOW, FLAT_MED, FLAT_HIGH
        DOWN_LOW, DOWN_MED, DOWN_HIGH

    La matriz P estima transiciones:
        estado_t -> estado_t+horizon

    Por ejemplo, si horizon=10 ticks, la matriz intenta modelar:
        estado actual -> estado probable dentro de 10 ticks

    La señal se genera agregando probabilidades direccionales:
        P(UP) = P(UP_LOW) + P(UP_MED) + P(UP_HIGH)
        P(DOWN) = P(DOWN_LOW) + P(DOWN_MED) + P(DOWN_HIGH)

    Esto es más coherente con contratos CALL/PUT que decidir únicamente
    por el estado individual más probable.
    """

    STATES = (
        "UP_LOW",
        "UP_MED",
        "UP_HIGH",
        "FLAT_LOW",
        "FLAT_MED",
        "FLAT_HIGH",
        "DOWN_LOW",
        "DOWN_MED",
        "DOWN_HIGH",
    )

    def __init__(
        self,
        estimated_window: int = 2000,
        feature_window: int = 15,
        target_horizon: int = 10,
        prob_threshold: float = 0.50,
        direction_margin: float = 0.03,
        min_row_samples: int = 30,
        use_momentum_filter: bool = True,
        include_full_buffer_in_signal: bool = False,
        execution_delay: int = 0,  # ← NUEVO PARÁMETRO
    ):
        """
        Parámetros principales:

        estimated_window:
            Cantidad mínima de ticks que debe tener el buffer antes de
            emitir señales.

        feature_window:
            Ventana usada para calcular el estado actual:
            dirección, volatilidad y momentum.

        target_horizon:
            Horizonte inicial en ticks para estimar la matriz.
            Si get_signal recibe duration=10, se usa horizon=10.

        prob_threshold:
            Probabilidad direccional mínima para emitir señal.

        direction_margin:
            Ventaja mínima que debe tener UP sobre DOWN, o viceversa.
            Evita operar cuando P_UP y P_DOWN están demasiado parecidos.

        min_row_samples:
            Mínimo de transiciones históricas necesarias para confiar
            en la fila correspondiente al estado actual.

        use_momentum_filter:
            Si True, exige que el momentum acompañe la dirección.

        include_full_buffer_in_signal:
            Si True, Signal.data contiene el buffer completo.
            Si False, Signal.data contiene solo una ventana reciente.

        execution_delay:
            Ticks de desfase esperados entre la generación de la señal y
            la ejecución real de la orden. Compensa el slippage temporal.

            - 0 (por defecto): comportamiento clásico, predice a T+duration.
            - 1: predice a T+duration+1, compensando 1 tick de latencia.
            - 2+: para latencias mayores o feeds de alta frecuencia.

            El horizonte efectivo usado por la matriz será:
                effective_horizon = duration + execution_delay
        """

        self.estimated_window = estimated_window
        self.feature_window = feature_window
        self.target_horizon = target_horizon

        self.prob_threshold = prob_threshold
        self.direction_margin = direction_margin
        self.min_row_samples = min_row_samples
        self.use_momentum_filter = use_momentum_filter
        self.include_full_buffer_in_signal = include_full_buffer_in_signal
        self.execution_delay = int(execution_delay)

        # Estados y mapeos.
        self.states = list(self.STATES)
        self.n_states = len(self.states)
        self.state_to_index = {state: i for i, state in enumerate(self.states)}

        self.up_indices = np.array(
            [self.state_to_index[s] for s in self.states if s.startswith("UP")],
            dtype=int,
        )

        self.down_indices = np.array(
            [self.state_to_index[s] for s in self.states if s.startswith("DOWN")],
            dtype=int,
        )

        self.flat_indices = np.array(
            [self.state_to_index[s] for s in self.states if s.startswith("FLAT")],
            dtype=int,
        )

        # Matriz de transición.
        # Inicialización uniforme.
        self.P = np.full((self.n_states, self.n_states), 1.0 / self.n_states, dtype=float)

        # Conteos crudos.
        # Importantes para saber cuánta muestra sostiene cada probabilidad.
        self.transition_counts = np.zeros((self.n_states, self.n_states), dtype=float)

        # Estado interno de entrenamiento.
        self.fitted = False
        self.fitted_horizon: int | None = None

        # Control de actualización interna.
        self._last_seen_tick: Tick | None = None
        self._ticks_since_last_update = 0
        self._last_update_time = time.time()

        # Configuración interna avanzada.
        # Puedes exponerlos en __init__ si quieres ajustarlos directamente.
        self._min_fit_ticks = 500
        # self._reestimate_every_ticks = 200
        self._reestimate_every_ticks = 10
        # self._reestimate_every_seconds = 300.0
        self._reestimate_every_seconds = 0.0
        self._laplace_alpha = 1.0

        # Umbrales de volatilidad.
        # Idealmente deberían calibrarse por instrumento.
        self._low_vol_threshold = 0.00015
        self._med_vol_threshold = 0.0004

        # Cache para cálculo de momentum normalizado.
        self._x_moment_cache: dict[int, tuple[np.ndarray, float]] = {}

    # ------------------------------------------------------------------
    # Cálculo de características
    # ------------------------------------------------------------------

    def calculate_features(
        self,
        recent_ticks: list[Tick],
        window: int | None = None,
    ) -> tuple[str, float]:
        """
        Calcula el estado de régimen para una ventana de ticks.

        Devuelve:
            estado, momentum_normalizado

        Estado:
            UP_LOW, UP_MED, UP_HIGH,
            FLAT_LOW, FLAT_MED, FLAT_HIGH,
            DOWN_LOW, DOWN_MED, DOWN_HIGH

        Momentum normalizado:
            Pendiente de regresión sobre log(precio), aproximadamente
            expresada como cambio logarítmico por tick.

            Mantiene el signo:
                momentum > 0 -> pendiente alcista
                momentum < 0 -> pendiente bajista

            Pero al ser normalizado/logarítmico es más comparable entre
            instrumentos con precios de distinta escala.
        """

        w = self.feature_window if window is None else int(window)

        if not recent_ticks:
            return "FLAT_LOW", 0.0

        if not isinstance(recent_ticks, list):
            recent_ticks = list(recent_ticks)

        prices = np.asarray([t.quote for t in recent_ticks[-w:]], dtype=float)

        if len(prices) < w:
            return "FLAT_LOW", 0.0

        # Retornos simples para estimar volatilidad.
        returns = np.diff(prices) / prices[:-1]
        volatility = float(np.std(returns)) if returns.size > 0 else 0.0

        mean_price = float(np.mean(prices))
        current_price = float(prices[-1])

        # Umbral dinámico para dirección.
        # Si la volatilidad es muy baja, se usa un mínimo absoluto.
        directional_threshold = max(volatility * 2.0, 5e-5)

        if mean_price == 0:
            direction = "FLAT"
        elif current_price > mean_price * (1.0 + directional_threshold):
            direction = "UP"
        elif current_price < mean_price * (1.0 - directional_threshold):
            direction = "DOWN"
        else:
            direction = "FLAT"

        # Clasificación de volatilidad.
        if volatility < self._low_vol_threshold:
            vol_bucket = "LOW"
        elif volatility < self._med_vol_threshold:
            vol_bucket = "MED"
        else:
            vol_bucket = "HIGH"

        momentum = self._normalized_momentum(prices)

        state = f"{direction}_{vol_bucket}"
        return state, momentum

    def _normalized_momentum(self, prices: np.ndarray) -> float:
        """
        Momentum normalizado.

        Se calcula como la pendiente de una regresión lineal sobre
        log(precio). Esto lo hace más comparable entre instrumentos.

        Si por alguna razón hay precios no positivos, se usa una
        normalización alternativa basada en slope / precio medio.
        """

        if len(prices) < 2:
            return 0.0

        # Caso normal: precios positivos.
        if np.all(prices > 0):
            log_prices = np.log(prices)
            x_centered, denominator = self._get_x_moments(len(log_prices))

            if denominator == 0:
                return 0.0

            y_centered = log_prices - float(np.mean(log_prices))
            slope_log = float(np.dot(x_centered, y_centered) / denominator)
            return slope_log

        # Fallback si hubiera precios no positivos.
        x = np.arange(len(prices), dtype=float)
        slope = float(np.polyfit(x, prices, 1)[0])
        mean_price = float(np.mean(prices))

        if mean_price == 0:
            return 0.0

        return slope / mean_price

    def _get_x_moments(self, window: int) -> tuple[np.ndarray, float]:
        """
        Cache para el vector x centrado y su denominador.

        Esto evita recalcular la parte fija de la regresión lineal
        cada vez que se calcula momentum.
        """

        cached = self._x_moment_cache.get(window)
        if cached is not None:
            return cached

        x = np.arange(window, dtype=float)
        x_centered = x - float(np.mean(x))
        denominator = float(np.dot(x_centered, x_centered))

        self._x_moment_cache[window] = (x_centered, denominator)
        return self._x_moment_cache[window]

    # ------------------------------------------------------------------
    # Actualización de la matriz de transición
    # ------------------------------------------------------------------

    def update_transition_matrix(
        self,
        ticks_buffer: list[Tick],
        horizon: int | None = None,
        verbose: bool = False,
    ) -> bool:
        """
        Reestima la matriz P.

        La matriz modela:
            estado_t -> estado_t+horizon

        Si horizon=10, intenta capturar el estado del régimen 10 ticks
        después del estado actual.

        Devuelve True si la matriz pudo actualizarse.
        """

        horizon = self.target_horizon if horizon is None else int(horizon)

        if horizon <= 0:
            return False

        if not isinstance(ticks_buffer, list):
            ticks_buffer = list(ticks_buffer)

        n = len(ticks_buffer)

        min_required = max(
            self._min_fit_ticks,
            self.feature_window + horizon + 1,
        )

        if n < min_required:
            return False

        # Secuencia de estados.
        # Cada estado se calcula con una ventana completa de feature_window.
        states: list[str] = []

        for start in range(n - self.feature_window + 1):
            window = ticks_buffer[start : start + self.feature_window]
            state, _ = self.calculate_features(window, window=self.feature_window)
            states.append(state)

        # Conteos de transición estado_t -> estado_t+horizon.
        counts = np.zeros((self.n_states, self.n_states), dtype=float)

        for idx in range(len(states) - horizon):
            origin_state = states[idx]
            destination_state = states[idx + horizon]

            i = self.state_to_index[origin_state]
            j = self.state_to_index[destination_state]

            counts[i, j] += 1

        row_sums = counts.sum(axis=1, keepdims=True)

        self.transition_counts = counts

        # Suavizado Laplace para evitar filas con ceros.
        self.P = (counts + self._laplace_alpha) / (
            row_sums + self._laplace_alpha * self.n_states
        )

        self.fitted = True
        self.fitted_horizon = horizon
        # self.target_horizon = horizon

        self._ticks_since_last_update = 0
        self._last_update_time = time.time()

        if ticks_buffer:
            self._last_seen_tick = ticks_buffer[-1]

        if verbose:
            total_transitions = int(counts.sum())
            print(
                f"🔄 Matriz P actualizada | "
                f"horizon={horizon} | "
                f"transiciones={total_transitions}"
            )

        return True

    # ------------------------------------------------------------------
    # Señal
    # ------------------------------------------------------------------

    def get_signal(self, data, duration: int = 10, duration_unit: str = "t", **kwargs):
        """
        Genera una señal CALL, PUT o None.

        data:
            Buffer completo de ticks. La estrategia lo usa para actualizar
            la matriz si corresponde.

        duration:
            Duración del contrato en ticks. También se usa como horizonte
            para estimar la matriz P.

        duration_unit:
            Por ahora solo se soporta "t" (ticks).
        """

        if duration_unit != "t":
            raise NotImplementedError(
                "Esta versión está preparada para duración en ticks. "
                "Si necesitas segundos o minutos, debes convertirlos a ticks."
            )

        horizon = int(duration)

        if horizon <= 0:
            return None

        # NUEVO: límite máximo de duración aceptada (ajustable por instrumento)
        max_allowed_horizon = kwargs.get("max_horizon", 50)

        # NUEVO: horizonte efectivo que compensa el desfase de ejecución.
        # Con execution_delay=0, effective_horizon == horizon (comportamiento clásico).
        effective_horizon = horizon + self.execution_delay

        if not isinstance(data, list):
            ticks_buffer = list(data)
        else:
            ticks_buffer = data

        if len(ticks_buffer) < self.estimated_window:
            return None

        # Registrar avance del stream de ticks.
        self._register_new_ticks(ticks_buffer)

        # Actualización interna de la matriz.
        if self._should_update(effective_horizon):

            updated = self.update_transition_matrix(
                ticks_buffer,
                horizon=effective_horizon,
                verbose=kwargs.get("verbose", False),
            )

            if not updated:
                return None

        if len(ticks_buffer) < self.feature_window:
            return None

        # Estado actual.
        current_window = ticks_buffer[-self.feature_window :]
        current_state, momentum = self.calculate_features(current_window)

        state_idx = self.state_to_index[current_state]

        # Soporte muestral de la fila actual.
        row_samples = int(self.transition_counts[state_idx].sum())

        if row_samples < self.min_row_samples:
            return None

        probs = self.P[state_idx]

        # Probabilidades direccionales agregadas.
        p_up = float(probs[self.up_indices].sum())
        p_down = float(probs[self.down_indices].sum())
        p_flat = float(probs[self.flat_indices].sum())

        direction = None
        confidence = None

        # Señal CALL.
        if p_up >= self.prob_threshold and (p_up - p_down) >= self.direction_margin:
            if (not self.use_momentum_filter) or momentum > 0:
                direction = "CALL"
                confidence = p_up

        # Señal PUT.
        elif p_down >= self.prob_threshold and (p_down - p_up) >= self.direction_margin:
            if (not self.use_momentum_filter) or momentum < 0:
                direction = "PUT"
                confidence = p_down

        if direction is None:
            return None

        most_probable_next_state = self.states[int(np.argmax(probs))]

        signal_data = self._select_signal_data(ticks_buffer, effective_horizon)

        return Signal(
            direction=direction,
            data=signal_data,
            duration=duration,
            duration_unit=duration_unit,
            extras={
                "model": "markov_regime_strategy",
                "current_state": current_state,
                "most_probable_next_state": most_probable_next_state,
                "p_up": p_up,
                "p_down": p_down,
                "p_flat": p_flat,
                "confidence": confidence,
                "direction_margin": abs(p_up - p_down),
                "momentum": momentum,
                # Campos de duración claramente diferenciados:
                "contract_duration": horizon,            # ← duración del contrato (lo que va al API)
                "model_horizon": effective_horizon,      # ← horizonte usado por el modelo
                "execution_delay": self.execution_delay, # ← solo informativo
                "row_samples": row_samples,
                "fitted_horizon": self.fitted_horizon,   # ← debe ser igual a model_horizon
            },
        )

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _should_update(self, horizon: int) -> bool:
        """
        Determina si corresponde actualizar la matriz P.
        """

        if not self.fitted:
            return True

        if self.fitted_horizon != horizon:
            return True

        if (
            self._reestimate_every_ticks > 0
            and self._ticks_since_last_update >= self._reestimate_every_ticks
        ):
            return True

        if (
            self._reestimate_every_seconds > 0
            and (time.time() - self._last_update_time) >= self._reestimate_every_seconds
        ):
            return True

        return False

    def _register_new_ticks(self, ticks_buffer: list[Tick]) -> None:
        """
        Registra cuántos ticks nuevos llegaron desde la última actualización.

        Nota:
            Esta implementación usa epoch. Funciona bien para instrumentos
            con ticks de 1 segundo. Si usas un feed con varios ticks por
            mismo epoch, conviene usar un identificador único de tick.
        """

        if not ticks_buffer:
            return

        last_tick = ticks_buffer[-1]

        if self._last_seen_tick is None:
            self._last_seen_tick = last_tick
            return

        last_epoch = self._last_seen_tick.epoch
        new_ticks = 0

        # Recorremos desde el final porque los ticks nuevos deberían
        # estar al final del buffer.
        for tick in reversed(ticks_buffer):
            if tick.epoch <= last_epoch:
                break
            new_ticks += 1

        if new_ticks > 0:
            self._ticks_since_last_update += new_ticks
            self._last_seen_tick = last_tick

    def _select_signal_data(self, ticks_buffer: list[Tick], horizon: int) -> list[Tick]:
        """
        Determina qué datos incluir dentro de Signal.

        Importante:
            La estrategia ya usó el buffer completo para calcular la señal.
            Lo que se incluye aquí es metadata/auditoría.

        Si include_full_buffer_in_signal=True:
            devuelve el buffer completo.

        Si include_full_buffer_in_signal=False:
            devuelve una ventana reciente suficiente para auditar el estado
            actual y el horizonte usado.
        """

        if self.include_full_buffer_in_signal:
            return ticks_buffer

        length = self.feature_window + horizon

        if length <= 0:
            return ticks_buffer

        return ticks_buffer[-length:]

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    def diagnose(self) -> dict[str, Any]:
        """
        Devuelve un diagnóstico simple del estado interno del modelo.

        Útil para revisar:
            - si la matriz está entrenada;
            - cuántas muestras tiene cada estado;
            - probabilidades direccionales por estado.
        """

        row_samples = self.transition_counts.sum(axis=1)

        directional_summary = {}

        for state, idx in self.state_to_index.items():
            probs = self.P[idx]

            directional_summary[state] = {
                "p_up": float(probs[self.up_indices].sum()),
                "p_down": float(probs[self.down_indices].sum()),
                "p_flat": float(probs[self.flat_indices].sum()),
                "row_samples": int(row_samples[idx]),
            }

        return {
            "fitted": self.fitted,
            "fitted_horizon": self.fitted_horizon,
            "target_horizon": self.target_horizon,
            "feature_window": self.feature_window,
            "estimated_window": self.estimated_window,
            "ticks_since_last_update": self._ticks_since_last_update,
            "total_transitions": int(self.transition_counts.sum()),
            "row_samples": {
                state: int(row_samples[i]) for i, state in enumerate(self.states)
            },
            "directional_summary": directional_summary,
            "P": self.P.tolist(),
        }

    # ------------------------------------------------------------------
    # Compatibilidad con nombres anteriores
    # ------------------------------------------------------------------

    def actualizar_matriz(self, ticks_buffer, *args, **kwargs):
        """
        Wrapper de compatibilidad con el nombre anterior.

        Permite llamadas viejas como:
            strategy.actualizar_matriz(ticks)
            strategy.actualizar_matriz(ticks, True)

        Y llamadas nuevas como:
            strategy.actualizar_matriz(ticks, horizon=10)
            strategy.actualizar_matriz(ticks, 10, True)
        """

        verbose = kwargs.pop("verbose", False)
        horizon = kwargs.pop("horizon", None)

        if args:
            first = args[0]

            if isinstance(first, bool):
                verbose = first
            elif isinstance(first, int):
                horizon = first

            if len(args) > 1:
                second = args[1]

                if isinstance(second, bool):
                    verbose = second
                elif horizon is None and isinstance(second, int):
                    horizon = second

        return self.update_transition_matrix(
            ticks_buffer=ticks_buffer,
            horizon=horizon,
            verbose=verbose,
        )

    def calcular_caracteristicas(self, recent_ticks, window: int | None = None):
        """
        Wrapper de compatibilidad con el nombre anterior.
        """

        return self.calculate_features(recent_ticks, window)