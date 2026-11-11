"""
Motor de Trading Optimizado
Arquitectura: Producer-Consumer con manejo robusto de errores
"""

import asyncio
from collections import deque
from datetime import UTC, datetime
import json
from random import randint, shuffle
from typing import Optional
import logging

from derivpy import MarketDataClient, OpenContract, Tick, TradingClient
from derivpy.exceptions import DerivAPIError, DerivConnectionError
import numpy as np
from config.filters_config import load_filters_config
from config.strategies_config import load_strategies_config
from core.models import Signal
from filters.base import Filter
from filters.spread_trend_filter import SpreadTrendFilter
from filters.volatility_filter import VolatilityFilter
from filters.volatility_regime_filter import VolatilityRegimeFilter
from filters.volatility_regime_filter_v2 import VolatilityRegimeFilterV2
from strategies.experimental_strategy import ExperimentalStrategy
from strategies.markov_regime_strategy import MarkovRegimeStrategy
from strategies.momentum_pullback_adaptive_strategy import MomentumPullbackAdaptiveStrategy
from strategies.momentum_pullback_five_strategy import MomentumPullbackFive
from strategies.momentum_pullback_velocity_strategy import MomentumPullbackVelocityStrategy
from strategies.rise_fall_montecarlo_strategy import RiseFallMontecarloStrategy
from risk.recovery_matrix_calculator import RecoveryMatrixCalculator, TradeResult
from config.config import settings

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, account_id: str, symbol: str):
        strategy_params = load_strategies_config()
        filters_params = load_filters_config()

        self._is_running = False
        self._market_data_client = MarketDataClient()
        self._trading_client = TradingClient(
            app_id=settings.DERIV_APP_ID,
            token=settings.DERIV_TOKEN,
            account_id=account_id
        )
        self.symbol = symbol
        
        # Buffer de ticks (producer)
        self.ticks_buffer = deque(maxlen=2000)
        self.tick_queue = asyncio.Queue(maxsize=100)  # Cola para procesamiento
        
        # Estrategia y filtros
        # self.strategy = MarkovRegimeStrategy(
        #     target_horizon=5,
        #     prob_threshold=0.25
        # )
        # self.strategy = RiseFallMontecarloStrategy(
        #     simulations=5000,
        #     threshold=0.51
        # )
        # self.strategy = RiseFallMontecarloStrategy(strategy_params.rise_fall_montecarlo)
        # self.filter: Filter | None = VolatilityFilter(
        #     lookback=200
        # )
        # self.strategy = ExperimentalStrategy()
        # self.strategy = MomentumPullbackAdaptiveStrategy()
        self.strategy = MomentumPullbackVelocityStrategy(
            impulse_ticks=3,
            min_velocity_ratio=1.25
        )
        # self.strategy = MomentumPullbackFive()
        self.filters = [
            # VolatilityFilter(filters_params.volatility),
            # SpreadTrendFilter()

            VolatilityRegimeFilterV2(
                min_efficiency=0.30,
                spike_multiplier=2.5
                # window=25
            )

            # VolatilityRegimeFilter(min_efficiency=0.30)
        ]
        
        # Gestión de capital
        self.recovery_matrix_calculator = RecoveryMatrixCalculator(
            payout_pct=90.5, max_trades=3, target_wins=1,
            use_early_exit=True,
            early_exit_threshold=0.70, # Default            
            # early_exit_threshold=0.638, # Default            
            # early_exit_threshold=0.344, # Emular comportamiento previo errado
            use_trailing_stop=not True,
            trailing_stop_pct=0.40,           # Corta si cae 40% desde el máximo
            use_min_viability=False,           # Activa C
            min_viability_pct=0.10            # No arriesga si el trade es < 10% del capital restante
        )
        self.trade_results: list[TradeResult] = []
        self.capital: float = 20.0
        self.remaining_capital: float = self.capital
        self.take_profit: float = 10200.0  # TODO: Mover a config
        self.stop_loss: float = 9960.0     # TODO: Mover a config
        
        # Estado
        self.last_update: Optional[Tick] = None
        self._active_trades = set()  # Trades en ejecución
        self._max_concurrent_trades = 1
        self._concurrency_warning_logged = False
        
        # Circuit Breaker
        self._connection_failures = 0
        self._last_failure_time = Optional[datetime]

        self.initial_round_capital: float = 0.0  # <-- AGREGAR
        self.round_target_profit: float = 0.0    # <-- AGREGAR
        self.max_round_pnl: float = 0.0  # <-- NUEVO: Para el Trailing Stop (Solución B)

        # Cooldown por losses consecutivos
        self._use_cooldown = not True
        self._consecutive_losses = 0
        self._cooldown_threshold = 1  # N losses consecutivos para activar cooldown
        self._cooldown_omissions = 1  # M señales a omitir durante cooldown
        self._in_cooldown = False
        self._cooldown_signals_omitted = 0
        self._cooldown_until_epoch: Optional[float] = None  # Epoch hasta el cual simulamos señal
        self._seconds_per_tick = 0.0

    async def start(self) -> None:
        """Inicia el motor con arquitectura producer-consumer."""
        self._is_running = True
        logger.info(f"🚀 Iniciando TradingEngine para: {self.symbol}")
        
        # Conectar clientes
        await self._market_data_client.connect()
        await self._trading_client.connect()

        # 2. INICIALIZAR LA PRIMERA RONDA (CRUCIAL)
        if not await self._initialize_new_round():
            logger.critical("❌ No se pudo inicializar la primera ronda. Deteniendo motor.")
            self._is_running = False
            await self.stop()
            return
        
        # Cargar histórico inicial
        await self._load_initial_ticks()
        # self.strategy.actualizar_matriz(self.ticks_buffer)
        self.last_update = self.ticks_buffer[-1]
        
        # Iniciar tareas en paralelo
        try:
            # Producer: Recibe ticks y los pone en la cola
            producer_task = asyncio.create_task(self._tick_producer())
            
            # Consumer: Procesa señales y ejecuta trades
            consumer_task = asyncio.create_task(self._signal_consumer())
            
            # Esperar a que ambas terminen (o fallen)
            await asyncio.gather(producer_task, consumer_task)
            
        except asyncio.CancelledError:
            logger.info("⏹️ Motor detenido de forma segura")
        except Exception as e:
            logger.error(f"❌ Error crítico en el motor: {e}", exc_info=True)
        finally:
            await self.stop()

    async def _tick_producer(self) -> None:
        """
        Producer: Recibe ticks del WebSocket y los encola para procesamiento.
        NO bloquea el stream con operaciones de trading.
        """
        try:
            async for tick in self._market_data_client.stream_ticks(self.symbol):
                if not self._is_running:
                    break
                
                # Validar continuidad del stream
                if self.ticks_buffer:
                    last_tick = self.ticks_buffer[-1]
                    if tick.epoch - last_tick.epoch > 2:
                        logger.warning("⚠️ Gap detectado en stream, recargando histórico")
                        await self._load_initial_ticks()
                    else:
                        self.ticks_buffer.append(tick)
                        self.last_update = tick
                
                # Poner tick en cola para procesamiento (no bloquea)
                try:
                    self.tick_queue.put_nowait(tick)
                except asyncio.QueueFull:
                    logger.warning("⚠️ Cola de ticks llena, descartando tick antiguo")
                    # Descartar el más viejo si la cola está llena
                    try:
                        self.tick_queue.get_nowait()
                        self.tick_queue.put_nowait(tick)
                    except asyncio.QueueEmpty:
                        pass
                        
        except DerivConnectionError as e:
            logger.error(f"🔌 Desconexión del stream de market data: {e}")
            self._connection_failures += 1
            # self._last_failure_time = datetime.utcnow()
            self._last_failure_time = datetime.now(UTC)
            # El consumer detectará esto y pausará operaciones
        except Exception as e:
            logger.error(f"❌ Error en producer: {e}", exc_info=True)

    def _cleanup_trade_task(self, task: asyncio.Task) -> None:
        """
        Callback que se ejecuta automáticamente cuando un trade termina.
        Elimina la tarea del conjunto de activos de forma segura.
        """
        # Usamos discard en lugar de remove para evitar KeyError 
        # si la tarea ya fue eliminada por alguna otra razón.
        self._active_trades.discard(task)
        
        # Opcional: Loggear si la tarea terminó con una excepción no manejada
        if task.exception():
            logger.error(f"❌ Trade terminado con error no manejado: {task.exception()}")

    # async def _signal_consumer(self) -> None:
    #     """Consumer con validación de concurrencia y manejo de cooldown y monitor de salud."""
        
    #     # Contadores de flujo de datos
    #     ticks_processed = 0          # Ticks que entraron al sistema
    #     signals_attempted = 0        # Veces que se llamó a get_signal()
    #     signals_generated = 0        # Señales válidas retornadas (no None)
    #     signals_passed_filter = 0    # Señales que pasaron todos los filtros
    #     signals_executed = 0         # Señales que se convirtieron en trades
    #     signals_ommited = 0          # Señales omitidas por cooldown
        
    #     last_log_time = asyncio.get_event_loop().time()

    #     while self._is_running:
    #         try:
    #             try:
    #                 tick = await asyncio.wait_for(self.tick_queue.get(), timeout=1.0)
    #                 ticks_processed += 1
            
    #                 # Log de salud cada 60 segundos
    #                 current_time = asyncio.get_event_loop().time()
    #                 if current_time - last_log_time > 60.0:
    #                     logger.debug(
    #                         f"📊 Métricas del motor (últimos 60s):\n"
    #                         f"  ├─ Ticks procesados: {ticks_processed}\n"
    #                         f"  ├─ Evaluaciones intentadas: {signals_attempted}\n"
    #                         f"  ├─ Señales generadas: {signals_generated}\n"
    #                         f"  ├─ Señales post-filtro: {signals_passed_filter}\n"
    #                         f"  ├─ Señales omitidas: {signals_ommited}\n"
    #                         f"  ├─ Trades ejecutados: {signals_executed}\n"
    #                         f"  ├─ Cola pendiente: {self.tick_queue.qsize()}\n"
    #                         f"  └─ Trades activos: {len(self._active_trades)}"
    #                     )
    #                     # Resetear contadores para el próximo período
    #                     ticks_processed = signals_attempted = signals_generated = 0
    #                     signals_passed_filter = signals_executed = 0
    #                     last_log_time = current_time
    #             except asyncio.TimeoutError:
    #                 continue
                
    #             # 0. ACTUALIZAR MATRIZ (si aplica)
    #             # self.strategy.actualizar_matriz(self.ticks_buffer)
                
    #             # 1. GESTIONAR ESTADO DE LA RONDA
    #             if not await self._ensure_active_round():
    #                 continue
                
    #             # 2. MANEJAR COOLDOWN
    #             if self._in_cooldown:
    #                 # Verificar si la señal virtual actual ha expirado
    #                 if self._cooldown_until_epoch is not None:
    #                     if tick.epoch >= self._cooldown_until_epoch:
    #                         # La señal virtual expiró
    #                         self._cooldown_signals_omitted += 1
    #                         logger.info(
    #                             f"⏸️ Cooldown: Señal virtual {self._cooldown_signals_omitted}/"
    #                             f"{self._cooldown_omissions} completada (omitida)"
    #                         )
                            
    #                         # Verificar si completamos todas las omisiones
    #                         if self._cooldown_signals_omitted >= self._cooldown_omissions:
    #                             logger.info(
    #                                 f"✅ Cooldown finalizado: {self._cooldown_omissions} señales omitidas. "
    #                                 f"Reanudando operaciones normales."
    #                             )
    #                             self._in_cooldown = False
    #                             self._cooldown_signals_omitted = 0
    #                             self._cooldown_until_epoch = None
    #                             # Resetear contador de losses consecutivos
    #                             self._consecutive_losses = 0
    #                         else:
    #                             # Resetear para esperar la próxima señal
    #                             self._cooldown_until_epoch = None
                    
    #                 # Si estamos en cooldown pero no hay señal virtual activa, 
    #                 # buscar una nueva señal para omitir
    #                 if self._in_cooldown and self._cooldown_until_epoch is None:
    #                     # Validar concurrencia antes de buscar señal
    #                     if not self._can_execute_trade():
    #                         continue
                        
    #                     # Obtener señal
    #                     signals_attempted += 1 # INCREMENTAR "attempted"
    #                     signal = self.strategy.get_signal(
    #                         data=self.ticks_buffer, 
    #                         duration=5
    #                     )

    #                     if signal is None:
    #                         continue # No hay señal, no incrementamos "generated"

    #                     # Señal válida generada
    #                     signals_generated += 1

    #                     # Filtrar señal                        
    #                     if self.filter is not None:
    #                         signal = self.filter(signal)

    #                     if signal is None:
    #                         continue # Filtrada, no incrementamos "passed_filter"
                                                
    #                     # Señal pasó todos los filtros
    #                     signals_passed_filter += 1

    #                     # Señal omitida por cooldown
    #                     signals_ommited += 1

    #                     # Iniciar período de omisión para esta señal
    #                     last_tick: Tick = self.ticks_buffer[-1]
    #                     diff_secs = tick.epoch - last_tick.epoch
    #                     self._cooldown_until_epoch = tick.epoch + (signal.duration * diff_secs)
    #                     logger.info(
    #                         f"⏸️ Cooldown activo: Omitiendo señal {signal.direction} "
    #                         f"(duración: {signal.duration} ticks)"
    #                     )
                    
    #                 # En cooldown, no procesar nada más
    #                 continue
                
    #             # 3. VALIDAR CONCURRENCIA (fuera de cooldown)
    #             if not self._can_execute_trade():
    #                 continue
                
    #             # 4. Obtener señal
    #             signals_attempted += 1 # INCREMENTAR "attempted"
    #             signal = self.strategy.get_signal(
    #                 data=self.ticks_buffer, 
    #                 duration=5
    #             )

    #             if signal is None:
    #                 continue # No hay señal, no incrementamos "generated"

    #             # Señal válida generada
    #             signals_generated += 1

    #             # Filtrar señal
    #             if self.filter is not None:
    #                 signal = self.filter(signal)

    #             if signal is None:
    #                 continue # Filtrada, no incrementamos "passed_filter"

    #              # Señal pasó todos los filtros
    #             signals_passed_filter += 1

    #             # 5. Calcular monto
    #             bet = self.recovery_matrix_calculator.calculate_next_bet(
    #                 self.trade_results,
    #                 self.remaining_capital
    #             )
                
    #             # 6. Ejecutar
    #             trade_task = asyncio.create_task(self._execute_trade(signal, bet))
    #             trade_task.add_done_callback(self._cleanup_trade_task)
    #             self._active_trades.add(trade_task)

    #             # Trade lanzado (no necesariamente ejecutado aún)
    #             signals_executed += 1
                
    #         except Exception as e:
    #             logger.error(f"❌ Error en consumer: {e}", exc_info=True)
    #             await asyncio.sleep(1.0)

    async def _signal_consumer(self) -> None:
        """Consumer que maneja cooldown como omisión de señales individuales."""
        
        # Contadores de flujo de datos
        ticks_processed = 0
        signals_attempted = 0
        signals_generated = 0
        signals_passed_filter = 0
        signals_executed = 0
        signals_omitted = 0
        
        last_log_time = asyncio.get_event_loop().time()

        while self._is_running:
            try:
                try:
                    tick = await asyncio.wait_for(self.tick_queue.get(), timeout=1.0)
                    ticks_processed += 1
            
                    # Log de salud cada 60 segundos
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_log_time > 60.0:
                        logger.debug(
                            f"📊 Métricas del motor (últimos 60s):\n"
                            f"  ├─ Ticks procesados: {ticks_processed}\n"
                            f"  ├─ Evaluaciones intentadas: {signals_attempted}\n"
                            f"  ├─ Señales generadas: {signals_generated}\n"
                            f"  ├─ Señales post-filtro: {signals_passed_filter}\n"
                            f"  ├─ Señales omitidas (cooldown): {signals_omitted}\n"
                            f"  ├─ Trades ejecutados: {signals_executed}\n"
                            f"  ├─ Cola pendiente: {self.tick_queue.qsize()}\n"
                            f"  └─ Tasks activas: {len(self._active_trades)}"
                        )
                        ticks_processed = signals_attempted = signals_generated = 0
                        signals_passed_filter = signals_executed = signals_omitted = 0
                        last_log_time = current_time
                except asyncio.TimeoutError:
                    continue
                
                # 1. GESTIONAR ESTADO DE LA RONDA
                if not await self._ensure_active_round():
                    continue
                
                # 2. VALIDAR CONCURRENCIA
                if not self._can_execute_trade():
                    continue
                
                # 3. Obtener señal
                signals_attempted += 1
                signal = self.strategy.get_signal(
                    data=list(self.ticks_buffer),
                    duration=1
                )
                # signal = self.get_signal(duration=1)

                if signal is None:
                    continue

                signals_generated += 1

                # 4. Filtrar señal
                for filter in self.filters:
                    signal = filter(signal)

                    if signal is None:
                        break

                if signal is None:
                    continue

                signals_passed_filter += 1

                # 5. MANEJAR COOLDOWN: Si estamos en cooldown, omitir la señal
                if self._in_cooldown:
                    signals_omitted += 1
                    
                    # Crear task de "trade virtual" que espera la duración de esta señal
                    cooldown_task = asyncio.create_task(
                        self._execute_cooldown_for_signal(signal, self._seconds_per_tick)
                    )
                    cooldown_task.add_done_callback(self._cleanup_trade_task)
                    self._active_trades.add(cooldown_task)
                    
                    logger.info(
                        f"⏸️ Cooldown: Omitiendo señal {signal.direction} "
                        f"({self._cooldown_signals_omitted + 1}/{self._cooldown_omissions}) "
                        f"- Duración: {signal.duration} ticks"
                    )
                    continue
                
                # 6. Calcular monto
                bet = self.recovery_matrix_calculator.calculate_next_bet(
                    self.trade_results,
                    self.remaining_capital
                )
                
                # 7. Ejecutar trade
                trade_task = asyncio.create_task(self._execute_trade(signal, bet))
                trade_task.add_done_callback(self._cleanup_trade_task)
                self._active_trades.add(trade_task)
                signals_executed += 1
                
            except Exception as e:
                logger.error(f"❌ Error en consumer: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _execute_cooldown_for_signal(self, signal: Signal, seconds_per_tick: float = 1.0) -> None:
        """
        Ejecuta un período de cooldown para una señal específica.
        Espera el tiempo equivalente a la duración de ese trade.
        """
        try:
            # Calcular duración en segundos basada en la señal
            # Asumiendo ~1 segundo por tick (ajustar según tu caso)            
            cooldown_seconds = signal.duration * seconds_per_tick
            
            logger.debug(
                f"⏸️ Cooldown virtual: Esperando {cooldown_seconds:.0f}s "
                f"(señal de {signal.duration} ticks)"
            )
            
            # Esperar la duración del trade virtual
            await asyncio.sleep(cooldown_seconds)
            
            # Incrementar contador de señales omitidas
            self._cooldown_signals_omitted += 1
            
            # Verificar si completamos todas las omisiones
            if self._cooldown_signals_omitted >= self._cooldown_omissions:
                logger.info(
                    f"✅ Cooldown finalizado: {self._cooldown_omissions} señales omitidas. "
                    f"Reanudando operaciones normales."
                )
                self._in_cooldown = False
                self._cooldown_signals_omitted = 0
                self._consecutive_losses = 0
            
        except asyncio.CancelledError:
            logger.info("⏹️ Cooldown virtual cancelado")
        except Exception as e:
            logger.error(f"❌ Error durante cooldown virtual: {e}", exc_info=True)

    def _can_execute_trade(self) -> bool:
        """
        Valida si se puede ejecutar un trade en este momento.
        Función pura de consulta (sin efectos secundarios).
        """
        # 1. Verificar conexión
        if not self._trading_client.is_connected:
            logger.warning("⚠️ Cliente desconectado, omitiendo trade")
            return False

        # Experimental        
        # random_number = randint(1, 100)
        # if random_number <= 60:
        #     return False
        
        # 2. Verificar capital mínimo de la ronda
        if self.remaining_capital < self.recovery_matrix_calculator.min_bet:
            logger.warning(
                f"⚠️ Capital de ronda insuficiente: {self.remaining_capital:.2f} < "
                f"{self.recovery_matrix_calculator.min_bet}"
            )
            return False
        
        # 3. Verificar límite de concurrencia
        active_count = len(self._active_trades)
        if active_count >= self._max_concurrent_trades:
            if self._concurrency_warning_logged:
                logger.warning(
                    f"⚠️ Límite de concurrencia alcanzado: {active_count}/{self._max_concurrent_trades} trades activos"
                )
            return False
        
        return True

    # async def _execute_trade(self, signal: Signal, bet) -> None:
    #     """
    #     Ejecuta un trade individual con manejo robusto de errores.
    #     Esta función corre en background y no bloquea el stream.
    #     """
    #     try:
    #         logger.info(f"💼 Ejecutando trade: {signal.direction} @ {bet['next_bet']}")
            
    #         # 1. Verificar conexión antes de operar
    #         if not self._trading_client.is_connected:
    #             logger.error("🔌 No conectado, abortando trade")
    #             return
            
    #         # 2. Ejecutar trade
    #         br = await self._trading_client.trade(
    #             contract_type=signal.direction,
    #             underlying_symbol=self.symbol,
    #             amount=float(round(bet['next_bet'], 2)),
    #             duration=signal.duration,
    #             duration_unit=signal.duration_unit
    #         )
            
    #         # 3. Esperar expiración del contrato
    #         contract_expired = False
    #         # Timeout máximo de espera: duración del contrato + 30 segundos de buffer
    #         max_wait = signal.duration * 2 + 30  # Ajustar según duración en segundos

    #         try:
    #             async with asyncio.timeout(max_wait):
    #                 async for c in self._trading_client.watch_contract(
    #                     br.contract_id, 
    #                     stop_when_closed=False
    #                 ):
    #                     if c.is_expired and c.status != "open":
    #                         break
    #         except asyncio.TimeoutError:
    #             logger.error(f"⏱️ Timeout esperando expiración del contrato {br.contract_id}")
    #             # Forzar consulta del estado final
    #         except DerivConnectionError as e:
    #             logger.error(f"🔌 Conexión perdida durante watch_contract: {e}")
    #             await asyncio.sleep(5.0)  # Dar tiempo a que el ping_loop reconecte

    #         # async for c in self._trading_client.watch_contract(
    #         #     br.contract_id, 
    #         #     stop_when_closed=False
    #         # ):
    #         #     if c.is_expired and c.status != "open":
    #         #         break
            
    #         # 4. Obtener resultado final
    #         contract: OpenContract | None = None
    #         max_retries = 3

    #         for attempt in range(max_retries):
    #             try:
    #                 contract = await self._trading_client.get_contract(br.contract_id)
    #                 break  # Éxito, salimos del loop
    #             except DerivConnectionError:
    #                 if attempt < max_retries - 1:
    #                     logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} para obtener contrato")
    #                     await asyncio.sleep(3.0)
    #                 else:
    #                     logger.error(f"❌ No se pudo obtener resultado del contrato {br.contract_id}")
    #                     # Aquí decides: contar como loss, ignorar, o marcar como "unknown"
    #                     return

    #         if contract is None:
    #             logger.error(f"❌ Contrato {br.contract_id} sin resultado definitivo")
    #             return
    #         # contract = await self._trading_client.get_contract(br.contract_id)
            
    #         # 5. Actualizar estado
    #         result: TradeResult = 'win' if contract.profit > 0 else 'loss'
    #         self.trade_results.append(result)
    #         self.remaining_capital += contract.profit

    #         # ==========================================================
    #         # ACTUALIZAR CONTADOR DE LOSSES CONSECUTIVOS
    #         # ==========================================================
    #         if self._use_cooldown:
    #             if result == 'loss':                
    #                 self._consecutive_losses += 1
    #                 logger.info(f"📉 Loss consecutivo #{self._consecutive_losses}")
                    
    #                 # Verificar si debemos activar cooldown
    #                 if self._consecutive_losses >= self._cooldown_threshold:
    #                     logger.warning(
    #                         f"⚠️ {self._consecutive_losses} losses consecutivos detectados. "
    #                         f"Activando cooldown de {self._cooldown_omissions} señales."
    #                     )
    #                     self._in_cooldown = True
    #                     self._cooldown_signals_omitted = 0
    #                     self._cooldown_until_epoch = None
    #             else:
    #                 # Resetear contador en caso de win
    #                 if self._consecutive_losses > 0:
    #                     logger.info(f"✅ Win detectado. Resetando contador de losses consecutivos.")
    #                 self._consecutive_losses = 0

    #         # ==========================================================
    #         # 2. EVALUACIÓN DE EARLY EXIT (SOLUCIÓN A)
    #         # ==========================================================
    #         current_pnl = self.remaining_capital - self.initial_round_capital

    #         # 2. Actualizar el máximo histórico de la ronda (Crucial para Solución B)
    #         if current_pnl > self.max_round_pnl:
    #             self.max_round_pnl = current_pnl

    #         # 🔍 LOG DE DIAGNÓSTICO (Agrégalo temporalmente)
    #         # ratio = (current_pnl / self.round_target_profit) if self.round_target_profit > 0 else 0.0
    #         # logger.info(
    #         #     f"🔍 DEBUG Early Exit -> PnL Actual: ${current_pnl:.2f} | "
    #         #     f"Target PnL: ${self.round_target_profit:.2f} | "
    #         #     f"Ratio: {ratio:.2f} | Threshold: {self.recovery_matrix_calculator.early_exit_threshold}"
    #         # )
            
    #         logger.info(
    #             f"✅ Trade completado: {result} | "
    #             f"Profit: {contract.profit:.2f} | "
    #             f"Capital restante: {self.remaining_capital:.2f}"
    #         )

    #         # ==========================================================
    #         # EVALUACIÓN EN CASCADA DE PROTECCIÓN DE GANANCIAS
    #         # ==========================================================
            
    #         # Solución A: Early Exit por Umbral            
    #         if self.recovery_matrix_calculator.should_close_round_early(current_pnl, self.round_target_profit):
    #             logger.info(
    #                 f"🎯 Early Exit activado: PnL ${current_pnl:.2f} >= "
    #                 f"{self.recovery_matrix_calculator.early_exit_threshold*100:.0f}% del target (${self.round_target_profit:.2f})"
    #             )
    #             # Finalizamos la ronda con éxito anticipado
    #             await self._finalize_round("EARLY_EXIT_PROFIT_THRESHOLD")
    #             # Iniciamos una nueva inmediatamente
    #             await self._initialize_new_round()
                
    #             # IMPORTANTE: Salimos de la función. No guardamos en DB como trade normal 
    #             # (o sí, dependiendo de tu preferencia, pero la ronda ya se cerró arriba).
    #             return 

    #         # Solución B: Trailing Stop de Ronda
    #         if self.recovery_matrix_calculator.should_activate_trailing_stop(current_pnl, self.max_round_pnl):
    #             logger.warning(
    #                 f"📉 Trailing Stop Activado: PnL cayó de ${self.max_round_pnl:.2f} a ${current_pnl:.2f} "
    #                 f"({((self.max_round_pnl - current_pnl)/self.max_round_pnl)*100:.1f}% de caída)"
    #             )
    #             await self._finalize_round("TRAILING_STOP_ACTIVATED")
    #             await self._initialize_new_round()
    #             return 
            
    #         # logger.info(
    #         #     f"✅ Trade completado: {result} | "
    #         #     f"Profit: {contract.profit:.2f} | "
    #         #     f"Capital restante: {self.remaining_capital:.2f}"
    #         # )
            
    #         # 6. Guardar en CouchDB (auditoría)
    #         # await self._save_trade_to_db(contract, signal, bet, result)
            
    #         # 7. Verificar Take Profit / Stop Loss
    #         balance = await self._trading_client.get_balance()
    #         if balance.balance >= self.take_profit:
    #             logger.info("🎯 Take Profit alcanzado")
    #             self._is_running = False
    #         elif balance.balance <= self.stop_loss:
    #             logger.info("🛑 Stop Loss alcanzado")
    #             self._is_running = False
    #         elif balance.balance < 0.35:
    #             logger.error("💸 Balance insuficiente")
    #             self._is_running = False
                
    #     except DerivConnectionError as e:
    #         logger.error(f"🔌 Error de conexión durante trade: {e}")
    #         self._connection_failures += 1
    #         # self._last_failure_time = datetime.utcnow()
    #         self._last_failure_time = datetime.now(UTC)
    #         # NO reiniciar la ronda, el trade no se ejecutó
            
    #     except DerivAPIError as e:
    #         logger.error(f"❌ Error de API durante trade: {e}")
    #         # Aquí puedes decidir si contar como loss o no
    #         # Por seguridad, no contar si fue error de infraestructura
            
    #     except Exception as e:
    #         logger.error(f"❌ Error inesperado en trade: {e}", exc_info=True)

    async def _execute_trade(self, signal: Signal, bet) -> None:
        """Ejecuta un trade y maneja la activación de cooldown si es necesario."""
        try:
            logger.info(f"💼 Ejecutando trade: {signal.direction} @ {bet['next_bet']}")
            
            if not self._trading_client.is_connected:
                logger.error("🔌 No conectado, abortando trade")
                return
            
            # Ejecutar trade
            br = await self._trading_client.trade(
                contract_type=signal.direction,
                underlying_symbol=self.symbol,
                amount=float(round(bet['next_bet'], 2)),
                duration=signal.duration,
                duration_unit=signal.duration_unit
            )

            # LOG Temporal
            print(json.dumps(signal.extras,indent=4))
            
            # Esperar expiración del contrato
            max_wait = signal.duration * 2 + 30

            try:
                async with asyncio.timeout(max_wait):
                    async for c in self._trading_client.watch_contract(
                        br.contract_id, 
                        stop_when_closed=False
                    ):
                        if c.is_expired and c.status != "open":
                            break
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout esperando expiración del contrato {br.contract_id}")
            except DerivConnectionError as e:
                logger.error(f"🔌 Conexión perdida durante watch_contract: {e}")
                await asyncio.sleep(5.0)

            # Obtener resultado final
            contract: OpenContract | None = None
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    contract = await self._trading_client.get_contract(br.contract_id)
                    await asyncio.sleep(5.0)
                    break
                except DerivConnectionError:
                    if attempt < max_retries - 1:
                        logger.warning(f"🔄 Retry {attempt + 1}/{max_retries} para obtener contrato")
                    else:
                        logger.error(f"❌ No se pudo obtener resultado del contrato {br.contract_id}")
                        return

            if contract is None:
                logger.error(f"❌ Contrato {br.contract_id} sin resultado definitivo")
                return
            
            # Actualizar estado
            result: TradeResult = 'win' if contract.profit > 0 else 'loss'
            self.trade_results.append(result)
            self.remaining_capital += contract.profit

            # ==========================================================
            # ACTUALIZAR CONTADOR DE LOSSES CONSECUTIVOS
            # ==========================================================
            should_activate_cooldown = False
            
            if self._use_cooldown:
                if result == 'loss':                
                    self._consecutive_losses += 1
                    logger.info(f"📉 Loss consecutivo #{self._consecutive_losses}")
                    
                    if self._consecutive_losses >= self._cooldown_threshold:
                        logger.warning(
                            f"⚠️ {self._consecutive_losses} losses consecutivos detectados. "
                            f"Activando cooldown de {self._cooldown_omissions} señales."
                        )
                        should_activate_cooldown = True
                else:
                    if self._consecutive_losses > 0:
                        logger.info(f"✅ Win detectado. Resetando contador de losses consecutivos.")
                    self._consecutive_losses = 0

            # Calcular PnL actual
            current_pnl = self.remaining_capital - self.initial_round_capital
            if current_pnl > self.max_round_pnl:
                self.max_round_pnl = current_pnl
            
            logger.info(
                f"✅ Trade completado: {result} | "
                f"Profit: {contract.profit:.2f} | "
                f"Capital restante: {self.remaining_capital:.2f}"
            )

            # Evaluación de Early Exit
            if self.recovery_matrix_calculator.should_close_round_early(current_pnl, self.round_target_profit):
                logger.info(
                    f"🎯 Early Exit activado: PnL ${current_pnl:.2f} >= "
                    f"{self.recovery_matrix_calculator.early_exit_threshold*100:.0f}% del target"
                )
                await self._finalize_round("EARLY_EXIT_PROFIT_THRESHOLD")
                await self._initialize_new_round()
                return 

            # Trailing Stop
            if self.recovery_matrix_calculator.should_activate_trailing_stop(current_pnl, self.max_round_pnl):
                logger.warning(
                    f"📉 Trailing Stop Activado: PnL cayó de ${self.max_round_pnl:.2f} a ${current_pnl:.2f}"
                )
                await self._finalize_round("TRAILING_STOP_ACTIVATED")
                await self._initialize_new_round()
                return 
            
            # Verificar Take Profit / Stop Loss
            balance = await self._trading_client.get_balance()
            if balance.balance >= self.take_profit:
                logger.info("🎯 Take Profit alcanzado")
                self._is_running = False
            elif balance.balance <= self.stop_loss:
                logger.info("🛑 Stop Loss alcanzado")
                self._is_running = False
            elif balance.balance < 0.35:
                logger.error("💸 Balance insuficiente")
                self._is_running = False
            
            # ==========================================================
            # ACTIVAR COOLDOWN (marca el flag, pero no crea task todavía)
            # ==========================================================
            if should_activate_cooldown:
                self._in_cooldown = True
                self._cooldown_signals_omitted = 0
                logger.info(
                    f"⏸️ Cooldown activado: Próximas {self._cooldown_omissions} señales serán omitidas"
                )
                
        except DerivConnectionError as e:
            logger.error(f"🔌 Error de conexión durante trade: {e}")
            self._connection_failures += 1
            self._last_failure_time = datetime.now(UTC)
            
        except DerivAPIError as e:
            logger.error(f"❌ Error de API durante trade: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error inesperado en trade: {e}", exc_info=True)

    def _is_circuit_breaker_open(self) -> bool:
        """Verifica si el circuit breaker está activo."""
        if self._connection_failures < 3:
            return False
        
        if self._last_failure_time:
            elapsed = (datetime.utcnow() - self._last_failure_time).total_seconds()
            if elapsed < 60:  # 60 segundos de cooldown
                return True
            else:
                # Resetear después del cooldown
                self._connection_failures = 0
        
        return False

    async def _save_trade_to_db(self, contract, signal, bet, result: TradeResult) -> None:
        """Guarda el trade en CouchDB para auditoría."""
        try:
            # TODO: Implementar guardado en CouchDB
            trade_doc = {
                "type": "trade",
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": self.symbol,
                "direction": signal.direction,
                "amount": bet['next_bet'],
                "duration": signal.duration,
                "result": result,
                "profit": contract.profit,
                "remaining_capital": self.remaining_capital,
                "round_wins": sum(1 for r in self.trade_results if r == 'win'),
                "round_losses": sum(1 for r in self.trade_results if r == 'loss'),
            }
            # await self.db.save(trade_doc)
            logger.debug(f"💾 Trade guardado en DB: {trade_doc}")
        except Exception as e:
            logger.error(f"❌ Error guardando trade en DB: {e}")

    async def _load_initial_ticks(self, verbose: bool = False) -> None:
        """Carga histórico de ticks con manejo de errores."""
        try:
            ticks_2, _ = await self._market_data_client.ticks_history(
                symbol=self.symbol,
                count=1000
            )
            ticks_1, _ = await self._market_data_client.ticks_history(
                symbol=self.symbol,
                count=1000,
                end=ticks_2[0].epoch
            )
            
            # Combinar y ordenar
            ticks = sorted(ticks_1 + ticks_2, key=lambda t: t.epoch)
            ticks = list({t.epoch: t for t in ticks}.values())

            self._seconds_per_tick = ticks[-1].epoch - ticks[-2].epoch
            self.ticks_buffer.clear()
            self.ticks_buffer.extend(ticks)

            if verbose:
                logger.info(f"✅ Cargados {len(ticks)} ticks históricos")
                
        except Exception as e:
            logger.error(f"❌ Error cargando histórico: {e}", exc_info=True)

    async def _ensure_active_round(self) -> bool:
        """
        Gestiona el ciclo de vida de la ronda. 
        Si la ronda actual es inviable, la finaliza e intenta iniciar una nueva.
        Retorna True si se puede operar, False si el motor debe detenerse globalmente.
        """
        # 1. Verificar el estado matemático de la ronda actual
        bet_info = self.recovery_matrix_calculator.calculate_next_bet(
            self.trade_results, 
            self.remaining_capital
        )

        # 2. Si la ronda NO está activa (por losses, max trades, o capital insuficiente)
        if bet_info["status"] != "ACTIVE":
            logger.warning(f"🔄 Ronda finalizada. Motivo: {bet_info['status']}. ")

            # Reiniciar contador de cooldown
            self._consecutive_losses = 0
            
            # Finalizar y registrar la ronda antes de resetear
            await self._finalize_round(bet_info["status"])

            # Experimental: Reiniciar conexion en TradingClient si se pierde una ronda
            await self._trading_client.disconnect()
            await asyncio.sleep(2.0)
            await self._trading_client.connect()
            
            # Intentar iniciar una nueva ronda
            can_continue = await self._initialize_new_round()
            if not can_continue:
                return False # El balance global no permite seguir

        # 3. Verificación final de seguridad (doble check)
        final_bet_info = self.recovery_matrix_calculator.calculate_next_bet(
            self.trade_results, 
            self.remaining_capital
        )
        
        if final_bet_info["status"] != "ACTIVE":
            logger.critical("🛑 Imposible iniciar nueva ronda. Capital global insuficiente.")
            self._is_running = False
            return False

        return True

    async def _initialize_new_round(self) -> bool:
        """
        Consulta el balance real y asigna capital para una nueva ronda.
        """
        try:
            # Obtener balance real del broker (fuente de la verdad)
            balance_resp = await self._trading_client.get_balance()
            current_total_balance = float(balance_resp.balance)
            
            # --- CONDICIONES GLOBALES DE PARADA (Hard Stops) ---
            if current_total_balance <= self.stop_loss:
                logger.critical(f"🛑 STOP LOSS GLOBAL ALCANZADO. Balance: {current_total_balance}")
                self._is_running = False
                return False
                
            if current_total_balance >= self.take_profit:
                logger.info(f"🎯 TAKE PROFIT GLOBAL ALCANZADO. Balance: {current_total_balance}")
                self._is_running = False
                return False
                
            if current_total_balance < self.recovery_matrix_calculator.min_bet:
                logger.critical(f"💀 BALANCE CRÍTICO. Insuficiente para el trade mínimo ({current_total_balance} < 0.35)")
                self._is_running = False
                return False

            # --- ASIGNACIÓN DE CAPITAL PARA LA NUEVA RONDA ---
            # Tu lógica: $15 por ronda si es posible. Si quedan $14, se asignan $14.
            round_allocation = 20.0  # Puedes mover esto a `config.py`
            new_round_capital = min(round_allocation, current_total_balance)

            # 1. Guardar el capital con el que inicia la ronda
            self.initial_round_capital = new_round_capital

            self.remaining_capital = new_round_capital
            self.trade_results = [] # Resetear historial de la ronda

            # Resetear el máximo de la nueva ronda
            self.max_round_pnl = 0.0  # <-- NUEVO

            # 2. Calcular el target de la ronda completa basado en matrix[0, 0]
            self.round_target_profit = self.recovery_matrix_calculator.get_round_target_profit(
                self.initial_round_capital
            )

            # Log de diagnóstico
            logger.info(
                f"✅ Nueva ronda iniciada. "
                f"Capital asignado: ${self.remaining_capital:.2f} | "
                f"Target PnL: ${self.round_target_profit:.2f} | "
                f"Matrix[0,0]: {self.recovery_matrix_calculator.matrix[0, 0]:.4f} | "
                f"(Balance Total: ${current_total_balance:.2f})"
            )
            
            # DEPRECATED: 2. Estimar el target de la ronda (Payout esperado del primer trade)
            # first_bet_info = self.recovery_matrix_calculator.calculate_next_bet([], self.initial_round_capital)
            # self.round_target_profit = first_bet_info['next_bet'] * self.recovery_matrix_calculator.payout            
            # logger.info(f"✅ Nueva ronda iniciada. Capital asignado: ${self.remaining_capital:.2f} (Balance Total: ${current_total_balance:.2f})")

            return True
            
        except Exception as e:
            logger.error(f"❌ Error al consultar balance para nueva ronda: {e}")
            return False

    async def _finalize_round(self, reason: str) -> None:
        """
        Registra el cierre de la ronda en CouchDB para auditoría y análisis de riesgo.
        """
        wins = sum(1 for r in self.trade_results if r == 'win')
        losses = sum(1 for r in self.trade_results if r == 'loss')
        net_round_profit = self.remaining_capital - self.capital # Aproximación si no se recargó aún
        
        round_doc = {
            "type": "round_summary",
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": self.symbol,
            "status": reason, # ej: "MAX_TRADES_REACHED", "INSUFFICIENT_CAPITAL"
            "starting_capital": self.capital,
            "ending_capital": self.remaining_capital,
            "round_profit": net_round_profit,
            "total_trades": wins + losses,
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if (wins + losses) > 0 else 0.0
        }
        
        try:
            # await self.db.save(round_doc) # Descomenta cuando tengas CouchDB activo
            logger.info(f"💾 Ronda registrada en DB: {wins}W / {losses}L | PnL: {net_round_profit:.2f}")
        except Exception as e:
            logger.error(f"Error guardando ronda en DB: {e}")

    async def stop(self) -> None:
        """Detiene el motor y limpia recursos."""
        if self._is_running:
            self._is_running = False
            logger.info("⏹️ Deteniendo TradingEngine...")
            
            # Esperar a que los trades activos terminen (con timeout)
            if self._active_trades:
                logger.info(f"⏳ Esperando {len(self._active_trades)} trades activos...")
                await asyncio.wait(self._active_trades, timeout=30.0)
            
            # Desconectar clientes
            await self._market_data_client.disconnect()
            await self._trading_client.disconnect()
            logger.info("✅ Motor detenido correctamente")

    def get_signal(self, duration: int, lookback: int = 15):
        if len(self.ticks_buffer) < lookback:
            return None
        
        ticks: list[Tick] = list(self.ticks_buffer)[-lookback:]
        high_tick = max(ticks, key=lambda t: t.quote)
        low_tick = min(ticks, key=lambda t: t.quote)
        pip_size = ticks[-1].pip_size
        call_signals: list[Signal] = []
        put_signals: list[Signal] = []
        count_none: int = 0
        step = 1 / (10 ** pip_size)

        for quote in np.arange(low_tick.quote, high_tick.quote + step, step):
            ticks_buffer = self.ticks_buffer.copy()
            tick: Tick = ticks_buffer[-1]
            t = Tick(
                epoch=tick.epoch + self._seconds_per_tick, 
                quote=quote,
                pip_size=pip_size
            )
            # tick.epoch += self._seconds_per_tick
            # tick.quote = quote
            ticks_buffer.append(t)
            
            signal = self.strategy.get_signal(
                data=ticks_buffer,
                duration=duration,
                duration_unit="t"
            )

            if signal is None:
                count_none += 1
            else:
                signal.data = self.ticks_buffer

                if signal.direction == "CALL":
                    call_signals.append(signal)
                    # put_signals.append(signal)
                else:
                    put_signals.append(signal)
                    # call_signals.append(signal)

        len_call_signals: int = len(call_signals)
        len_put_signals: int = len(put_signals)
        total = len_call_signals + len_put_signals # + count_none

        if total == 0:
            return None

        max_threshold = 1.0
        min_threshold = 0.51
        if min_threshold <= (len_call_signals / total) <= max_threshold:
            return call_signals[-1]
            # return put_signals[-1]        
        if min_threshold <= (len_put_signals / total) <= max_threshold:
            return put_signals[-1]
            # return call_signals[-1]
                
        return None

