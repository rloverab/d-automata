"""
Motor de Trading Optimizado
Arquitectura: Producer-Consumer con manejo robusto de errores
"""

import asyncio
from collections import deque
from datetime import UTC, datetime
import json
import math
from typing import Optional
import logging

from derivpy import MarketDataClient, OpenContract, Tick, TradingClient
from derivpy.exceptions import DerivAPIError, DerivConnectionError
import numpy as np
from config.filters_config import load_filters_config
from config.strategies_config import load_strategies_config
from core.models import Signal
from filters.volatility_regime_filter_v2 import VolatilityRegimeFilterV2
from optimizer.duration_optimizer import DurationOptimizer
from strategies.momentum_pullback_velocity_strategy import MomentumPullbackVelocityStrategy
from risk.recovery_calculator_closed_form import RecoveryCalculatorClosedForm as calc_bet, TradeResult
from config.config import settings
from strategies.packs_strategy import VolatilitySqueezeConfig, VolatilitySqueezeStrategy, ZScoreReversionConfig, ZScoreReversionStrategy

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, account_id: str, symbol: str):
        self._is_running = False
        self._market_data_client = MarketDataClient()
        self._trading_client = TradingClient(
            app_id=settings.DERIV_APP_ID,
            token=settings.DERIV_TOKEN,
            account_id=account_id
        )
        self.symbol = symbol
        
        # Buffer de ticks (producer)
        self.ticks_buffer = deque(maxlen=3600)
        self.tick_queue = asyncio.Queue(maxsize=100)  # Cola para procesamiento
        
        # Estrategia y filtros        
        # self.strategy = MomentumPullbackVelocityStrategy(
        #     impulse_ticks=1, #3,
        #     min_velocity_ratio=1.1 #1.25
        # )
        # self.strategy = ZScoreReversionStrategy(ZScoreReversionConfig())
        self.strategy = VolatilitySqueezeStrategy(VolatilitySqueezeConfig())
        # self.strategy = MomentumPullbackFive()
        self.filters = [
            # VolatilityFilter(filters_params.volatility),
            # SpreadTrendFilter()

            # VolatilityRegimeFilterV2(
            #     # min_efficiency=0.30,
            #     spike_multiplier=2.5,
            #     window=30
            # )

            # VolatilityRegimeFilter(min_efficiency=0.30)
        ]
        self.duration = 0
        
        # Gestión de capital
        self.payout_pct = 91.5
        self.max_trades = 2
        self.target_wins = 1
        self.min_bet = 0.35
        self.trade_results = deque(maxlen=self.max_trades)        
        self.capital: float = 20.0
        self.remaining_capital: float = self.capital
        self.take_profit: float = 10100.0  # TODO: Mover a config
        self.stop_loss: float = 9960.0     # TODO: Mover a config

        # ✅ AGREGAR: Parámetros de protección (antes estaban en la instancia)
        self.use_early_exit = False
        self.early_exit_threshold = 0.70
        self.use_trailing_stop = False
        self.trailing_stop_pct = 0.50
        self.round_target_profit: float = 0.0
        
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

        # ✅ NUEVO: Optimizador de duración
        self.duration_optimizer = DurationOptimizer(
            strategy=self.strategy,
            filters=self.filters,
            duration_range=range(1, 11),
            min_win_rate=0.55,
            max_trades_per_round=self.max_trades,
            payout_pct=self.payout_pct,
            min_sample_size=5,
            hysteresis_margin=0.025#0.10
        )
        
        # Configuración del optimizador
        self._optimizer_enabled = True
        self._optimizer_interval_seconds = 180 #300  # Cada 5 minutos
        self._optimizer_task: Optional[asyncio.Task] = None
        self._last_optimizer_run: float = 0

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

            # ✅ NUEVO: Tarea de optimización en background
            if self._optimizer_enabled:
                self._optimizer_task = asyncio.create_task(self._duration_optimizer_loop())
            
            # Esperar a que ambas terminen (o fallen)
            tasks = [producer_task, consumer_task]
            if self._optimizer_task:
                tasks.append(self._optimizer_task)
            
            await asyncio.gather(*tasks)
            
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

                # 3.5. Verificar la duración
                if self.duration == 0:
                    continue

                signal = self.strategy.get_signal(
                    data=list(self.ticks_buffer),
                    duration=self.duration
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
                bet = calc_bet.calculate_next_bet(
                    history=list(self.trade_results),
                    current_round_capital=self.remaining_capital,
                    payout_pct=self.payout_pct,
                    max_trades=self.max_trades,
                    target_wins=self.target_wins
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
        
        # 2. Verificar capital mínimo de la ronda
        if self.remaining_capital < self.min_bet:
            logger.warning(
                f"⚠️ Capital de ronda insuficiente: {self.remaining_capital:.2f} < "
                f"{self.min_bet}"
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
            # print(json.dumps(signal.extras,indent=4))
            
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
            if self.use_early_exit:
                if calc_bet.should_close_round_early(
                    current_pnl, 
                    self.round_target_profit,
                    self.early_exit_threshold  # ← Usar atributo del engine
                ):
                    logger.info(
                        f"🎯 Early Exit activado: PnL ${current_pnl:.2f} >= "
                        f"{self.early_exit_threshold*100:.0f}% del target"  # ← CORREGIDO
                    )
                    await self._finalize_round("EARLY_EXIT_PROFIT_THRESHOLD")
                    await self._initialize_new_round()
                    return
            # if calc_bet.should_close_round_early(current_pnl, self.round_target_profit):
            #     logger.info(
            #         f"🎯 Early Exit activado: PnL ${current_pnl:.2f} >= "
            #         f"{self.recovery_matrix_calculator.early_exit_threshold*100:.0f}% del target"
            #     )
            #     await self._finalize_round("EARLY_EXIT_PROFIT_THRESHOLD")
            #     await self._initialize_new_round()
            #     return 

            # Trailing Stop
            if calc_bet.should_activate_trailing_stop(current_pnl, self.max_round_pnl):
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

    async def _load_initial_ticks(self, count: int = 3600, verbose: bool = False) -> None:
        """Carga histórico de ticks con manejo de errores."""
        try:
            all_ticks = []
            if count > 1000:
                batches = math.ceil(count / 1000)
                end = "latest"
                for i in range(batches):
                    batch_size = min(1000, count - len(all_ticks))
                    ts, _ = await self._market_data_client.ticks_history(symbol=self.symbol, count=batch_size, end=end)
                    if not ts:
                        break
                    all_ticks = ts + all_ticks
                    end = ts[0].epoch - 1
                    await asyncio.sleep(0.1)                
            else:
                all_ticks, _ = await self._market_data_client.ticks_history(symbol=self.symbol, count=count)

            sorted_ticks = sorted(all_ticks, key=lambda t: t.epoch)
            self.ticks_buffer.clear()
            self.ticks_buffer.extend(sorted_ticks)
            
            if len(all_ticks) > 0 and verbose:
                logger.info(f"✅ Cargados {len(all_ticks)} ticks históricos")

        except Exception as e:
            pass        

    async def _ensure_active_round(self) -> bool:
        """
        Gestiona el ciclo de vida de la ronda. 
        Si la ronda actual es inviable, la finaliza e intenta iniciar una nueva.
        Retorna True si se puede operar, False si el motor debe detenerse globalmente.
        """
        # 1. Verificar el estado matemático de la ronda actual
        bet_info = calc_bet.calculate_next_bet(
            history=list(self.trade_results),
            current_round_capital=self.remaining_capital,
            payout_pct=self.payout_pct,
            max_trades=self.max_trades,
            target_wins=self.target_wins
        )

        # 2. Si la ronda NO está activa (por losses, max trades, o capital insuficiente)
        if bet_info["status"] != "ACTIVE":
            logger.warning(f"🔄 Ronda finalizada. Motivo: {bet_info['status']}. ")

            # Reiniciar contador de cooldown
            self._consecutive_losses = 0
            
            # Finalizar y registrar la ronda antes de resetear
            await self._finalize_round(bet_info["status"])
            
            # Intentar iniciar una nueva ronda
            can_continue = await self._initialize_new_round()
            if not can_continue:
                return False # El balance global no permite seguir

        # 3. Verificación final de seguridad (doble check)
        final_bet_info = calc_bet.calculate_next_bet(
            history=list(self.trade_results),
            current_round_capital=self.remaining_capital,
            payout_pct=self.payout_pct,
            max_trades=self.max_trades,
            target_wins=self.target_wins
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
                
            # if current_total_balance < self.recovery_matrix_calculator.min_bet:
            if current_total_balance < self.min_bet:
                logger.critical(f"💀 BALANCE CRÍTICO. Insuficiente para el trade mínimo ({current_total_balance} < 0.35)")
                self._is_running = False
                return False

            # --- ASIGNACIÓN DE CAPITAL PARA LA NUEVA RONDA ---
            # Tu lógica: $15 por ronda si es posible. Si quedan $14, se asignan $14.
            round_allocation = 15.0  # Puedes mover esto a `config.py`
            new_round_capital = min(round_allocation, current_total_balance)

            # 1. Guardar el capital con el que inicia la ronda
            self.initial_round_capital = new_round_capital

            self.remaining_capital = new_round_capital
            self.trade_results.clear() # Resetear historial de la ronda

            # Resetear el máximo de la nueva ronda
            self.max_round_pnl = 0.0  # <-- NUEVO

            # 2. Calcular el target de la ronda completa basado en matrix[0, 0]
            self.round_target_profit = calc_bet.get_round_target_profit(
                initial_round_capital=self.initial_round_capital,
                payout_pct=self.payout_pct,
                max_trades=self.max_trades,
                target_wins=self.target_wins
            )

            # Log de diagnóstico
            logger.info(
                f"✅ Nueva ronda iniciada. "
                f"Capital asignado: ${self.remaining_capital:.2f} | "
                f"Target PnL: ${self.round_target_profit:.2f} | "
                f"(Balance Total: ${current_total_balance:.2f})"
            )
            
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
        # net_round_profit = self.remaining_capital - self.capital # Aproximación si no se recargó aún
        # ✅ CORRECCIÓN: Usar initial_round_capital en lugar de capital fijo
        net_round_profit = self.remaining_capital - self.initial_round_capital
        
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

    async def _duration_optimizer_loop(self) -> None:
        """
        Tarea en background que ejecuta el optimizador periódicamente.
        NO interfiere con producer/consumer porque usa copias del buffer.
        """
        logger.info(f"🔬 Optimizador de duración iniciado | Intervalo: {self._optimizer_interval_seconds}s")
        
        # Esperar inicial antes de primera optimización
        await asyncio.sleep(60)
        
        while self._is_running:
            try:
                # Verificar intervalo
                current_time = asyncio.get_event_loop().time()
                if current_time - self._last_optimizer_run < self._optimizer_interval_seconds:
                    await asyncio.sleep(10)
                    continue
                
                self._last_optimizer_run = current_time
                
                # ✅ CRÍTICO: Copiar buffer para evitar race conditions
                ticks_copy = list(self.ticks_buffer)
                
                if len(ticks_copy) < 500:
                    logger.debug("⏸️ Buffer insuficiente para optimización")
                    await asyncio.sleep(30)
                    continue
                
                # Ejecutar optimización (puede tomar varios segundos)
                new_duration = await self.duration_optimizer.optimize(
                    ticks_buffer=ticks_copy,
                    current_duration=self.duration
                )
                
                # Actualizar duración si cambió
                if new_duration is not None and new_duration != self.duration:
                    old_duration = self.duration
                    self.duration = new_duration
                    logger.info(
                        f"🔄 DURACIÓN ACTUALIZADA: {old_duration} → {new_duration}"
                    )
                    
                    # Opcional: Guardar en CouchDB para auditoría
                    await self._save_duration_change(old_duration, new_duration)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error en optimizador de duración: {e}", exc_info=True)
                await asyncio.sleep(30)
        
        logger.info("🔬 Optimizador de duración detenido")

    async def _save_duration_change(self, old_duration: int, new_duration: int) -> None:
        """Guarda cambio de duración en CouchDB para auditoría."""
        try:
            doc = {
                "type": "duration_change",
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": self.symbol,
                "old_duration": old_duration,
                "new_duration": new_duration,
                "optimizer_summary": self.duration_optimizer.get_recommendation_summary()
            }
            # await self.db.save(doc)
            logger.debug(f"💾 Cambio de duración registrado: {doc}")
        except Exception as e:
            logger.error(f"Error guardando cambio de duración: {e}")

    async def stop(self) -> None:
        """Detiene el motor y limpia recursos."""
        if self._is_running:
            self._is_running = False
            logger.info("⏹️ Deteniendo TradingEngine...")

            # ✅ Cancelar tarea de optimización
            if self._optimizer_task and not self._optimizer_task.done():
                self._optimizer_task.cancel()
                try:
                    await self._optimizer_task
                except asyncio.CancelledError:
                    pass
            
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
        if min_threshold <= (len_put_signals / total) <= max_threshold:
            return put_signals[-1]
                
        return None

