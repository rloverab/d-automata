# engine.py
"""
Motor de Trading Optimizado
Arquitectura: Producer-Consumer con manejo robusto de errores y reconexión unificada
"""

import asyncio
from collections import deque
import contextlib
from datetime import UTC, datetime
import logging
import math
from typing import Optional

from derivpy import MarketDataClient, OpenContract, Tick, TradingClient
from derivpy.exceptions import DerivAPIError, DerivConnectionError
from derivpy.reconnection import ReconnectionConfig, ReconnectionEvent, ReconnectionMode
from websockets import ConnectionClosedError, ConnectionClosedOK

from config.config import settings
from core.models import Signal
from optimizer.duration_optimizer import DurationOptimizer
from risk.recovery_matrix_calculator import RecoveryMatrixCalculator, TradeResult
from strategies.packs_strategy import ZScoreReversionConfig, ZScoreReversionStrategy

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, account_id: str, symbol: str, mode: ReconnectionMode):
        md_config = ReconnectionConfig(
            mode=mode,
            max_delay=120.0,
            force_history_reload_after=3,
            zombie_timeout=180.0,
            health_check_interval=30.0,
        )

        trading_config = ReconnectionConfig(
            mode=mode,
            max_delay=120.0,
            force_history_reload_after=3,
            zombie_timeout=300.0,
            health_check_interval=30.0,
        )

        self._is_running = False
        self._shutdown_event = asyncio.Event()
        self._market_data_client = MarketDataClient(reconnection_config=md_config)
        self._trading_client = TradingClient(
            app_id=settings.DERIV_APP_ID,
            token=settings.DERIV_TOKEN,
            account_id=account_id,
            reconnection_config=trading_config,
        )
        self.symbol = symbol

        # Modo de reconexión global
        self._reconnection_mode = mode
        self._last_user_alert: float = 0.0
        self._alert_cooldown: float = 300.0

        # Health Monitor y control de fallos
        self._last_tick_received: float = 0.0
        self._connection_failures: int = 0
        self._last_failure_time: Optional[datetime] = None

        # Buffer de ticks (producer)
        self.ticks_buffer = deque(maxlen=3600)
        self.tick_queue = asyncio.Queue(maxsize=100)

        # Estrategia y filtros
        self.strategy = ZScoreReversionStrategy(
            ZScoreReversionConfig(
                short_window=3,
                long_window=40,
                z_threshold=2.85,
                default_duration=4,
            )
        )
        self.filters = []
        self.duration = self.strategy.config.default_duration

        # Gestión de capital
        self.payout_pct = 90.0
        self.max_trades = 3
        self.target_wins = 1
        self.min_bet = 0.35
        self.trade_results = deque(maxlen=self.max_trades)
        self.initial_capital: float = 40.0
        self.round_allocation: float = 10.0
        self.remaining_capital: float = 0.0
        self.take_profit: float = 10100.0
        self.stop_loss: float = 9960.0

        self.use_early_exit = False
        self.early_exit_threshold = 0.70
        self.use_trailing_stop = False
        self.trailing_stop_pct = 0.50
        self.round_target_profit: float = 0.0

        self.recovery_calculator = RecoveryMatrixCalculator(
            payout_pct=self.payout_pct,
            max_trades=self.max_trades,
            target_wins=self.target_wins,
            min_bet=self.min_bet,
            use_early_exit=self.use_early_exit,
            early_exit_threshold=self.early_exit_threshold,
            use_trailing_stop=self.use_trailing_stop,
            trailing_stop_pct=self.trailing_stop_pct,
        )

        # Estado
        self.last_update: Optional[Tick] = None
        self._active_trades: set[asyncio.Task] = set()
        self._max_concurrent_trades = 1
        self._concurrency_warning_logged = False

        self.initial_round_capital: float = 0.0
        self.max_round_pnl: float = 0.0

        # Cooldown por losses consecutivos
        self._use_cooldown = False
        self._consecutive_losses = 0
        self._cooldown_threshold = 1
        self._cooldown_omissions = 1
        self._in_cooldown = False
        self._cooldown_signals_omitted = 0
        self._cooldown_until_epoch: Optional[float] = None
        self._seconds_per_tick = 0.0

        # Optimizador de duración y polaridad de señal
        self._optimizer_enabled = True
        self._optimizer_invert_enabled = True
        self._optimizer_interval_seconds = 60
        self._optimizer_task: Optional[asyncio.Task] = None
        self._last_optimizer_run: float = 0

        self.duration_optimizer = DurationOptimizer(
            strategy=self.strategy,
            filters=self.filters,
            duration_range=range(1, 11),
            min_win_rate=0.6,
            max_trades_per_round=self.max_trades,
            payout_pct=self.payout_pct,
            min_sample_size=5,
            hysteresis_margin=0.15,
            evaluate_inversion=self._optimizer_invert_enabled,
            max_buffer_size=3600,
        )

        self._invert_signal: bool = False
        self._inverted_signal: bool = False

        self.session_start_balance: float = 0.0

    async def start(self) -> None:
        """Inicia el motor con arquitectura producer-consumer."""
        self._is_running = True
        self._shutdown_event.clear()
        logger.info(f"🚀 Iniciando TradingEngine para: {self.symbol}")

        async def handle_connection_event(
            client_name: str,
            event: ReconnectionEvent,
            failure_count: int,
        ) -> bool:
            """Manejador centralizado de eventos de conexión."""
            try:
                if event == ReconnectionEvent.REQUIRES_HISTORY_RELOAD:
                    logger.info(f"🔄 {client_name}: Recarga de histórico requerida")
                    if client_name == "MarketDataClient":
                        await self._load_initial_ticks()
                        self._last_tick_received = asyncio.get_running_loop().time()
                    return True

                elif event == ReconnectionEvent.HEALTH_CHECK_PASSED:
                    logger.debug(f"✅ {client_name}: Recuperación completa")
                    if client_name == "MarketDataClient":
                        self._last_tick_received = asyncio.get_running_loop().time()
                    return True

                elif event == ReconnectionEvent.FATAL_FAILURE:
                    if self._reconnection_mode == ReconnectionMode.INFINITE_RETRY:
                        current_time = asyncio.get_running_loop().time()
                        if current_time - self._last_user_alert > self._alert_cooldown:
                            logger.critical(
                                f"🚨 ALERTA: {client_name} fallo #{failure_count}. "
                                f"Modo supervivencia activo."
                            )
                            self._last_user_alert = current_time

                        if failure_count > 0 and failure_count % 10 == 0:
                            await self._emergency_cleanup()

                        return True
                    else:
                        logger.critical(
                            f"💀 {client_name} fallo fatal #{failure_count}. Deteniendo motor."
                        )
                        asyncio.create_task(self.stop())
                        return False

                return True

            except Exception as exc:
                logger.error(f"❌ Error en manejador de eventos: {exc}", exc_info=True)
                return event != ReconnectionEvent.FATAL_FAILURE

        self._market_data_client.set_event_callback(handle_connection_event, timeout=15.0)
        self._trading_client.set_event_callback(handle_connection_event, timeout=15.0)

        await self._market_data_client.connect()
        await self._trading_client.connect()

        if not await self._initialize_new_round():
            logger.critical("❌ No se pudo inicializar la primera ronda. Deteniendo motor.")
            self._is_running = False
            await self.stop()
            return

        await self._load_initial_ticks()
        if self.ticks_buffer:
            self.last_update = self.ticks_buffer[-1]
        self._last_tick_received = asyncio.get_running_loop().time()

        try:
            producer_task = asyncio.create_task(self._tick_producer(), name="engine-producer")
            consumer_task = asyncio.create_task(self._signal_consumer(), name="engine-consumer")
            health_monitor_task = asyncio.create_task(self._health_monitor(), name="engine-health")
            connection_health_task = asyncio.create_task(
                self._connection_health_check(), name="engine-conn-health"
            )

            tasks = [
                producer_task,
                consumer_task,
                health_monitor_task,
                connection_health_task,
            ]

            if self._optimizer_enabled:
                self.duration_optimizer.evaluate_inversion = self._optimizer_invert_enabled
                self._optimizer_task = asyncio.create_task(
                    self._duration_optimizer_loop(), name="engine-optimizer"
                )
                tasks.append(self._optimizer_task)

            _, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        except asyncio.CancelledError:
            logger.info("⏹️ Motor cancelado")
        except Exception as e:
            logger.error(f"❌ Error crítico en el motor: {e}", exc_info=True)
        finally:
            await self.stop()

    async def _tick_producer(self) -> None:
        """
        Producer: Recibe ticks del WebSocket y los encola para procesamiento.
        Mantiene el bucle vivo si ocurre una caída transitoria del stream.
        """
        while self._is_running and not self._shutdown_event.is_set():
            try:
                if not self._market_data_client.is_connection_stable:
                    await self._market_data_client.wait_until_stable(timeout=5.0)
                    continue

                async for tick in self._market_data_client.stream_ticks(self.symbol):
                    if not self._is_running or self._shutdown_event.is_set():
                        return

                    if not self._market_data_client.is_connection_stable:
                        await asyncio.sleep(0.2)
                        continue

                    self._last_tick_received = asyncio.get_running_loop().time()

                    if self.ticks_buffer:
                        last_tick = self.ticks_buffer[-1]
                        if tick.epoch - last_tick.epoch > 5:
                            logger.warning("⚠️ Gap detectado en stream, recargando histórico")
                            await self._load_initial_ticks()
                        else:
                            self.ticks_buffer.append(tick)
                            self.last_update = tick
                    else:
                        self.ticks_buffer.append(tick)
                        self.last_update = tick

                    try:
                        self.tick_queue.put_nowait(tick)
                    except asyncio.QueueFull:
                        logger.warning("⚠️ Cola de ticks llena, descartando tick antiguo")
                        try:
                            self.tick_queue.get_nowait()
                            self.tick_queue.put_nowait(tick)
                        except asyncio.QueueEmpty:
                            pass

                if self._is_running and not self._shutdown_event.is_set():
                    logger.warning("🔄 Stream de ticks finalizado, esperando reconexión para reabrir...")
                    await asyncio.sleep(1.0)

            except DerivConnectionError as e:
                logger.warning(f"🔌 Desconexión en stream de market data: {e}")
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                logger.info("⏹️ Producer cancelado")
                raise
            except Exception as e:
                logger.error(f"❌ Error en producer: {e}", exc_info=True)
                await asyncio.sleep(2.0)

    async def _health_monitor(self) -> None:
        """Watchdog que previene estados zombies forzando reconexión real si el stream se congela."""
        zombie_timeout = self._market_data_client._reconnection_config.zombie_timeout
        check_interval = self._market_data_client._reconnection_config.health_check_interval

        logger.info(
            "🏥 Health Monitor | Modo: %s | Zombie timeout: %ss",
            self._reconnection_mode.name,
            zombie_timeout,
        )

        await asyncio.sleep(15)
        self._last_tick_received = asyncio.get_running_loop().time()

        while self._is_running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(check_interval)

                if not self._is_running or self._shutdown_event.is_set():
                    break

                current_time = asyncio.get_running_loop().time()
                time_since_last_tick = current_time - self._last_tick_received

                if not self._market_data_client.is_connection_stable:
                    logger.debug("🏥 Esperando reconexión del MarketDataClient...")
                    continue

                if time_since_last_tick > zombie_timeout:
                    if self._reconnection_mode == ReconnectionMode.INFINITE_RETRY:
                        logger.warning(
                            f"⚠️ Sin ticks por {time_since_last_tick:.0f}s (conexión zombie). "
                            f"Forzando reconexión de MarketDataClient y limpieza..."
                        )
                        await self._market_data_client.force_reconnect(
                            reason=f"Sin ticks durante {time_since_last_tick:.0f}s"
                        )
                        await self._market_data_client.wait_until_stable(timeout=30.0)
                        await self._emergency_cleanup()
                        self._last_tick_received = asyncio.get_running_loop().time()
                    else:
                        logger.critical(
                            f"💀 Health Monitor: Sin ticks por {time_since_last_tick:.0f}s. "
                            f"Deteniendo motor."
                        )
                        await self.stop()
                        break

            except asyncio.CancelledError:
                logger.info("⏹️ Health Monitor cancelado")
                break
            except Exception as e:
                logger.error(f"Error en Health Monitor: {e}", exc_info=True)

    async def _emergency_cleanup(self) -> None:
        """Limpieza profunda periódica para prevenir memory leaks."""
        try:
            self.ticks_buffer.clear()
            self._last_tick_received = asyncio.get_running_loop().time()

            for task in list(self._active_trades):
                if task.done():
                    self._active_trades.discard(task)

            if self._trading_client.is_connection_stable:
                try:
                    portfolio = await self._trading_client.get_portfolio()
                    if portfolio:
                        logger.warning(
                            f"⚠️ {len(portfolio)} contratos abiertos detectados "
                            f"durante limpieza de emergencia"
                        )
                except Exception as e:
                    logger.error(f"Error verificando portfolio: {e}")

            if self._market_data_client.is_connection_stable:
                await self._load_initial_ticks()
            logger.info("✅ Limpieza profunda completada")

        except Exception as e:
            logger.error(f"❌ Error en limpieza de emergencia: {e}", exc_info=True)

    def _cleanup_trade_task(self, task: asyncio.Task) -> None:
        """Elimina la tarea del conjunto de activos cuando termina."""
        self._active_trades.discard(task)
        if not task.cancelled() and task.exception():
            logger.error(f"❌ Trade terminado con error no manejado: {task.exception()}")

    async def _signal_consumer(self) -> None:
        """Consumer que maneja cooldown como omisión de señales individuales."""
        ticks_processed = 0
        signals_attempted = 0
        signals_generated = 0
        signals_passed_filter = 0
        signals_executed = 0
        signals_omitted = 0

        last_log_time = asyncio.get_running_loop().time()

        while self._is_running and not self._shutdown_event.is_set():
            try:
                try:
                    _ = await asyncio.wait_for(self.tick_queue.get(), timeout=1.0)
                    self.tick_queue.task_done()
                except asyncio.TimeoutError:
                    continue

                if self._shutdown_event.is_set():
                    break

                if (
                    not self._market_data_client.is_connection_stable
                    or not self._trading_client.is_connection_stable
                ):
                    logger.debug("⏸️ Motor pausado: Esperando estabilización de WebSockets...")
                    await asyncio.sleep(1.0)
                    continue

                ticks_processed += 1

                current_time = asyncio.get_running_loop().time()
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

                if not await self._ensure_active_round():
                    continue

                if not self._can_execute_trade():
                    continue

                signals_attempted += 1

                if self.duration == 0:
                    continue

                signal = self.strategy.get_signal(
                    data=list(self.ticks_buffer),
                    duration=self.duration,
                    inverted_signal=self._inverted_signal,
                )

                if signal is None:
                    continue

                signals_generated += 1

                for flt in self.filters:
                    signal = flt(signal)
                    if signal is None:
                        break

                if signal is None:
                    continue

                signals_passed_filter += 1

                if self._in_cooldown:
                    signals_omitted += 1

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

                bet = self.recovery_calculator.calculate_next_bet(
                    history=list(self.trade_results),
                    current_round_capital=self.remaining_capital,
                )

                if bet["status"] != "ACTIVE":
                    continue

                trade_task = asyncio.create_task(self._execute_trade(signal, bet))
                trade_task.add_done_callback(self._cleanup_trade_task)
                self._active_trades.add(trade_task)
                signals_executed += 1

            except asyncio.CancelledError:
                logger.info("⏹️ Consumer cancelado")
                break
            except Exception as e:
                logger.error(f"❌ Error en consumer: {e}", exc_info=True)
                await asyncio.sleep(1.0)

    async def _execute_cooldown_for_signal(
        self, signal: Signal, seconds_per_tick: float = 1.0
    ) -> None:
        """Ejecuta un período de cooldown para una señal específica."""
        try:
            cooldown_seconds = signal.duration * seconds_per_tick
            logger.debug(
                f"⏸️ Cooldown virtual: Esperando {cooldown_seconds:.0f}s "
                f"(señal de {signal.duration} ticks)"
            )
            await asyncio.sleep(cooldown_seconds)

            self._cooldown_signals_omitted += 1
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
        """Valida si se puede ejecutar un trade en este momento."""
        if not self._trading_client.is_connection_stable:
            logger.warning("⚠️ Cliente desconectado, omitiendo trade")
            return False

        if self.remaining_capital < self.min_bet:
            logger.warning(
                f"⚠️ Capital de ronda insuficiente: {self.remaining_capital:.2f} < {self.min_bet}"
            )
            return False

        active_count = len(self._active_trades)
        if active_count >= self._max_concurrent_trades:
            if not self._concurrency_warning_logged:
                logger.debug(
                    f"⚠️ Límite de concurrencia alcanzado: {active_count}/{self._max_concurrent_trades} trades activos"
                )
                self._concurrency_warning_logged = True
            return False

        self._concurrency_warning_logged = False
        return True

    async def _execute_trade(self, signal: Signal, bet) -> None:
        """Ejecuta un trade delegando la reconexión al gestor único de TradingClient."""
        try:
            logger.info(
                f"💼 Ejecutando trade: {signal.direction} @ {bet['next_bet']} | "
                f"Señal invertida: {'si' if self._inverted_signal else 'no'}"
            )

            if not await self._ensure_trading_connection(timeout=15.0):
                logger.error("🔌 No se pudo establecer conexión de trading, abortando trade")
                return

            try:
                br = await self._trading_client.trade(
                    contract_type=signal.direction,
                    underlying_symbol=self.symbol,
                    amount=float(round(bet["next_bet"], 2)),
                    duration=signal.duration,
                    duration_unit=signal.duration_unit,
                )
            except (ConnectionClosedOK, ConnectionClosedError, DerivConnectionError) as e:
                logger.warning(f"🔌 Conexión interrumpida durante trade: {e}")
                logger.info("🔄 Esperando estabilización de TradingClient para reintentar...")

                if not await self._ensure_trading_connection(timeout=20.0):
                    logger.error("❌ No se pudo restablecer conexión, abortando trade")
                    return

                try:
                    br = await self._trading_client.trade(
                        contract_type=signal.direction,
                        underlying_symbol=self.symbol,
                        amount=float(round(bet["next_bet"], 2)),
                        duration=signal.duration,
                        duration_unit=signal.duration_unit,
                    )
                except Exception as retry_error:
                    logger.error(f"❌ Error en reintento de trade: {retry_error}")
                    return

            max_wait = signal.duration * 2 + 30

            try:
                async with asyncio.timeout(max_wait):
                    async for c in self._trading_client.watch_contract(
                        br.contract_id,
                        stop_when_closed=False,
                    ):
                        if c.is_expired and c.status != "open":
                            break
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout esperando expiración del contrato {br.contract_id}")
            except DerivConnectionError as e:
                logger.warning(f"🔌 Conexión perdida durante watch_contract: {e}")
                await self._ensure_trading_connection(timeout=20.0)

            contract: OpenContract | None = None
            max_retries = 3

            for attempt in range(max_retries):
                try:
                    await self._ensure_trading_connection(timeout=15.0)
                    contract = await self._trading_client.get_contract(br.contract_id)
                    if contract.status != "open":
                        break
                    await asyncio.sleep(2.0)
                except DerivConnectionError:
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"🔄 Retry {attempt + 1}/{max_retries} para obtener contrato {br.contract_id}"
                        )
                        await asyncio.sleep(2.0)
                    else:
                        logger.error(
                            f"❌ No se pudo obtener resultado del contrato {br.contract_id}"
                        )
                        return

            if contract is None:
                logger.error(f"❌ Contrato {br.contract_id} sin resultado definitivo")
                return

            result: TradeResult = "win" if contract.profit > 0 else "loss"
            self.trade_results.append(result)
            self.remaining_capital += contract.profit

            should_activate_cooldown = False

            if self._use_cooldown:
                if result == "loss":
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
                        logger.info("✅ Win detectado. Reseteando contador de losses consecutivos.")
                    self._consecutive_losses = 0

            current_pnl = self.remaining_capital - self.initial_round_capital
            if current_pnl > self.max_round_pnl:
                self.max_round_pnl = current_pnl

            logger.info(
                f"✅ Trade completado: {result} | "
                f"Profit: {contract.profit:.2f} | "
                f"Capital restante: {self.remaining_capital:.2f}"
            )

            if self.use_early_exit:
                if self.recovery_calculator.should_close_round_early(
                    current_pnl,
                    self.round_target_profit,
                ):
                    logger.info(
                        f"🎯 Early Exit activado: PnL ${current_pnl:.2f} >= "
                        f"{self.early_exit_threshold * 100:.0f}% del target"
                    )
                    await self._finalize_round("EARLY_EXIT_PROFIT_THRESHOLD")
                    await self._initialize_new_round()
                    return

            if self.recovery_calculator.should_activate_trailing_stop(
                current_pnl, self.max_round_pnl
            ):
                logger.warning(
                    f"📉 Trailing Stop Activado: PnL cayó de ${self.max_round_pnl:.2f} a ${current_pnl:.2f}"
                )
                await self._finalize_round("TRAILING_STOP_ACTIVATED")
                await self._initialize_new_round()
                return

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

            if should_activate_cooldown:
                self._in_cooldown = True
                self._cooldown_signals_omitted = 0
                logger.info(
                    f"⏸️ Cooldown activado: Próximas {self._cooldown_omissions} señales serán omitidas"
                )

        except DerivConnectionError as e:
            logger.error(f"🔌 Error de conexión durante trade: {e}")

        except DerivAPIError as e:
            logger.error(f"❌ Error de API durante trade: {e}")

        except Exception as e:
            logger.error(f"❌ Error inesperado en trade: {e}", exc_info=True)

    def _is_circuit_breaker_open(self) -> bool:
        """Verifica si el circuit breaker está activo."""
        if self._connection_failures < 3:
            return False

        if self._last_failure_time:
            elapsed = (datetime.now(UTC) - self._last_failure_time).total_seconds()
            if elapsed < 60:
                return True
            self._connection_failures = 0

        return False

    async def _save_trade_to_db(self, contract, signal, bet, result: TradeResult) -> None:
        """Guarda el trade en CouchDB para auditoría."""
        try:
            trade_doc = {
                "type": "trade",
                "timestamp": datetime.now(UTC).isoformat(),
                "symbol": self.symbol,
                "direction": signal.direction,
                "amount": bet["next_bet"],
                "duration": signal.duration,
                "result": result,
                "profit": contract.profit,
                "remaining_capital": self.remaining_capital,
                "round_wins": sum(1 for r in self.trade_results if r == "win"),
                "round_losses": sum(1 for r in self.trade_results if r == "loss"),
            }
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
                for _ in range(batches):
                    batch_size = min(1000, count - len(all_ticks))
                    ts, _ = await self._market_data_client.ticks_history(
                        symbol=self.symbol, count=batch_size, end=end
                    )
                    if not ts:
                        break
                    all_ticks = ts + all_ticks
                    end = ts[0].epoch - 1
                    await asyncio.sleep(0.1)
            else:
                all_ticks, _ = await self._market_data_client.ticks_history(
                    symbol=self.symbol, count=count
                )

            sorted_ticks = sorted(all_ticks, key=lambda t: t.epoch)
            self.ticks_buffer.clear()
            self.ticks_buffer.extend(sorted_ticks)

            if len(all_ticks) > 0 and verbose:
                logger.info(f"✅ Cargados {len(all_ticks)} ticks históricos")

        except Exception as e:
            logger.error(f"❌ Error cargando ticks iniciales: {e}")

    async def _ensure_active_round(self) -> bool:
        """Gestiona el ciclo de vida de la ronda."""
        bet_info = self.recovery_calculator.calculate_next_bet(
            history=list(self.trade_results),
            current_round_capital=self.remaining_capital,
        )

        if bet_info["status"] != "ACTIVE":
            logger.warning(f"🔄 Ronda finalizada. Motivo: {bet_info['status']}.")
            self._consecutive_losses = 0

            await self._finalize_round(bet_info["status"])

            can_continue = await self._initialize_new_round()
            if not can_continue:
                return False

            if (
                not self._optimizer_invert_enabled
                and bet_info["status"] == "MAX_TRADES_REACHED"
                and self._invert_signal
            ):
                self._inverted_signal = not self._inverted_signal

        final_bet_info = self.recovery_calculator.calculate_next_bet(
            history=list(self.trade_results),
            current_round_capital=self.remaining_capital,
        )

        if final_bet_info["status"] != "ACTIVE":
            logger.critical("🛑 Imposible iniciar nueva ronda. Capital global insuficiente.")
            self._is_running = False
            return False

        return True

    async def _initialize_new_round(self) -> bool:
        """Consulta el balance real y asigna capital para una nueva ronda."""
        try:
            balance_resp = await self._trading_client.get_balance()
            current_total_balance = float(balance_resp.balance)

            if self.session_start_balance == 0.0:
                self.session_start_balance = current_total_balance

            if current_total_balance <= self.stop_loss:
                logger.critical(f"🛑 STOP LOSS GLOBAL ALCANZADO. Balance: {current_total_balance}")
                self._is_running = False
                return False

            if current_total_balance >= self.take_profit:
                logger.info(f"🎯 TAKE PROFIT GLOBAL ALCANZADO. Balance: {current_total_balance}")
                self._is_running = False
                return False

            if current_total_balance < self.min_bet:
                logger.critical(
                    f"💀 BALANCE CRÍTICO. Insuficiente para el trade mínimo ({current_total_balance} < 0.35)"
                )
                self._is_running = False
                return False

            new_round_capital = min(self.round_allocation, current_total_balance)

            self.initial_round_capital = new_round_capital
            self.remaining_capital = new_round_capital
            self.trade_results.clear()
            self.max_round_pnl = 0.0

            self.round_target_profit = self.recovery_calculator.get_round_target_profit(
                initial_round_capital=self.initial_round_capital
            )

            equity_change = current_total_balance - self.session_start_balance
            equity_change_pct = (
                (equity_change / self.session_start_balance) * 100
                if self.session_start_balance > 0
                else 0.0
            )

            if equity_change >= 0:
                status_line = f"   ├─ Equity vs Inicio Sesión: +${equity_change:.2f} (+{equity_change_pct:.2f}%)"
            else:
                drawdown = abs(equity_change)
                drawdown_pct = abs(equity_change_pct)
                status_line = f"   ├─ Drawdown vs Inicio Sesión: -${drawdown:.2f} (-{drawdown_pct:.2f}%)"

            logger.info(
                f"✅ Nueva ronda iniciada.\n"
                f"   ├─ Asignación Ronda: ${self.remaining_capital:.2f} (de ${current_total_balance:.2f} disponibles)\n"
                f"{status_line}\n"
                f"   └─ Target PnL (Ganancia Neta de la Ronda): ${self.round_target_profit:.2f}"
            )

            return True

        except Exception as e:
            logger.error(f"❌ Error al consultar balance para nueva ronda: {e}", exc_info=True)
            return False

    async def _finalize_round(self, reason: str) -> None:
        """Registra el cierre de la ronda en CouchDB para auditoría y análisis de riesgo."""
        wins = sum(1 for r in self.trade_results if r == "win")
        losses = sum(1 for r in self.trade_results if r == "loss")
        net_round_profit = self.remaining_capital - self.initial_round_capital

        logger.info(
            f"💾 Ronda registrada: {wins}W / {losses}L | Motivo: {reason} | PnL: {net_round_profit:.2f}"
        )

    async def _duration_optimizer_loop(self) -> None:
        """Tarea en background que ejecuta el optimizador periódicamente."""
        logger.info(
            f"🔬 Optimizador iniciado | Intervalo: {self._optimizer_interval_seconds}s | "
            f"Evaluar inversión: {self._optimizer_invert_enabled}"
        )

        await asyncio.sleep(15)

        while self._is_running and not self._shutdown_event.is_set():
            try:
                current_time = asyncio.get_running_loop().time()
                if current_time - self._last_optimizer_run < self._optimizer_interval_seconds:
                    await asyncio.sleep(10)
                    continue

                self._last_optimizer_run = current_time
                ticks_copy = list(self.ticks_buffer)

                if len(ticks_copy) < 500:
                    logger.debug("⏸️ Buffer insuficiente para optimización")
                    await asyncio.sleep(30)
                    continue

                self.duration_optimizer.evaluate_inversion = self._optimizer_invert_enabled
                decision = await self.duration_optimizer.optimize_with_inversion(
                    ticks_buffer=ticks_copy,
                    current_duration=self.duration,
                    current_inverted=self._inverted_signal,
                )

                changed = False

                # Si no hay duración recomendada, se conserva la duración actual
                if decision.duration is not None and decision.duration != self.duration:
                    old_duration = self.duration
                    self.duration = decision.duration
                    changed = True
                    logger.info(f"🔄 DURACIÓN ACTUALIZADA: {old_duration} → {self.duration}")

                # Si se evalúa inversión y hay recomendación, actualizar; si es None, conservar actual
                if (
                    self._optimizer_invert_enabled
                    and decision.inverted_signal is not None
                    and decision.inverted_signal != self._inverted_signal
                ):
                    old_inverted = self._inverted_signal
                    self._inverted_signal = decision.inverted_signal
                    changed = True
                    logger.info(
                        f"🔀 INVERSIÓN DE SEÑAL ACTUALIZADA: {old_inverted} → {self._inverted_signal}"
                    )

                if changed:
                    await self._save_optimizer_change()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error en optimizador de duración: {e}", exc_info=True)
                await asyncio.sleep(30)

        logger.info("🔬 Optimizador de duración detenido")

    async def _save_optimizer_change(self) -> None:
        """Guarda cambios del optimizador en CouchDB para auditoría."""
        try:
            logger.debug(
                f"💾 Cambio de optimizador -> Duración: {self.duration}, "
                f"Invertida: {self._inverted_signal} "
                f"| Resumen: {self.duration_optimizer.get_recommendation_summary()}"
            )
        except Exception as e:
            logger.error(f"Error guardando cambio de optimizador: {e}")

    async def _ensure_trading_connection(self, timeout: float = 15.0) -> bool:
        """
        Espera a que TradingClient esté estable sin interferir con su gestor de reconexión.
        Si no está reconectando ni conectado, dispara force_reconnect() una sola vez.
        """
        if self._trading_client.is_connection_stable:
            return True

        if not self._trading_client.is_reconnecting:
            await self._trading_client.force_reconnect(
                reason="Verificación previa/durante ejecución de trade"
            )

        return await self._trading_client.wait_until_stable(timeout=timeout)

    async def _connection_health_check(self) -> None:
        """
        Supervisa el tiempo desconectado de ambos clientes y dispara force_reconnect
        solo si ninguno está ya ejecutando su ciclo de reconexión.
        """
        check_interval = 60.0
        max_disconnected_time = 300.0

        md_disconnected_since = None
        trading_disconnected_since = None

        while self._is_running and not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(check_interval)

                current_time = asyncio.get_running_loop().time()

                # Verificar MarketDataClient
                if not self._market_data_client.is_connection_stable:
                    if md_disconnected_since is None:
                        md_disconnected_since = current_time
                        logger.warning("⚠️ MarketDataClient desconectado")
                    else:
                        disconnected_duration = current_time - md_disconnected_since
                        if (
                            disconnected_duration > max_disconnected_time
                            and not self._market_data_client.is_reconnecting
                        ):
                            logger.critical(
                                f"💀 MarketDataClient desconectado por {disconnected_duration:.0f}s "
                                f"sin tarea activa. Disparando force_reconnect..."
                            )
                            await self._market_data_client.force_reconnect(
                                reason=f"Inactivo > {max_disconnected_time:.0f}s"
                            )
                            md_disconnected_since = None
                else:
                    if md_disconnected_since is not None:
                        logger.info("✅ MarketDataClient reconectado")
                        md_disconnected_since = None

                # Verificar TradingClient
                if not self._trading_client.is_connection_stable:
                    if trading_disconnected_since is None:
                        trading_disconnected_since = current_time
                        logger.warning("⚠️ TradingClient desconectado")
                    else:
                        disconnected_duration = current_time - trading_disconnected_since
                        if (
                            disconnected_duration > max_disconnected_time
                            and not self._trading_client.is_reconnecting
                        ):
                            logger.critical(
                                f"💀 TradingClient desconectado por {disconnected_duration:.0f}s "
                                f"sin tarea activa. Disparando force_reconnect..."
                            )
                            await self._trading_client.force_reconnect(
                                reason=f"Inactivo > {max_disconnected_time:.0f}s"
                            )
                            trading_disconnected_since = None
                else:
                    if trading_disconnected_since is not None:
                        logger.info("✅ TradingClient reconectado")
                        trading_disconnected_since = None

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"Error en health check: {exc}", exc_info=True)

    async def stop(self) -> None:
        """Detiene el motor y limpia recursos."""
        if self._is_running:
            self._is_running = False
            self._shutdown_event.set()
            logger.info("⏹️ Deteniendo TradingEngine...")

            if self._optimizer_task and not self._optimizer_task.done():
                self._optimizer_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._optimizer_task

            if self._active_trades:
                logger.info(f"⏳ Esperando {len(self._active_trades)} trades activos...")
                await asyncio.wait(self._active_trades, timeout=30.0)

            await self._market_data_client.disconnect()
            await self._trading_client.disconnect()
            logger.info("✅ Motor detenido correctamente")