"""
Motor de Backtesting Determinista v3.1
Alineado con TradingEngine: Gestión de Capital Real, Rondas, Early Exit, Trailing Stop, SL/TP y Cooldown
Arquitectura: Event-Loop Sincrónico con simulación estricta de delay de entrada
"""

import asyncio
from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from derivpy import Tick

from core.models import Signal, TradeRecord
from filters.base import Filter
from risk.recovery_matrix_calculator import RecoveryMatrixCalculator
from strategies.base import Strategy
from utils.tick_exporter import TickExporter
from utils.trade_exporter import TradeExporter

logger = logging.getLogger(__name__)


class BacktestingEngine:
    """
    Motor de backtesting que replica fielmente la lógica de TradingEngine:
    1. Delay de entrada (señal en T, ejecución en T+1)
    2. Duración real de la orden en ticks
    3. Concurrencia parametrizable (default 1)
    4. Cadena de filtros pre-ejecución
    5. Cooldown opcional por pérdidas consecutivas (omisión de señales)
    6. Gestión de rondas con RecoveryMatrixCalculator (Early Exit, Trailing Stop, Viabilidad)
    7. SL/TP globales con parada inmediata post-trade
    8. Contabilidad alineada con el balance real
    """

    def __init__(
        self,
        strategy: Strategy,
        filters: List[Filter],
        max_concurrent_trades: int = 1,
        payout_pct: float = 90.0,
        initial_capital: float = 100.0,
        round_allocation: float = 15.0,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        max_trades: int = 3,
        target_wins: int = 1,
        min_bet: float = 0.35,
        use_early_exit: bool = False,
        early_exit_threshold: float = 0.70,
        use_trailing_stop: bool = False,
        trailing_stop_pct: float = 0.50,
        use_min_viability: bool = False,
        min_viability_pct: float = 0.10,
        use_fixed_stake: bool = False,
        fixed_stake_amount: float = 1.0,
        duration: int = 5,
        invert_signal: bool = False,
        inverted_signal: bool = False,
        use_cooldown: bool = False,
        cooldown_threshold: int = 1,
        cooldown_omissions: int = 1,
        max_buffer_size: int = 3600,
    ):
        self.strategy: Strategy = strategy
        self.filters = filters
        self.max_concurrent_trades = max_concurrent_trades
        self.max_buffer_size = max_buffer_size

        # Validación de consistencia de payout_pct (espera formato 0-100, ej. 90.0)
        if 0 < payout_pct < 1:
            logger.warning(
                f"⚠️ payout_pct={payout_pct} parece estar en formato decimal. "
                f"Convirtiendo a porcentaje (×100)"
            )
            payout_pct = payout_pct * 100

        if payout_pct <= 0 or payout_pct > 100:
            logger.error(f"❌ payout_pct={payout_pct} fuera de rango (0, 100].")
            raise ValueError("payout_pct debe estar entre 0 y 100")

        self.payout_pct = payout_pct

        # Configuración de Capital
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.round_allocation = round_allocation
        self.take_profit = take_profit
        self.stop_loss = stop_loss

        # Configuración de Matriz
        self.max_trades = max_trades
        self.target_wins = target_wins
        self.min_bet = min_bet

        # Protección de ganancias y viabilidad
        self.use_early_exit = use_early_exit
        self.early_exit_threshold = early_exit_threshold
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        self.use_min_viability = use_min_viability
        self.min_viability_pct = min_viability_pct

        # Modo de Apuesta
        self.use_fixed_stake = use_fixed_stake
        self.fixed_stake_amount = fixed_stake_amount

        # Duración de la operación
        self.duration = duration

        # Instancia única de RecoveryMatrixCalculator (alineada con TradingEngine)
        self.recovery_calculator = RecoveryMatrixCalculator(
            payout_pct=self.payout_pct,
            max_trades=self.max_trades,
            target_wins=self.target_wins,
            min_bet=self.min_bet,
            use_early_exit=self.use_early_exit,
            early_exit_threshold=self.early_exit_threshold,
            use_trailing_stop=self.use_trailing_stop,
            trailing_stop_pct=self.trailing_stop_pct,
            use_min_viability=self.use_min_viability,
            min_viability_pct=self.min_viability_pct,
        )

        # Estado Interno
        self.ticks_buffer: List[Tick] = []
        self.active_trades: List[Dict[str, Any]] = []
        self.completed_trades: List[TradeRecord] = []

        # Estado de Ronda (alineado con TradingEngine)
        self.remaining_round_capital: float = 0.0
        self.round_trade_results: deque[str] = deque(maxlen=self.max_trades)
        self.current_round_id: int = 0
        self.round_starting_balance: float = 0.0
        self.initial_round_capital: float = 0.0
        self.round_target_profit: float = 0.0
        self.max_round_pnl: float = 0.0

        # Tracking de resultados de rondas
        self.round_results: List[str] = []
        self.round_close_reasons: List[str] = []

        # Cooldown por losses consecutivos (alineado con TradingEngine)
        self._use_cooldown = use_cooldown
        self._consecutive_losses = 0
        self._cooldown_threshold = cooldown_threshold
        self._cooldown_omissions = cooldown_omissions
        self._in_cooldown = False
        self._cooldown_signals_omitted = 0
        self._cooldown_until_idx: Optional[int] = None

        # Métricas de Flujo
        self.signals_attempted = 0
        self.signals_generated = 0
        self.signals_passed_filter = 0
        self.signals_omitted_cooldown = 0
        self.signals_skipped_concurrency = 0
        self.signals_skipped_capital = 0

        # Flag de parada
        self._stop_requested = False

        # Inversión de señal
        self._invert_signal: bool = invert_signal
        self._inverted_signal: bool = inverted_signal

    async def load_ticks(self, symbol: str, count: int = 5000):
        """Carga histórico usando MarketDataClient."""
        import math
        from derivpy import MarketDataClient

        client = MarketDataClient()
        try:
            await client.connect()
            all_ticks = []
            if count > 1000:
                batches = math.ceil(count / 1000)
                end = "latest"
                for _ in range(batches):
                    batch_size = min(1000, count - len(all_ticks))
                    ts, _ = await client.ticks_history(symbol=symbol, count=batch_size, end=end)
                    if not ts:
                        break
                    all_ticks = ts + all_ticks
                    end = ts[0].epoch - 1
                    await asyncio.sleep(0.3)
            else:
                all_ticks, _ = await client.ticks_history(symbol=symbol, count=count)

            self.ticks_buffer = sorted(all_ticks, key=lambda t: t.epoch)
            logger.info(f"✅ Cargados {len(self.ticks_buffer)} ticks para backtest")
        finally:
            await client.disconnect()

    def _check_global_limits(self) -> bool:
        """Verifica SL, TP y balance mínimo (equivalente a TradingEngine)."""
        if self.stop_loss is not None and self.current_balance <= self.stop_loss:
            logger.info(f"🛑 SL GLOBAL ALCANZADO: ${self.current_balance:.2f}")
            return False

        if self.take_profit is not None and self.current_balance >= self.take_profit:
            logger.info(f"🎯 TP GLOBAL ALCANZADO: ${self.current_balance:.2f}")
            return False

        if self.current_balance < self.min_bet:
            logger.warning(
                f"💀 BALANCE CRÍTICO: ${self.current_balance:.2f} < ${self.min_bet:.2f}"
            )
            return False

        return True

    def _initialize_new_round(self) -> bool:
        """Asigna capital y calcula target de ronda (igual que TradingEngine._initialize_new_round)."""
        if not self._check_global_limits():
            return False

        allocation = min(self.round_allocation, self.current_balance)
        self.initial_round_capital = allocation
        self.remaining_round_capital = allocation
        self.round_trade_results.clear()
        self.current_round_id += 1
        self.round_starting_balance = self.current_balance
        self.max_round_pnl = 0.0

        self.round_target_profit = self.recovery_calculator.get_round_target_profit(
            initial_round_capital=self.initial_round_capital
        )

        logger.debug(
            f"🔄 Ronda #{self.current_round_id} | Capital: ${allocation:.2f} | "
            f"Target PnL: ${self.round_target_profit:.2f}"
        )
        return True

    def _finalize_round(self, reason: str = "UNKNOWN") -> str:
        """Registra cierre de ronda con motivo explícito."""
        round_pnl = self.remaining_round_capital - self.initial_round_capital
        result = "win" if round_pnl > 0 else "loss"
        self.round_results.append(result)
        self.round_close_reasons.append(reason)

        logger.debug(
            f"🏁 Ronda #{self.current_round_id}: {result} | "
            f"Motivo: {reason} | PnL: ${round_pnl:.2f}"
        )
        return result

    def _ensure_active_round(self) -> bool:
        """
        Replica exactamente TradingEngine._ensure_active_round().
        Verifica si la ronda actual sigue activa; si terminó, la cierra, abre una nueva
        y valida que sea operable.
        """
        if self.use_fixed_stake:
            if self.remaining_round_capital < self.fixed_stake_amount:
                self._consecutive_losses = 0
                self._finalize_round("INSUFFICIENT_CAPITAL")
                if not self._initialize_new_round():
                    return False
                if self.remaining_round_capital < self.fixed_stake_amount:
                    return False
            return True

        bet_info = self.recovery_calculator.calculate_next_bet(
            history=list(self.round_trade_results),
            current_round_capital=self.remaining_round_capital,
        )

        if bet_info["status"] != "ACTIVE":
            logger.debug(f"🔄 Ronda finalizada. Motivo: {bet_info['status']}.")
            self._consecutive_losses = 0

            self._finalize_round(bet_info["status"])

            if not self._initialize_new_round():
                return False

            if bet_info["status"] == "MAX_TRADES_REACHED" and self._invert_signal:
                self._inverted_signal = not self._inverted_signal

        final_bet_info = self.recovery_calculator.calculate_next_bet(
            history=list(self.round_trade_results),
            current_round_capital=self.remaining_round_capital,
        )

        if final_bet_info["status"] != "ACTIVE":
            logger.warning(
                f"🛑 Imposible operar nueva ronda ({final_bet_info['status']})."
            )
            self._stop_requested = True
            return False

        return True

    def _get_next_stake(self) -> Optional[float]:
        """Calcula el stake basado en modo fijo o RecoveryMatrixCalculator."""
        if self.use_fixed_stake:
            stake = self.fixed_stake_amount
        else:
            bet_info = self.recovery_calculator.calculate_next_bet(
                history=list(self.round_trade_results),
                current_round_capital=self.remaining_round_capital,
            )
            if bet_info["status"] != "ACTIVE":
                return None
            stake = float(bet_info["next_bet"])

        if stake > self.remaining_round_capital or stake > self.current_balance:
            return None

        return round(stake, 2)

    def run(self) -> Dict[str, Any]:
        """Bucle principal determinista del backtest."""
        if len(self.ticks_buffer) < 100:
            raise ValueError("Buffer de ticks insuficiente para backtest")

        self._stop_requested = False

        # Iniciar primera ronda
        if not self._initialize_new_round():
            return {"error": "Capital inicial insuficiente o límites alcanzados"}

        logger.info(
            f"🚀 Backtest | Strat: {self.strategy.__class__.__name__} | "
            f"Mode: {'Fixed' if self.use_fixed_stake else 'Matrix'} | "
            f"Concurrent: {self.max_concurrent_trades}"
        )

        start_idx = 50
        for i in range(start_idx, len(self.ticks_buffer)):
            current_tick = self.ticks_buffer[i]

            # 1. Resolver trades expirados en este tick
            self._resolve_expiring_trades(i, current_tick)

            if self._stop_requested or not self._check_global_limits():
                break

            # 2. Gestionar cooldown virtual en curso (simula el tiempo de espera de signal.duration)
            if self._in_cooldown and self._cooldown_until_idx is not None:
                if i >= self._cooldown_until_idx:
                    self._cooldown_until_idx = None
                    self._cooldown_signals_omitted += 1
                    if self._cooldown_signals_omitted >= self._cooldown_omissions:
                        self._in_cooldown = False
                        self._cooldown_signals_omitted = 0
                        self._consecutive_losses = 0
                else:
                    self.signals_skipped_concurrency += 1
                    continue

            # 3. Verificar estado de ronda ANTES de evaluar señal (TradingEngine._ensure_active_round)
            if not self._ensure_active_round():
                break

            # 4. Concurrencia (TradingEngine._can_execute_trade)
            if len(self.active_trades) >= self.max_concurrent_trades:
                self.signals_skipped_concurrency += 1
                continue

            if self.duration == 0:
                continue

            # 5. Evaluar Estrategia (ventana acotada como deque(maxlen=3600) en TradingEngine)
            self.signals_attempted += 1
            window_start = max(0, i + 1 - self.max_buffer_size)
            buffer_slice = self.ticks_buffer[window_start : i + 1]

            signal: Optional[Signal] = self.strategy.get_signal(
                data=buffer_slice,
                duration=self.duration,
                inverted_signal=self._inverted_signal,
            )

            if signal is None:
                continue
            self.signals_generated += 1

            # 6. Cadena de Filtros
            filtered_signal: Optional[Signal] = signal
            for flt in self.filters:
                filtered_signal = flt(filtered_signal)
                if filtered_signal is None:
                    break
            if filtered_signal is None:
                continue
            self.signals_passed_filter += 1

            # 7. Cooldown por omisión de señal (igual que TradingEngine._signal_consumer)
            if self._in_cooldown:
                self.signals_omitted_cooldown += 1
                self._cooldown_until_idx = i + filtered_signal.duration
                continue

            # 8. Obtener Stake
            stake = self._get_next_stake()
            if stake is None:
                self.signals_skipped_capital += 1
                continue

            # 9. Ejecución con Delay (T+1)
            next_idx = i + 1
            if next_idx >= len(self.ticks_buffer):
                break

            entry_tick = self.ticks_buffer[next_idx]
            exit_idx = next_idx + filtered_signal.duration

            if exit_idx >= len(self.ticks_buffer):
                break

            # Reservar stake del capital de ronda
            self.remaining_round_capital = round(self.remaining_round_capital - stake, 2)
            matrix_step = len(self.round_trade_results) if not self.use_fixed_stake else 0

            self.active_trades.append(
                {
                    "entry_idx": next_idx,
                    "exit_idx": exit_idx,
                    "direction": filtered_signal.direction,
                    "entry_price": entry_tick.quote,
                    "stake": stake,
                    "signal_epoch": current_tick.epoch,
                    "duration": filtered_signal.duration,
                    "matrix_step": matrix_step,
                    "metadata": getattr(filtered_signal, "extras", {}),
                }
            )

        # Cerrar última ronda si quedó con trades registrados y no se finalizó
        if len(self.round_trade_results) > 0:
            self._finalize_round(reason="END_OF_DATA")

        return self._generate_report()

    def _resolve_expiring_trades(self, current_idx: int, current_tick: Tick):
        """Resuelve trades expirados y evalúa cierres tempranos alineados con TradingEngine."""
        still_active = []
        should_stop = False

        for trade in self.active_trades:
            if trade["exit_idx"] == current_idx:
                exit_price = current_tick.quote
                entry_price = trade["entry_price"]
                stake = trade["stake"]

                is_win = (
                    trade["direction"] == "CALL" and exit_price > entry_price
                ) or (
                    trade["direction"] == "PUT" and exit_price < entry_price
                )

                if is_win:
                    profit = round(stake * (self.payout_pct / 100.0), 2)
                    pnl = profit
                    self.remaining_round_capital = round(
                        self.remaining_round_capital + stake + profit, 2
                    )
                else:
                    pnl = -stake

                self.current_balance = round(self.current_balance + pnl, 2)

                result_str = "win" if is_win else "loss"
                self.round_trade_results.append(result_str)

                # Actualizar racha de pérdidas para cooldown
                should_activate_cooldown = False
                if self._use_cooldown:
                    if result_str == "loss":
                        self._consecutive_losses += 1
                        if self._consecutive_losses >= self._cooldown_threshold:
                            should_activate_cooldown = True
                    else:
                        self._consecutive_losses = 0

                # CORRECCIÓN CRÍTICA: PnL de la ronda respecto a initial_round_capital
                current_round_pnl = round(
                    self.remaining_round_capital - self.initial_round_capital, 2
                )
                if current_round_pnl > self.max_round_pnl:
                    self.max_round_pnl = current_round_pnl

                sim_trade = TradeRecord(
                    entry_epoch=self.ticks_buffer[trade["entry_idx"]].epoch,
                    exit_epoch=current_tick.epoch,
                    direction=trade["direction"],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    duration_ticks=trade["duration"],
                    result=result_str,
                    stake=stake,
                    pnl=pnl,
                    balance_after=self.current_balance,
                    round_id=self.current_round_id,
                    matrix_step=trade["matrix_step"],
                    signal_generation_epoch=trade["signal_epoch"],
                    metadata=trade["metadata"],
                )
                self.completed_trades.append(sim_trade)

                logger.debug(
                    f"📊 Trade: {result_str} | Stake: ${stake:.2f} | "
                    f"PnL: ${pnl:.2f} | RoundPnL: ${current_round_pnl:.2f} | "
                    f"Balance: ${self.current_balance:.2f}"
                )

                # Evaluar cierre temprano usando RecoveryMatrixCalculator (igual que TradingEngine)
                close_decision = self.recovery_calculator.should_close_round(
                    current_pnl=current_round_pnl,
                    target_profit=self.round_target_profit,
                    max_pnl=self.max_round_pnl,
                )

                if close_decision["close"]:
                    reason = (
                        "EARLY_EXIT_PROFIT_THRESHOLD"
                        if close_decision["reason"] == "EARLY_EXIT"
                        else "TRAILING_STOP_ACTIVATED"
                    )
                    logger.info(
                        f"🏁 Cierre temprano: {reason} | PnL Ronda: ${current_round_pnl:.2f}"
                    )
                    self._finalize_round(reason=reason)
                    if not self._initialize_new_round():
                        should_stop = True
                        break
                    continue

                if not self._check_global_limits():
                    should_stop = True
                    break

                if should_activate_cooldown:
                    self._in_cooldown = True
                    self._cooldown_signals_omitted = 0

            else:
                still_active.append(trade)

        self.active_trades = still_active
        if should_stop:
            self._stop_requested = True

    def _generate_report(self) -> Dict[str, Any]:
        """Genera reporte completo con métricas de calidad de señal y rendimiento."""
        total = len(self.completed_trades)
        if total == 0:
            return {"error": "No se completaron trades en el backtest"}

        wins = sum(1 for t in self.completed_trades if t.result == "win")
        gross_profit = sum(t.pnl for t in self.completed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.completed_trades if t.pnl < 0))

        # Racha máxima de pérdidas de trades
        max_consec_loss = curr = 0
        for t in self.completed_trades:
            if t.result == "loss":
                curr += 1
                max_consec_loss = max(max_consec_loss, curr)
            else:
                curr = 0

        # Métricas de rondas
        rounds_won = sum(1 for r in self.round_results if r == "win")
        rounds_lost = sum(1 for r in self.round_results if r == "loss")
        total_rounds = len(self.round_results)

        max_consec_round_loss = curr_round = 0
        for r in self.round_results:
            if r == "loss":
                curr_round += 1
                max_consec_round_loss = max(max_consec_round_loss, curr_round)
            else:
                curr_round = 0

        close_reason_counts: Dict[str, int] = {}
        for reason in self.round_close_reasons:
            close_reason_counts[reason] = close_reason_counts.get(reason, 0) + 1

        report = {
            "strategy": self.strategy.__class__.__name__,
            "capital_mode": "FIXED" if self.use_fixed_stake else "RECOVERY_MATRIX",
            "round_allocation": self.round_allocation,
            "total_ticks": len(self.ticks_buffer),
            "signal_quality": {
                "attempted": self.signals_attempted,
                "generated": self.signals_generated,
                "passed_filter": self.signals_passed_filter,
                "filter_efficiency": round(
                    self.signals_passed_filter / max(1, self.signals_generated), 4
                ),
                "executed": total,
                "omitted_cooldown": self.signals_omitted_cooldown,
                "skipped_concurrency": self.signals_skipped_concurrency,
                "skipped_capital": self.signals_skipped_capital,
            },
            "round_metrics": {
                "total_rounds": total_rounds,
                "rounds_won": rounds_won,
                "rounds_lost": rounds_lost,
                "round_win_rate": round(rounds_won / max(1, total_rounds), 4),
                "max_consecutive_round_losses": max_consec_round_loss,
                "close_reasons": close_reason_counts,
            },
            "performance": {
                "initial_capital": self.initial_capital,
                "final_balance": round(self.current_balance, 2),
                "net_pnl": round(self.current_balance - self.initial_capital, 2),
                "win_rate": round(wins / total, 4),
                "profit_factor": round(gross_profit / gross_loss, 4)
                if gross_loss > 0
                else float("inf"),
                "max_consecutive_losses": max_consec_loss,
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2),
            },
        }

        logger.info(json.dumps(report, indent=2))
        return report


# =============================================================================
# PUNTO DE ENTRADA INDEPENDIENTE
# =============================================================================
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    from strategies.packs_strategy import ZScoreReversionConfig, ZScoreReversionStrategy

    strategy = ZScoreReversionStrategy(
        ZScoreReversionConfig(
            short_window=3,
            long_window=40,
            z_threshold=2.85,
        )
    )

    filters = []
    symbol = "1HZ100V"
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("exports") / f"{symbol}_{run_tag}"

    durations_to_test = list(range(1, 11))
    ticks: list[Tick] = []

    for duration in durations_to_test:
        engine = BacktestingEngine(
            strategy=strategy,
            filters=filters,
            max_concurrent_trades=1,
            payout_pct=90.0,
            initial_capital=40.0,
            round_allocation=10.0,
            take_profit=100.0,
            stop_loss=0.0,
            max_trades=3,
            target_wins=1,
            min_bet=0.35,
            use_fixed_stake=False,
            fixed_stake_amount=1.0,
            duration=duration,
            invert_signal=False,
            inverted_signal=False,
        )

        if len(ticks) == 0:
            await engine.load_ticks(symbol, 1800)
            ticks = engine.ticks_buffer
            TickExporter.to_jsonl(ticks, out_dir / f"ticks_{symbol}.jsonl")
            TickExporter.to_csv(ticks, out_dir / f"ticks_{symbol}.csv")
        else:
            engine.ticks_buffer = ticks

        print(f"----- Duration: {duration} -----")
        engine.run()

        prefix_trades = f"trades_{strategy.__class__.__name__}_{symbol}_dur{duration}t"
        TradeExporter.to_jsonl(engine.completed_trades, out_dir / f"{prefix_trades}.jsonl")
        TradeExporter.to_csv(engine.completed_trades, out_dir / f"{prefix_trades}.csv")


if __name__ == "__main__":
    asyncio.run(main())