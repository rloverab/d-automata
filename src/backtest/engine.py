"""
Motor de Backtesting Determinista v3.0
Alineado con TradingEngine: Gestión de Capital Real, Rondas, Early Exit, Trailing Stop, SL/TP
Arquitectura: Event-Loop Sincrónico con simulación estricta de delay de entrada
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import json

from derivpy import Tick
from core.models import Signal
from filters.base import Filter
from filters.volatility_regime_filter_v2 import VolatilityRegimeFilterV2
from risk.recovery_matrix_calculator import RecoveryMatrixCalculator

logger = logging.getLogger(__name__)


@dataclass
class SimulatedTrade:
    """Representa un trade completado con contexto financiero real."""
    entry_epoch: float
    exit_epoch: float
    direction: str
    entry_price: float
    exit_price: float
    duration_ticks: int
    result: str  # 'win' | 'loss'
    stake: float
    pnl: float
    balance_after: float
    round_id: int
    matrix_step: int  # 0 = base, 1+ = recuperación
    signal_generation_epoch: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class BacktestingEngine:
    """
    Motor de backtesting que replica fielmente la lógica de TradingEngine:
    1. Delay de entrada (señal en T, ejecución en T+1)
    2. Duración real de la orden
    3. Concurrencia parametrizable (default 1)
    4. Cadena de filtros pre-ejecución
    5. Gestión de rondas con RecoveryMatrixCalculator (Early Exit, Trailing Stop)
    6. SL/TP globales con parada inmediata post-trade
    7. Contabilidad sin doble conteo de balance
    """

    def __init__(
        self,
        strategy,
        filters: List[Filter],
        max_concurrent_trades: int = 1,
        payout_pct: float = 0.905,
        initial_capital: float = 100.0,
        round_allocation: float = 15.0,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
        use_fixed_stake: bool = False,
        fixed_stake_amount: float = 1.0,
        recovery_matrix_params: Optional[Dict] = None
    ):
        self.strategy = strategy
        self.filters = filters
        self.max_concurrent_trades = max_concurrent_trades
        self.payout_pct = payout_pct

        # Configuración de Capital
        self.initial_capital = initial_capital
        self.current_balance = initial_capital
        self.round_allocation = round_allocation
        self.take_profit = take_profit
        self.stop_loss = stop_loss

        # Modo de Apuesta
        self.use_fixed_stake = use_fixed_stake
        self.fixed_stake_amount = fixed_stake_amount

        # Recovery Matrix Setup
        rm_params = recovery_matrix_params or {
            "payout_pct": payout_pct * 100,
            "max_trades": 3,
            "target_wins": 1,
            "use_early_exit": False
        }
        self.recovery_calculator = RecoveryMatrixCalculator(**rm_params)

        # Estado Interno
        self.ticks_buffer: List[Tick] = []
        self.active_trades: List[Dict] = []
        self.completed_trades: List[SimulatedTrade] = []

        # Estado de Ronda / Martingala (alineado con TradingEngine)
        self.remaining_round_capital: float = 0.0
        self.round_trade_results: List[str] = []
        self.current_round_id: int = 0
        self.round_starting_balance: float = 0.0
        self.initial_round_capital: float = 0.0
        self.round_target_profit: float = 0.0
        self.max_round_pnl: float = 0.0

        # Tracking de resultados de rondas
        self.round_results: List[str] = []
        self.round_close_reasons: List[str] = []

        # Métricas de Flujo
        self.signals_attempted = 0
        self.signals_generated = 0
        self.signals_passed_filter = 0
        self.signals_skipped_concurrency = 0
        self.signals_skipped_capital = 0

        # Flag de parada
        self._stop_requested = False

    async def load_ticks(self, symbol: str, count: int = 5000):
        """Carga histórico usando MarketDataClient."""
        from derivpy import MarketDataClient
        import math

        client = MarketDataClient()
        try:
            await client.connect()
            all_ticks = []
            if count > 1000:
                batches = math.ceil(count / 1000)
                end = "latest"
                for i in range(batches):
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

    def _initialize_new_round(self) -> bool:
        """Asigna capital y calcula target de ronda (igual que TradingEngine)."""
        if self.current_balance < self.recovery_calculator.min_bet:
            logger.warning("⛔ Balance insuficiente para nueva ronda")
            return False

        allocation = min(self.round_allocation, self.current_balance)
        self.remaining_round_capital = allocation
        self.round_trade_results = []
        self.current_round_id += 1
        self.round_starting_balance = self.current_balance

        # Alineación con TradingEngine._initialize_new_round
        self.initial_round_capital = allocation
        self.round_target_profit = self.recovery_calculator.get_round_target_profit(allocation)
        self.max_round_pnl = 0.0

        logger.debug(
            f"🔄 Ronda #{self.current_round_id} | Capital: ${allocation:.2f} | "
            f"Target: ${self.round_target_profit:.2f}"
        )
        return True

    def _finalize_round(self, reason: str = "UNKNOWN") -> str:
        """Registra cierre de ronda con motivo explícito."""
        round_pnl = self.current_balance - self.round_starting_balance
        result = 'win' if round_pnl > 0 else 'loss'
        self.round_results.append(result)
        self.round_close_reasons.append(reason)

        logger.debug(
            f"🏁 Ronda #{self.current_round_id}: {result} | "
            f"Motivo: {reason} | PnL: ${round_pnl:.2f}"
        )
        return result

    def _get_next_stake(self) -> Optional[float]:
        """Calcula el stake basado en modo fijo o matriz de recuperación."""
        if self.use_fixed_stake:
            stake = self.fixed_stake_amount
        else:
            bet_info = self.recovery_calculator.calculate_next_bet(
                self.round_trade_results,
                self.remaining_round_capital
            )
            if bet_info["status"] != "ACTIVE":
                return None
            stake = bet_info["next_bet"]

        # Validación de seguridad
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
            return {"error": "Capital inicial insuficiente"}

        logger.info(
            f"🚀 Backtest | Strat: {self.strategy.__class__.__name__} | "
            f"Mode: {'Fixed' if self.use_fixed_stake else 'Matrix'} | "
            f"Concurrent: {self.max_concurrent_trades}"
        )

        start_idx = 50
        for i in range(start_idx, len(self.ticks_buffer)):
            current_tick = self.ticks_buffer[i]

            # 1. Resolver trades expirados
            self._resolve_expiring_trades(i, current_tick)

            # Parada inmediata por SL/TP o flag interno
            if self._stop_requested:
                break
            if self.take_profit and self.current_balance >= self.take_profit:
                logger.info(f"🎯 TP GLOBAL: ${self.current_balance:.2f}")
                break
            if self.stop_loss and self.current_balance <= self.stop_loss:
                logger.info(f"🛑 SL GLOBAL: ${self.current_balance:.2f}")
                break

            # 2. Verificar estado de ronda ANTES de buscar señal
            # Equivalente a TradingEngine._ensure_active_round()
            bet_check = self.recovery_calculator.calculate_next_bet(
                self.round_trade_results, self.remaining_round_capital
            )
            if bet_check["status"] != "ACTIVE":
                self._finalize_round(reason=bet_check["status"])
                if not self._initialize_new_round():
                    break
                continue  # Re-evaluar en siguiente tick con nueva ronda

            # 3. Concurrencia
            if len(self.active_trades) >= self.max_concurrent_trades:
                self.signals_skipped_concurrency += 1
                continue

            # 4. Evaluar Estrategia
            self.signals_attempted += 1
            buffer_slice = self.ticks_buffer[:i + 1]

            signal: Optional[Signal] = self.strategy.get_signal(
                data=buffer_slice, duration=8
            )

            if signal is None:
                continue
            self.signals_generated += 1

            # 5. Filtros
            filtered_signal = signal
            for f in self.filters:
                filtered_signal = f(filtered_signal)
                if filtered_signal is None:
                    break
            if filtered_signal is None:
                continue
            self.signals_passed_filter += 1

            # 6. Obtener Stake
            stake = self._get_next_stake()
            if stake is None:
                self._finalize_round(reason="INSUFFICIENT_STAKE")
                if not self._initialize_new_round():
                    break
                stake = self._get_next_stake()
                if stake is None:
                    self.signals_skipped_capital += 1
                    continue

            # 7. Ejecución con Delay (T+1)
            next_idx = i + 1
            if next_idx >= len(self.ticks_buffer):
                break

            entry_tick = self.ticks_buffer[next_idx]
            exit_idx = next_idx + filtered_signal.duration

            if exit_idx >= len(self.ticks_buffer):
                break

            # CORRECCIÓN: Solo reservar del capital de ronda, NO de current_balance
            self.remaining_round_capital -= stake

            matrix_step = len(self.round_trade_results) if not self.use_fixed_stake else 0

            self.active_trades.append({
                'entry_idx': next_idx,
                'exit_idx': exit_idx,
                'direction': filtered_signal.direction,
                'entry_price': entry_tick.quote,
                'stake': stake,
                'signal_epoch': current_tick.epoch,
                'duration': filtered_signal.duration,
                'matrix_step': matrix_step,
                'metadata': getattr(filtered_signal, 'extras', {})
            })

        # Cerrar última ronda si quedó activa
        if len(self.round_trade_results) > 0:
            self._finalize_round(reason="END_OF_DATA")

        return self._generate_report()

    def _resolve_expiring_trades(self, current_idx: int, current_tick: Tick):
        """Resuelve trades expirados con contabilidad limpia y evaluación de cierre temprano."""
        still_active = []
        should_stop = False

        for trade in self.active_trades:
            if trade['exit_idx'] == current_idx:
                exit_price = current_tick.quote
                entry_price = trade['entry_price']
                stake = trade['stake']

                is_win = (trade['direction'] == 'CALL' and exit_price > entry_price) or \
                         (trade['direction'] == 'PUT' and exit_price < entry_price)

                # Contabilidad limpia: sin doble conteo
                if is_win:
                    profit = stake * self.payout_pct
                    pnl = profit
                else:
                    profit = 0.0
                    pnl = -stake

                self.current_balance += pnl
                self.remaining_round_capital += stake + pnl  # Devolver stake al pool + resultado

                result_str = 'win' if is_win else 'loss'
                self.round_trade_results.append(result_str)

                # Actualizar max_round_pnl para Trailing Stop
                current_round_pnl = self.current_balance - self.initial_round_capital
                if current_round_pnl > self.max_round_pnl:
                    self.max_round_pnl = current_round_pnl

                sim_trade = SimulatedTrade(
                    entry_epoch=self.ticks_buffer[trade['entry_idx']].epoch,
                    exit_epoch=current_tick.epoch,
                    direction=trade['direction'],
                    entry_price=entry_price,
                    exit_price=exit_price,
                    duration_ticks=trade['duration'],
                    result=result_str,
                    stake=stake,
                    pnl=pnl,
                    balance_after=self.current_balance,
                    round_id=self.current_round_id,
                    matrix_step=trade['matrix_step'],
                    signal_generation_epoch=trade['signal_epoch'],
                    metadata=trade['metadata']
                )
                self.completed_trades.append(sim_trade)

                logger.debug(
                    f"📊 Trade: {result_str} | Stake: ${stake:.2f} | "
                    f"PnL: ${pnl:.2f} | Balance: ${self.current_balance:.2f}"
                )

                # Evaluar cierre temprano de ronda (Early Exit + Trailing Stop)
                close_decision = self.recovery_calculator.should_close_round(
                    current_pnl=current_round_pnl,
                    target_profit=self.round_target_profit,
                    max_pnl=self.max_round_pnl
                )

                if close_decision["close"]:
                    logger.info(
                        f"🏁 Cierre temprano: {close_decision['reason']} | "
                        f"PnL Ronda: ${current_round_pnl:.2f}"
                    )
                    self._finalize_round(reason=close_decision["reason"])
                    self._initialize_new_round()

                # Verificación SL/TP global inmediata post-trade
                if self.take_profit and self.current_balance >= self.take_profit:
                    logger.info(f"🎯 TP GLOBAL: ${self.current_balance:.2f}")
                    should_stop = True
                    break
                if self.stop_loss and self.current_balance <= self.stop_loss:
                    logger.info(f"🛑 SL GLOBAL: ${self.current_balance:.2f}")
                    should_stop = True
                    break

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

        wins = sum(1 for t in self.completed_trades if t.result == 'win')
        losses = total - wins
        gross_profit = sum(t.pnl for t in self.completed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.completed_trades if t.pnl < 0))

        # Racha máxima de pérdidas de trades
        max_consec_loss = curr = 0
        for t in self.completed_trades:
            if t.result == 'loss':
                curr += 1
                max_consec_loss = max(max_consec_loss, curr)
            else:
                curr = 0

        # Métricas de rondas
        rounds_won = sum(1 for r in self.round_results if r == 'win')
        rounds_lost = sum(1 for r in self.round_results if r == 'loss')
        total_rounds = len(self.round_results)

        max_consec_round_loss = curr_round = 0
        for r in self.round_results:
            if r == 'loss':
                curr_round += 1
                max_consec_round_loss = max(max_consec_round_loss, curr_round)
            else:
                curr_round = 0

        # Distribución de motivos de cierre
        close_reason_counts = {}
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
                "skipped_concurrency": self.signals_skipped_concurrency,
                "skipped_capital": self.signals_skipped_capital
            },
            "round_metrics": {
                "total_rounds": total_rounds,
                "rounds_won": rounds_won,
                "rounds_lost": rounds_lost,
                "round_win_rate": round(rounds_won / max(1, total_rounds), 4),
                "max_consecutive_round_losses": max_consec_round_loss,
                "close_reasons": close_reason_counts
            },
            "performance": {
                "initial_capital": self.initial_capital,
                "final_balance": round(self.current_balance, 2),
                "net_pnl": round(self.current_balance - self.initial_capital, 2),
                "win_rate": round(wins / total, 4),
                "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else float('inf'),
                "max_consecutive_losses": max_consec_loss,
                "gross_profit": round(gross_profit, 2),
                "gross_loss": round(gross_loss, 2)
            }
        }

        logger.info(json.dumps(report, indent=2))
        return report


# =============================================================================
# PUNTO DE ENTRADA INDEPENDIENTE
# =============================================================================
async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )

    # Importar estrategia y filtros según necesidad
    from strategies.momentum_pullback_velocity_strategy import MomentumPullbackVelocityStrategy
    # from filters.volatility_regime_filter_v2 import VolatilityRegimeFilterV2

    strategy = MomentumPullbackVelocityStrategy(
        impulse_ticks=1,
        min_velocity_ratio=1.1
    )

    filters = [
        VolatilityRegimeFilterV2(
            # min_efficiency=0.3,
            spike_multiplier=2.5,
            window=30
        ),
    ]

    engine = BacktestingEngine(
        strategy=strategy,
        filters=filters,
        max_concurrent_trades=1,
        initial_capital=40.0,
        round_allocation=15.0,
        take_profit=1000.0,
        stop_loss=0.0,
        use_fixed_stake=False,
        fixed_stake_amount=1.0,
        recovery_matrix_params={
            "payout_pct": 90,
            "max_trades": 2,
            "target_wins": 1,
            "use_early_exit": False,
            "early_exit_threshold": 0.9,
            "use_trailing_stop": False,
            "trailing_stop_pct": 0.40
        }
    )

    await engine.load_ticks("1HZ100V", int(12*3600/1))
    report = engine.run()

    # Opcional: Guardar en CouchDB
    # db.save({"type": "backtest_report", "timestamp": datetime.utcnow().isoformat(), **report})


if __name__ == "__main__":
    asyncio.run(main())