"""
Motor de Backtesting Determinista v2
Incluye: Gestión de Capital Real (Martingala/Fixed), Rondas, SL/TP Globales
Arquitectura: Event-Loop Sincrónico con simulación de delay de entrada
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging
import json

from derivpy import Tick
from config.filters_config import load_filters_config
from core.models import Signal
from filters.base import Filter
# Ajusta esta importación según tu estructura real
from filters.spread_trend_filter import SpreadTrendFilter
from filters.volatility_filter import VolatilityFilter
from filters.volatility_regime_filter import VolatilityRegimeFilter
from filters.volatility_regime_filter_v2 import VolatilityRegimeFilterV2
from risk.recovery_matrix_calculator import RecoveryMatrixCalculator
from strategies.momentum_pullback_adaptive_strategy import MomentumPullbackAdaptiveStrategy
from strategies.momentum_pullback_five_strategy import MomentumPullbackFive
from strategies.momentum_pullback_velocity_strategy import MomentumPullbackVelocityStrategy 

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
        
        # Estado de Ronda / Martingala
        self.remaining_round_capital = 0.0
        self.round_trade_results: List[str] = []
        self.current_round_id = 0
        
        # Métricas de Flujo
        self.signals_attempted = 0
        self.signals_generated = 0
        self.signals_passed_filter = 0
        self.signals_skipped_concurrency = 0
        self.signals_skipped_capital = 0

    async def load_ticks(self, symbol: str, count: int = 5000):
        """Carga histórico (lógica existente preservada)."""
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
                    if not ts: break
                    all_ticks = ts + all_ticks
                    end = ts[0].epoch - 1
                    await asyncio.sleep(0.3)
            else:
                all_ticks, _ = await client.ticks_history(symbol=symbol, count=count)
                
            self.ticks_buffer = sorted(all_ticks, key=lambda t: t.epoch)
            logger.info(f"✅ Cargados {len(self.ticks_buffer)} ticks")
        finally:
            await client.disconnect()

    def _initialize_new_round(self):
        """Asigna capital para una nueva sesión de martingala."""
        if self.current_balance < self.recovery_calculator.min_bet:
            logger.warning("⛔ Balance insuficiente para nueva ronda")
            return False
            
        allocation = min(self.round_allocation, self.current_balance)
        self.remaining_round_capital = allocation
        self.round_trade_results = []
        self.current_round_id += 1
        logger.debug(f"🔄 Nueva Ronda #{self.current_round_id} | Asignado: ${allocation:.2f}")
        return True

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

    def run(self):
        if len(self.ticks_buffer) < 100:
            raise ValueError("Buffer insuficiente")

        # Iniciar primera ronda
        if not self._initialize_new_round():
            return {"error": "Capital inicial insuficiente"}

        logger.info(f"🚀 Backtest | Strat: {self.strategy.__class__.__name__} | Mode: {'Fixed' if self.use_fixed_stake else 'Matrix'}")
        
        start_idx = 50 
        for i in range(start_idx, len(self.ticks_buffer)):
            current_tick = self.ticks_buffer[i]
            
            # 1. Resolver trades expirados
            self._resolve_expiring_trades(i, current_tick)
            
            # Verificar SL/TP Global después de resolver trades
            if self.take_profit and self.current_balance >= self.take_profit:
                logger.info(f"🎯 TAKE PROFIT alcanzado: ${self.current_balance:.2f}")
                break
            if self.stop_loss and self.current_balance <= self.stop_loss:
                logger.info(f"🛑 STOP LOSS alcanzado: ${self.current_balance:.2f}")
                break
            
            # 2. Concurrencia
            if len(self.active_trades) >= self.max_concurrent_trades:
                self.signals_skipped_concurrency += 1
                continue
            
            # 3. Evaluar Estrategia
            self.signals_attempted += 1
            buffer_slice = self.ticks_buffer[:i+1]
            
            signal: Optional[Signal] = self.strategy.get_signal(
                data=buffer_slice, duration=1
            )
            
            if signal is None:
                continue
            self.signals_generated += 1
            
            # 4. Filtros
            filtered_signal = signal
            for f in self.filters:
                filtered_signal = f(filtered_signal)
                if filtered_signal is None:
                    break
            if filtered_signal is None:
                continue
            self.signals_passed_filter += 1
            
            # 5. Validar Capital y Obtener Stake
            stake = self._get_next_stake()
            if stake is None:
                # Ronda agotada o sin capital, intentar nueva ronda
                if not self._initialize_new_round():
                    break  # Fin definitivo
                stake = self._get_next_stake()
                if stake is None:
                    self.signals_skipped_capital += 1
                    continue
            
            # 6. Ejecución con Delay (T+1)
            next_idx = i + 1
            if next_idx >= len(self.ticks_buffer):
                break
                
            entry_tick = self.ticks_buffer[next_idx]
            exit_idx = next_idx + filtered_signal.duration
            
            if exit_idx >= len(self.ticks_buffer):
                break
                
            # Descontar stake preventivamente (reserva)
            self.remaining_round_capital -= stake
            self.current_balance -= stake
            
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

        return self._generate_report()

    def _resolve_expiring_trades(self, current_idx: int, current_tick: Tick):
        still_active = []
        for trade in self.active_trades:
            if trade['exit_idx'] == current_idx:
                exit_price = current_tick.quote
                entry_price = trade['entry_price']
                stake = trade['stake']
                
                is_win = (trade['direction'] == 'CALL' and exit_price > entry_price) or \
                         (trade['direction'] == 'PUT' and exit_price < entry_price)
                
                profit = stake * self.payout_pct if is_win else 0.0
                pnl = profit - stake  # Neto real
                
                # Actualizar balances
                self.current_balance += stake + profit  # Devuelve stake + ganancia
                self.remaining_round_capital += stake + profit
                
                result_str = 'win' if is_win else 'loss'
                self.round_trade_results.append(result_str)
                
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
            else:
                still_active.append(trade)
        self.active_trades = still_active

    def _generate_report(self) -> Dict[str, Any]:
        total = len(self.completed_trades)
        if total == 0:
            return {"error": "No trades completed"}
            
        wins = sum(1 for t in self.completed_trades if t.result == 'win')
        gross_profit = sum(t.pnl for t in self.completed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.completed_trades if t.pnl < 0))
        
        max_consec_loss = curr = 0
        for t in self.completed_trades:
            if t.result == 'loss':
                curr += 1
                max_consec_loss = max(max_consec_loss, curr)
            else:
                curr = 0
        
        return {
            "strategy": self.strategy.__class__.__name__,
            "capital_mode": "FIXED" if self.use_fixed_stake else "RECOVERY_MATRIX",
            "round_allocation": self.round_allocation,
            "total_ticks": len(self.ticks_buffer),
            "signal_quality": {
                "attempted": self.signals_attempted,
                "generated": self.signals_generated,
                "passed_filter": self.signals_passed_filter,
                "filter_efficiency": round(self.signals_passed_filter / max(1, self.signals_generated), 4),
                "executed": total,
                "skipped_concurrency": self.signals_skipped_concurrency,
                "skipped_capital": self.signals_skipped_capital
            },
            "performance": {
                "initial_capital": self.initial_capital,
                "final_balance": round(self.current_balance, 2),
                "net_pnl": round(self.current_balance - self.initial_capital, 2),
                "win_rate": round(wins / total, 4),
                "profit_factor": round(gross_profit / gross_loss, 4) if gross_loss > 0 else float('inf'),
                "max_consecutive_losses": max_consec_loss,
                "total_rounds": self.current_round_id
            }
        }


# =============================================================================
# EJEMPLO DE USO
# =============================================================================
async def main():
    logging.basicConfig(level=logging.INFO)

    strategy = MomentumPullbackVelocityStrategy(
        impulse_ticks=1, #3
        min_velocity_ratio=1.1 #1.25
    )
    # strategy = MomentumPullbackAdaptiveStrategy(impulse_ticks=1)
    # strategy = MomentumPullbackFive()
    # filters_params = load_filters_config()
    filters = [
        VolatilityRegimeFilterV2(
            min_efficiency=0.30, #0.30,
            spike_multiplier=2.5, #2.5
            window=16
        ),
        # VolatilityRegimeFilter(
        #     min_efficiency=0.30,
        #     max_volatility_percentile=100            
        # ),
        # SpreadTrendFilter(window_size=10, min_ticks=15),
        # VolatilityFilter(filters_params.volatility)
    ]
    
    engine = BacktestingEngine(
        strategy=strategy,  # Tu estrategia aquí
        filters=filters,     # Tus filtros aquí
        initial_capital=40.0,
        round_allocation=15.0,
        take_profit=240.0,      # 6x capital
        stop_loss=0.0,         # 0.4x capital
        use_fixed_stake=False,  # ← Cambiar a True para probar sin martingala
        fixed_stake_amount=1.0,
        recovery_matrix_params={
            "payout_pct": 90.0,
            "max_trades": 4,
            "target_wins": 1,
            "use_early_exit": False
        }
    )
    
    await engine.load_ticks("1HZ100V", 12*3600)
    report = engine.run()
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    asyncio.run(main())