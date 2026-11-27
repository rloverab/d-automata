"""
Optimizador Dinámico de Duración y Polaridad de Señal (Pydantic V2)
Ejecuta backtests internos sobre el buffer de ticks usando scoring ponderado e histéresis.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from derivpy import Tick
from pydantic import BaseModel, ConfigDict, Field

from filters.base import Filter

logger = logging.getLogger(__name__)


class DurationReport(BaseModel):
    """Resultado de simulación para una combinación de duración y polaridad de señal."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duration: int = Field(..., ge=1)
    inverted_signal: bool = False
    total_trades: int = Field(..., ge=0)
    wins: int = Field(..., ge=0)
    losses: int = Field(..., ge=0)
    win_rate: float = Field(..., ge=0.0, le=1.0)
    profit_factor: float = Field(..., ge=0.0)
    max_consecutive_losses: int = Field(..., ge=0)
    net_pnl: float = 0.0
    signals_generated: int = Field(default=0, ge=0)
    signals_passed_filter: int = Field(default=0, ge=0)
    score: float = Field(default=0.0, ge=0.0)


class OptimizationResult(BaseModel):
    """Resultado final de una corrida de optimización."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    duration: Optional[int] = None
    inverted_signal: Optional[bool] = None
    best_report: Optional[DurationReport] = None


class OptimizerWeights(BaseModel):
    """Pesos ponderados para el cálculo de score."""

    win_rate: float = Field(default=0.35, ge=0.0)
    profit_factor: float = Field(default=0.25, ge=0.0)
    consecutive_losses: float = Field(default=0.20, ge=0.0)
    sample_size: float = Field(default=0.10, ge=0.0)
    filter_efficiency: float = Field(default=0.10, ge=0.0)

    def normalized(self) -> Dict[str, float]:
        total = (
            self.win_rate
            + self.profit_factor
            + self.consecutive_losses
            + self.sample_size
            + self.filter_efficiency
        )
        if total <= 0:
            return {
                "win_rate": 0.35,
                "profit_factor": 0.25,
                "consecutive_losses": 0.20,
                "sample_size": 0.10,
                "filter_efficiency": 0.10,
            }
        return {
            "win_rate": self.win_rate / total,
            "profit_factor": self.profit_factor / total,
            "consecutive_losses": self.consecutive_losses / total,
            "sample_size": self.sample_size / total,
            "filter_efficiency": self.filter_efficiency / total,
        }


class DurationOptimizer:
    """
    Optimizador que evalúa duraciones (y opcionalmente polaridad de señal)
    mediante backtesting interno y selección basada en score ponderado + histéresis.
    """

    def __init__(
        self,
        strategy: Any,
        filters: List[Filter],
        duration_range: range = range(1, 11),
        min_win_rate: float = 0.60,
        max_trades_per_round: int = 3,
        payout_pct: float = 90.0,
        min_sample_size: int = 5,
        hysteresis_margin: float = 0.10,
        evaluate_inversion: bool = False,
        max_buffer_size: int = 3600,
        weights: Optional[OptimizerWeights] = None,
    ):
        self.strategy = strategy
        self.filters = filters
        self.duration_range = duration_range
        self.min_win_rate = min_win_rate
        self.max_trades_per_round = max(1, max_trades_per_round)
        self.payout_pct = payout_pct
        self.min_sample_size = min_sample_size
        self.hysteresis_margin = hysteresis_margin
        self.evaluate_inversion = evaluate_inversion
        self.max_buffer_size = max_buffer_size
        self.weights = (weights or OptimizerWeights()).normalized()

        # Estado interno
        self.current_optimal_duration: Optional[int] = None
        self.current_optimal_inverted: Optional[bool] = None
        self.last_optimization_epoch: float = 0.0
        self.optimization_history: List[Dict[str, Any]] = []

    async def optimize(
        self,
        ticks_buffer: List[Tick],
        current_duration: int,
        current_inverted: bool = False,
    ) -> Optional[int]:
        """Retorna únicamente la duración seleccionada (compatibilidad con llamadas simples)."""
        decision = await self.optimize_with_inversion(
            ticks_buffer=ticks_buffer,
            current_duration=current_duration,
            current_inverted=current_inverted,
        )
        return decision.duration

    async def optimize_with_inversion(
        self,
        ticks_buffer: List[Tick],
        current_duration: int,
        current_inverted: bool = False,
    ) -> OptimizationResult:
        """
        Ejecuta backtests internos para cada duración (y señal normal/invertida si está activo)
        y selecciona la mejor opción mediante score ponderado e histéresis.
        """
        if len(ticks_buffer) < 100:
            return OptimizationResult(duration=None, inverted_signal=None)

        polarities = [False, True] if self.evaluate_inversion else [current_inverted]
        reports: List[DurationReport] = []

        for is_inv in polarities:
            for duration in self.duration_range:
                try:
                    report = await self._run_backtest(
                        ticks_buffer=ticks_buffer,
                        duration=duration,
                        inverted_signal=is_inv,
                    )
                    if report is not None:
                        reports.append(report)
                except Exception as exc:
                    logger.warning(
                        f"Error en backtest interno (duration={duration}, inverted={is_inv}): {exc}"
                    )

        if not reports:
            return OptimizationResult(duration=None, inverted_signal=None)

        self._calculate_scores(reports)
        best_report = max(reports, key=lambda r: r.score)

        selected_duration, selected_inverted = self._apply_hysteresis(
            best_report=best_report,
            current_duration=current_duration,
            current_inverted=current_inverted,
        )

        self.current_optimal_duration = selected_duration
        self.current_optimal_inverted = selected_inverted
        self.last_optimization_epoch = float(ticks_buffer[-1].epoch)

        self.optimization_history.append(
            {
                "timestamp": ticks_buffer[-1].epoch,
                "current_duration": current_duration,
                "current_inverted": current_inverted,
                "best_duration": best_report.duration,
                "best_inverted": best_report.inverted_signal,
                "selected_duration": selected_duration,
                "selected_inverted": selected_inverted,
                "best_score": best_report.score,
                "reports_summary": [
                    {
                        "duration": r.duration,
                        "inverted": r.inverted_signal,
                        "win_rate": r.win_rate,
                        "profit_factor": r.profit_factor,
                        "max_consecutive_losses": r.max_consecutive_losses,
                        "net_pnl": r.net_pnl,
                        "trades": r.total_trades,
                        "score": r.score,
                    }
                    for r in reports
                ],
            }
        )

        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]

        return OptimizationResult(
            duration=selected_duration,
            inverted_signal=selected_inverted,
            best_report=best_report,
        )

    async def _run_backtest(
        self,
        ticks_buffer: List[Tick],
        duration: int,
        inverted_signal: bool = False,
    ) -> Optional[DurationReport]:
        """
        Simula trades en el buffer de ticks con delay T -> T+1, duración real en ticks
        y concurrencia = 1, alineado con BacktestingEngine y TradingEngine.
        """
        if len(ticks_buffer) < 100 or duration <= 0:
            return None

        payout_mult = self.payout_pct / 100.0
        wins = 0
        losses = 0
        max_consec_losses = 0
        current_consec_losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        signals_generated = 0
        signals_passed_filter = 0

        start_idx = 50
        active_exit_idx: Optional[int] = None

        for i in range(start_idx, len(ticks_buffer)):
            if active_exit_idx is not None and i < active_exit_idx:
                continue
            if active_exit_idx is not None and i == active_exit_idx:
                active_exit_idx = None

            window_start = max(0, i + 1 - self.max_buffer_size)
            buffer_slice = ticks_buffer[window_start : i + 1]

            signal = self.strategy.get_signal(
                data=buffer_slice,
                duration=duration,
                inverted_signal=inverted_signal,
            )
            if signal is None:
                continue

            signals_generated += 1

            filtered_signal = signal
            for flt in self.filters:
                filtered_signal = flt(filtered_signal)
                if filtered_signal is None:
                    break

            if filtered_signal is None:
                continue

            signals_passed_filter += 1

            entry_idx = i + 1
            exit_idx = entry_idx + filtered_signal.duration
            if exit_idx >= len(ticks_buffer):
                break

            entry_price = ticks_buffer[entry_idx].quote
            exit_price = ticks_buffer[exit_idx].quote

            is_win = (
                filtered_signal.direction == "CALL" and exit_price > entry_price
            ) or (
                filtered_signal.direction == "PUT" and exit_price < entry_price
            )

            if is_win:
                wins += 1
                gross_profit += payout_mult
                current_consec_losses = 0
            else:
                losses += 1
                gross_loss += 1.0
                current_consec_losses += 1
                max_consec_losses = max(max_consec_losses, current_consec_losses)

            active_exit_idx = exit_idx

        total_trades = wins + losses
        if total_trades < self.min_sample_size:
            return None

        win_rate = wins / total_trades
        raw_pf = (gross_profit / gross_loss) if gross_loss > 0 else 10.0

        return DurationReport(
            duration=duration,
            inverted_signal=inverted_signal,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=round(win_rate, 4),
            profit_factor=round(min(raw_pf, 10.0), 4),
            max_consecutive_losses=max_consec_losses,
            net_pnl=round(gross_profit - gross_loss, 4),
            signals_generated=signals_generated,
            signals_passed_filter=signals_passed_filter,
        )

    def _calculate_scores(self, reports: List[DurationReport]) -> None:
        """Calcula el score ponderado de cada candidato usando los pesos configurados."""
        if not reports:
            return

        max_pf = max(r.profit_factor for r in reports) or 1.0
        max_trades = max(r.total_trades for r in reports) or 1

        for report in reports:
            wr_score = min(report.win_rate / max(0.01, self.min_win_rate), 1.0)
            pf_score = min(report.profit_factor / max_pf, 1.0) if max_pf > 0 else 0.0
            cl_score = max(
                0.0,
                1.0 - (report.max_consecutive_losses / (self.max_trades_per_round + 1)),
            )
            ss_score = min(report.total_trades / max_trades, 1.0)
            fe_score = (
                report.signals_passed_filter / report.signals_generated
                if report.signals_generated > 0
                else 0.0
            )

            weighted_score = (
                self.weights["win_rate"] * wr_score
                + self.weights["profit_factor"] * pf_score
                + self.weights["consecutive_losses"] * cl_score
                + self.weights["sample_size"] * ss_score
                + self.weights["filter_efficiency"] * fe_score
            )

            report.score = round(weighted_score, 4)

    def _apply_hysteresis(
        self,
        best_report: DurationReport,
        current_duration: int,
        current_inverted: bool,
    ) -> Tuple[int, bool]:
        """Evita cambios frecuentes si la mejora en score no supera el margen de histéresis."""
        if self.current_optimal_duration is None or not self.optimization_history:
            return best_report.duration, best_report.inverted_signal

        if (
            best_report.duration == current_duration
            and best_report.inverted_signal == current_inverted
        ):
            return current_duration, current_inverted

        current_entry = next(
            (
                r
                for r in self.optimization_history[-1].get("reports_summary", [])
                if r["duration"] == current_duration
                and r["inverted"] == current_inverted
            ),
            None,
        )

        if current_entry is None:
            return best_report.duration, best_report.inverted_signal

        improvement = best_report.score - float(current_entry.get("score", 0.0))
        if improvement > self.hysteresis_margin:
            logger.info(
                f"📈 Cambio justificado: dur {current_duration} (inv={current_inverted}) → "
                f"dur {best_report.duration} (inv={best_report.inverted_signal}) "
                f"(mejora de score: +{improvement:.3f})"
            )
            return best_report.duration, best_report.inverted_signal

        logger.debug(
            f"📊 Manteniendo configuración actual: dur {current_duration} "
            f"(inv={current_inverted}, mejora: +{improvement:.3f} <= {self.hysteresis_margin:.3f})"
        )
        return current_duration, current_inverted

    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Retorna un resumen estructurado de la última optimización."""
        if not self.optimization_history:
            return {"status": "no_data"}

        last = self.optimization_history[-1]
        return {
            "timestamp": last["timestamp"],
            "current_duration": last["current_duration"],
            "current_inverted": last["current_inverted"],
            "recommended_duration": last["selected_duration"],
            "recommended_inverted": last["selected_inverted"],
            "best_score": last["best_score"],
            "top_3_candidates": sorted(
                last["reports_summary"],
                key=lambda x: x["score"],
                reverse=True,
            )[:3],
        }