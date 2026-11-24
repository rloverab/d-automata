"""
Optimizador Dinámico de Duración de Trades
Ejecuta backtests internos para determinar la duración óptima
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import logging

from derivpy import Tick
from filters.base import Filter

logger = logging.getLogger(__name__)


@dataclass
class DurationReport:
    """Resultado del backtest para una duración específica."""
    duration: int
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    max_consecutive_losses: int
    net_pnl: float
    signals_generated: int
    signals_passed_filter: int
    score: float = 0.0  # Score ponderado para selección


class DurationOptimizer:
    """
    Optimizador que ejecuta backtests internos para determinar
    la duración óptima de trades en tiempo real.
    """
    
    def __init__(
        self,
        strategy,
        filters: List[Filter],
        duration_range: range = range(1, 11),  # 1 a 10 ticks
        min_win_rate: float = 0.60,
        max_trades_per_round: int = 3,
        payout_pct: float = 91.5,
        min_sample_size: int = 5,  # Mínimo de trades para considerar válido
        hysteresis_margin: float = 0.10  # Margen para evitar cambios frecuentes
    ):
        self.strategy = strategy
        self.filters = filters
        self.duration_range = duration_range
        self.min_win_rate = min_win_rate
        self.max_trades_per_round = max_trades_per_round
        self.payout_pct = payout_pct
        self.min_sample_size = min_sample_size
        self.hysteresis_margin = hysteresis_margin
        
        # Estado
        self.current_optimal_duration: Optional[int] = None
        self.last_optimization_epoch: float = 0
        self.optimization_history: List[Dict] = []
        
        # Pesos para scoring ponderado
        self.weights = {
            'win_rate': 0.35,
            'profit_factor': 0.25,
            'consecutive_losses': 0.20,
            'sample_size': 0.10,
            'filter_efficiency': 0.10
        }

    async def optimize(
        self, 
        ticks_buffer: List[Tick],
        current_duration: int
    ) -> Optional[int]:
        """
        Ejecuta backtests para todas las duraciones y retorna la óptima.
        
        Args:
            ticks_buffer: Copia del buffer de ticks actual
            current_duration: Duración actualmente en uso
            
        Returns:
            Nueva duración óptima, o None si no se debe cambiar
        """
        # logger.info(f"🔬 Iniciando optimización de duración | Ticks: {len(ticks_buffer)}")
        
        reports: List[DurationReport] = []
        
        # Ejecutar backtest para cada duración
        for duration in self.duration_range:
            try:
                report = await self._run_backtest(
                    ticks_buffer=ticks_buffer,
                    duration=duration
                )
                if report:
                    reports.append(report)
                    logger.debug(
                        f"  Duration {duration}: WR={report.win_rate:.1%} | "
                        f"PF={report.profit_factor:.2f} | "
                        f"Trades={report.total_trades}"
                    )
            except Exception as e:
                logger.warning(f"Error en backtest duration={duration}: {e}")
                continue
        
        if not reports:
            # logger.warning("⚠️ No se generaron reportes válidos")
            return None
            # return 0
        
        # Calcular scores ponderados
        self._calculate_scores(reports)
        
        # Seleccionar mejor duración
        best_report = max(reports, key=lambda r: r.score)
        
        # Aplicar criterio de histéresis
        new_duration = self._apply_hysteresis(best_report, current_duration)
        
        # Registrar en historial
        self.optimization_history.append({
            'timestamp': ticks_buffer[-1].epoch if ticks_buffer else 0,
            'current_duration': current_duration,
            'best_duration': best_report.duration,
            'selected_duration': new_duration,
            'best_score': best_report.score,
            'reports_summary': [
                {
                    'duration': r.duration,
                    'win_rate': r.win_rate,
                    'profit_factor': r.profit_factor,
                    'trades': r.total_trades,
                    'score': r.score
                } for r in reports
            ]
        })
        
        # Mantener historial acotado
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        # logger.info(
        #     f"✅ Optimización completa | "
        #     f"Mejor: {best_report.duration} (score={best_report.score:.3f}) | "
        #     f"Seleccionada: {new_duration}"
        # )
        
        return new_duration

    async def _run_backtest(
        self, 
        ticks_buffer: List[Tick], 
        duration: int
    ) -> Optional[DurationReport]:
        """
        Ejecuta un backtest simplificado para una duración específica.
        Versión optimizada que solo calcula métricas esenciales.
        """
        if len(ticks_buffer) < 100:
            return None
        
        wins = 0
        losses = 0
        max_consec_losses = 0
        current_consec_losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        signals_generated = 0
        signals_passed_filter = 0
        
        # Simulación simplificada (sin gestión de capital compleja)
        start_idx = 50
        i = start_idx
        
        while i < len(ticks_buffer) - duration - 1:
            # Evaluar señal en tick i
            buffer_slice = ticks_buffer[:i + 1]
            
            signal = self.strategy.get_signal(
                data=buffer_slice,
                duration=duration
            )
            
            if signal is None:
                i += 1
                continue
            
            signals_generated += 1
            
            # Aplicar filtros
            filtered_signal = signal
            for f in self.filters:
                filtered_signal = f(filtered_signal)
                if filtered_signal is None:
                    break
            
            if filtered_signal is None:
                i += 1
                continue
            
            signals_passed_filter += 1
            
            # Entrada en tick i+1, salida en tick i+1+duration
            entry_idx = i + 1
            exit_idx = entry_idx + duration
            
            if exit_idx >= len(ticks_buffer):
                break
            
            entry_price = ticks_buffer[entry_idx].quote
            exit_price = ticks_buffer[exit_idx].quote
            
            # Determinar resultado
            is_win = (
                (filtered_signal.direction == 'CALL' and exit_price > entry_price) or
                (filtered_signal.direction == 'PUT' and exit_price < entry_price)
            )
            
            if is_win:
                wins += 1
                gross_profit += 1.0  # Normalizado
                current_consec_losses = 0
            else:
                losses += 1
                gross_loss += 1.0  # Normalizado
                current_consec_losses += 1
                max_consec_losses = max(max_consec_losses, current_consec_losses)
            
            # Saltar a después del trade (simular no concurrencia)
            i = exit_idx + 1
        
        total_trades = wins + losses
        
        if total_trades < self.min_sample_size:
            return None
        
        win_rate = wins / total_trades if total_trades > 0 else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')
        
        return DurationReport(
            duration=duration,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            profit_factor=min(profit_factor, 10.0),  # Cap para evitar inf
            max_consecutive_losses=max_consec_losses,
            net_pnl=gross_profit - gross_loss,
            signals_generated=signals_generated,
            signals_passed_filter=signals_passed_filter
        )

    def _calculate_scores(self, reports: List[DurationReport]) -> None:
        """Calcula score ponderado para cada reporte."""
        if not reports:
            return
        
        # Normalizar métricas para scoring
        max_pf = max(r.profit_factor for r in reports) or 1
        max_trades = max(r.total_trades for r in reports) or 1
        
        for report in reports:
            # Win rate score (0-1, ideal >= min_win_rate)
            wr_score = min(report.win_rate / self.min_win_rate, 1.0)
            
            # Profit factor score (normalizado)
            pf_score = min(report.profit_factor / max_pf, 1.0) if max_pf > 0 else 0
            
            # Consecutive losses score (penalización)
            cl_score = max(0, 1 - (report.max_consecutive_losses / self.max_trades_per_round))
            
            # Sample size score (más trades = más confiable)
            ss_score = min(report.total_trades / max_trades, 1.0)
            
            # Filter efficiency score
            fe_score = (
                report.signals_passed_filter / report.signals_generated 
                if report.signals_generated > 0 else 0
            )
            
            # Score ponderado
            report.score = (
                self.weights['win_rate'] * wr_score +
                self.weights['profit_factor'] * pf_score +
                self.weights['consecutive_losses'] * cl_score +
                self.weights['sample_size'] * ss_score +
                self.weights['filter_efficiency'] * fe_score
            )
            
            # Penalización dura si no cumple criterios mínimos
            if report.win_rate < self.min_win_rate:
                report.score *= 0.5
            
            if report.max_consecutive_losses > self.max_trades_per_round:
                report.score *= 0.3

    def _apply_hysteresis(
        self, 
        best_report: DurationReport, 
        current_duration: int
    ) -> int:
        """
        Aplica histéresis para evitar cambios frecuentes de duración.
        Solo cambia si la nueva duración es significativamente mejor.
        """
        # Si no hay duración actual, usar la mejor
        if self.current_optimal_duration is None:
            return best_report.duration
        
        # Si la mejor es la misma que la actual, mantener
        if best_report.duration == current_duration:
            return current_duration
        
        # Buscar reporte de la duración actual para comparar scores
        current_report = next(
            (r for r in self.optimization_history[-1].get('reports_summary', []) 
             if r['duration'] == current_duration),
            None
        )
        
        # Si no hay datos de la duración actual, cambiar
        if current_report is None:
            return best_report.duration
        
        # Solo cambiar si la mejora supera el margen de histéresis
        improvement = best_report.score - current_report.get('score', 0)
        
        if improvement > self.hysteresis_margin:
            logger.info(
                f"📈 Cambio justificado: {current_duration} → {best_report.duration} "
                f"(mejora: {improvement:.3f})"
            )
            return best_report.duration
        else:
            logger.info(
                f"📊 Manteniendo duración {current_duration} "
                f"(mejora insuficiente: {improvement:.3f})"
            )
            return current_duration

    def get_recommendation_summary(self) -> Dict[str, Any]:
        """Retorna resumen de la última optimización para logging/análisis."""
        if not self.optimization_history:
            return {"status": "no_data"}
        
        last = self.optimization_history[-1]
        return {
            "timestamp": last['timestamp'],
            "current_duration": last['current_duration'],
            "recommended_duration": last['selected_duration'],
            "best_score": last['best_score'],
            "top_3_durations": sorted(
                last['reports_summary'], 
                key=lambda x: x['score'], 
                reverse=True
            )[:3]
        }