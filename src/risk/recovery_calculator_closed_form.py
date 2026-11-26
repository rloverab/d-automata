"""
Calculadora de Recuperación en Forma Cerrada (Closed-Form)
Arquitectura: Clase estática (sin estado) para máxima eficiencia
"""

import math
from typing import List, Literal, Dict, Union

TradeResult = Literal['win', 'loss', 'draw']


class RecoveryCalculatorClosedForm:
    """
    Calculadora de gestión de recuperación usando fórmulas cerradas.
    Todos los métodos son estáticos para evitar overhead de instanciación.
    """

    @staticmethod
    def calculate_next_bet(
        history: List[TradeResult],
        current_round_capital: float,
        payout_pct: float,
        max_trades: int,
        target_wins: int,
        min_bet: float = 0.35,
        initial_round_capital: float | None = None  # 🆕 Necesario
    ) -> Dict[str, Union[float, str, int]]:
        """
        Calcula el próximo stake usando Martingala de Ganancia Fija.
        Garantiza PnL constante sin importar el orden de resultados.
        """
        wins = sum(1 for x in history if x == 'win')
        losses = sum(1 for x in history if x == 'loss')
        trades_done = wins + losses
        trades_remaining = max_trades - trades_done
        wins_needed = target_wins - wins

        # Validaciones de estado
        if wins >= target_wins:
            return {"status": "TARGET_REACHED", "next_bet": 0.0, "wins": wins, "losses": losses, "trades_remaining": trades_remaining, "wins_needed": 0}
        if trades_done >= max_trades:
            return {"status": "MAX_TRADES_REACHED", "next_bet": 0.0, "wins": wins, "losses": losses, "trades_remaining": 0, "wins_needed": wins_needed}
        if wins_needed > trades_remaining:
            return {"status": "ROUND_LOST_IMPOSSIBLE", "next_bet": 0.0, "wins": wins, "losses": losses, "trades_remaining": trades_remaining, "wins_needed": wins_needed}
        if current_round_capital < min_bet:
            return {"status": "INSUFFICIENT_CAPITAL", "next_bet": 0.0, "wins": wins, "losses": losses, "trades_remaining": trades_remaining, "wins_needed": wins_needed}

        payout_decimal = payout_pct / 100.0

        # 🆕 Calcular target profit basado en el capital inicial de la ronda
        if initial_round_capital is None:
            initial_round_capital = current_round_capital  # Fallback
        
        # Fórmula de Martingala de Ganancia Fija
        # target_profit = initial_round_capital * ((1 + payout)^(1/wins_needed) - 1)
        # Simplificado para wins_needed=1:
        if wins_needed == 1:
            target_profit = initial_round_capital * payout_decimal / (1 + payout_decimal)
        else:
            # Para múltiples wins, usar raíz n-ésima
            target_profit = initial_round_capital * ((1 + payout_decimal) ** (1/wins_needed) - 1)

        # 🆕 Calcular pérdidas acumuladas en esta ronda
        accumulated_losses = initial_round_capital - current_round_capital

        # 🆕 Fórmula de recuperación: next_bet = (target_profit + accumulated_losses) / payout_decimal
        if trades_remaining == 1:
            next_bet = current_round_capital  # All-in en último intento
        else:
            next_bet = (target_profit + accumulated_losses) / payout_decimal

        # Validaciones de límites
        if next_bet < min_bet:
            next_bet = min_bet
        elif next_bet > current_round_capital:
            next_bet = current_round_capital

        return {
            "status": "ACTIVE",
            "next_bet": round(next_bet, 2),
            "wins": wins,
            "losses": losses,
            "trades_remaining": trades_remaining,
            "wins_needed": wins_needed
        }

    @staticmethod
    def get_round_target_profit(
        initial_round_capital: float,
        payout_pct: float,
        max_trades: int,
        target_wins: int
    ) -> float:
        """
        Calcula el profit objetivo neto de la ronda completa.
        Ahora usa la distribución exacta de capital para recuperación.
        """
        if initial_round_capital <= 0:
            return 0.0

        # Simulamos el primer trade para obtener la apuesta inicial
        first_trade_info = RecoveryCalculatorClosedForm.calculate_next_bet(
            history=[],
            current_round_capital=initial_round_capital,
            payout_pct=payout_pct,
            max_trades=max_trades,
            target_wins=target_wins
        )

        first_bet = first_trade_info["next_bet"]
        payout_decimal = payout_pct / 100.0
        
        # La ganancia neta es exactamente la apuesta inicial * el payout
        target_profit = first_bet * payout_decimal

        return round(target_profit, 2)

    @staticmethod
    def should_close_round_early(
        current_pnl: float,
        target_profit: float,
        early_exit_threshold: float = 0.70
    ) -> bool:
        """
        Solución A: Cierra si se alcanza un % del objetivo.
        
        Args:
            current_pnl: PnL actual de la ronda
            target_profit: Profit objetivo de la ronda
            early_exit_threshold: Umbral para cierre temprano (0.70 = 70%)
        """
        if target_profit <= 0:
            return False
        return (current_pnl / target_profit) >= early_exit_threshold

    @staticmethod
    def should_activate_trailing_stop(
        current_pnl: float,
        max_pnl: float,
        trailing_stop_pct: float = 0.50
    ) -> bool:
        """
        Solución B: Cierra si el PnL cae un % desde su máximo histórico.
        
        Args:
            current_pnl: PnL actual de la ronda
            max_pnl: Máximo PnL alcanzado en la ronda
            trailing_stop_pct: Porcentaje de caída para activar (0.50 = 50%)
        """
        if max_pnl <= 0:
            return False
        
        drawdown_from_max = (max_pnl - current_pnl) / max_pnl
        return drawdown_from_max >= trailing_stop_pct

    @staticmethod
    def should_close_round(
        current_pnl: float,
        target_profit: float,
        max_pnl: float,
        use_early_exit: bool = False,
        early_exit_threshold: float = 0.70,
        use_trailing_stop: bool = False,
        trailing_stop_pct: float = 0.50
    ) -> Dict[str, Union[bool, str]]:
        """
        Evalúa todas las condiciones de cierre temprano de la ronda.
        """
        if use_early_exit and RecoveryCalculatorClosedForm.should_close_round_early(
            current_pnl, target_profit, early_exit_threshold
        ):
            return {"close": True, "reason": "EARLY_EXIT"}

        if use_trailing_stop and RecoveryCalculatorClosedForm.should_activate_trailing_stop(
            current_pnl, max_pnl, trailing_stop_pct
        ):
            return {"close": True, "reason": "TRAILING_STOP"}

        return {"close": False, "reason": "NONE"}