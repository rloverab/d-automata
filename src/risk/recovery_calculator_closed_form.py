"""
Calculadora de Recuperación usando Matriz Precomputada
Arquitectura: Clase estática con caché de matrices
"""

import numpy as np
from typing import List, Literal, Dict, Union
from functools import lru_cache

TradeResult = Literal['win', 'loss', 'draw']


class RecoveryCalculatorClosedForm:
    """
    Calculadora de gestión de recuperación usando matriz precomputada.
    Todos los métodos son estáticos. Las matrices se cachean para eficiencia.
    """

    @staticmethod
    @lru_cache(maxsize=128)
    def _build_matrix(payout_pct: float, max_trades: int, target_wins: int) -> np.ndarray:
        """Construye y cachea la matriz de recuperación."""
        n = max_trades
        e = target_wins
        profit = payout_pct / 100.0

        matrix = np.full((n, e + 1), np.nan)

        for r in range(n - 1, -1, -1):
            for c in range(e + 1):
                if c >= e:
                    matrix[r, c] = 1.0
                elif (e - c) == (n - r):
                    matrix[r, c] = (profit + 1) ** (n - r)
                elif r < n - 1:
                    rc0 = matrix[r + 1, c]
                    rc1 = matrix[r + 1, c + 1]

                    if not np.isnan(rc0) and not np.isnan(rc1) and (rc0 + profit * rc1) != 0:
                        matrix[r, c] = (profit + 1) * rc0 * rc1 / (rc0 + profit * rc1)

        return matrix

    @staticmethod
    def calculate_next_bet(
        history: List[TradeResult],
        current_round_capital: float,
        payout_pct: float,
        max_trades: int,
        target_wins: int,
        min_bet: float = 0.35
    ) -> Dict[str, Union[float, str, int]]:
        """Calcula el próximo stake usando matriz precomputada."""
        wins = sum(1 for x in history if x == 'win')
        losses = sum(1 for x in history if x == 'loss')
        trades_done = wins + losses
        trades_remaining = max_trades - trades_done
        wins_needed = target_wins - wins

        # Validaciones de estado
        if wins >= target_wins:
            return {
                "status": "TARGET_REACHED",
                "next_bet": 0.0,
                "wins": wins,
                "losses": losses,
                "trades_remaining": trades_remaining,
                "wins_needed": 0
            }

        if trades_done >= max_trades:
            return {
                "status": "MAX_TRADES_REACHED",
                "next_bet": 0.0,
                "wins": wins,
                "losses": losses,
                "trades_remaining": 0,
                "wins_needed": wins_needed
            }

        if wins_needed > trades_remaining:
            return {
                "status": "ROUND_LOST_IMPOSSIBLE",
                "next_bet": 0.0,
                "wins": wins,
                "losses": losses,
                "trades_remaining": trades_remaining,
                "wins_needed": wins_needed
            }

        if current_round_capital < min_bet:
            return {
                "status": "INSUFFICIENT_CAPITAL",
                "next_bet": 0.0,
                "wins": wins,
                "losses": losses,
                "trades_remaining": trades_remaining,
                "wins_needed": wins_needed
            }

        # Obtener matriz cacheada
        matrix = RecoveryCalculatorClosedForm._build_matrix(payout_pct, max_trades, target_wins)
        payout_decimal = payout_pct / 100.0

        r = trades_done
        c = wins

        if r + 1 >= max_trades:
            next_bet = current_round_capital
        else:
            r1c0 = matrix[r + 1, c]
            r1c1 = matrix[r + 1, c + 1]

            if not np.isnan(r1c0) and not np.isnan(r1c1) and (r1c0 + payout_decimal * r1c1) != 0:
                a = 1 - ((payout_decimal + 1) * r1c1) / (r1c0 + payout_decimal * r1c1)
                next_bet = a * current_round_capital
            else:
                next_bet = current_round_capital

        # Validaciones de límites
        if np.isnan(next_bet) or next_bet <= 0:
            next_bet = current_round_capital

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
        """Calcula el profit objetivo basado en matrix[0,0]."""
        if initial_round_capital <= 0:
            return 0.0

        matrix = RecoveryCalculatorClosedForm._build_matrix(payout_pct, max_trades, target_wins)
        initial_multiplier = matrix[0, 0]

        if np.isnan(initial_multiplier):
            return 0.0

        target_profit = (initial_multiplier - 1.0) * initial_round_capital
        return max(0.0, float(target_profit))

    @staticmethod
    def should_close_round_early(
        current_pnl: float,
        target_profit: float,
        early_exit_threshold: float = 0.70
    ) -> bool:
        """Cierra si se alcanza un % del objetivo."""
        if target_profit <= 0:
            return False
        return (current_pnl / target_profit) >= early_exit_threshold

    @staticmethod
    def should_activate_trailing_stop(
        current_pnl: float,
        max_pnl: float,
        trailing_stop_pct: float = 0.50
    ) -> bool:
        """Cierra si el PnL cae un % desde su máximo."""
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
        """Evalúa todas las condiciones de cierre temprano."""
        if use_early_exit and RecoveryCalculatorClosedForm.should_close_round_early(
            current_pnl, target_profit, early_exit_threshold
        ):
            return {"close": True, "reason": "EARLY_EXIT"}

        if use_trailing_stop and RecoveryCalculatorClosedForm.should_activate_trailing_stop(
            current_pnl, max_pnl, trailing_stop_pct
        ):
            return {"close": True, "reason": "TRAILING_STOP"}

        return {"close": False, "reason": "NONE"}