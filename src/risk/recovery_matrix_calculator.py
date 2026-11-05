import numpy as np
from typing import List, Literal, Dict, Union

TradeResult = Literal['win', 'loss', 'draw']

class RecoveryMatrixCalculator:
    """
    Calculadora pura de gestión de recuperación (Martingala Controlada).
    Ahora con mecanismos modulares de protección de ganancias (Early Exit, Trailing Stop, Viabilidad).
    """
    
    def __init__(
        self, 
        payout_pct: float, 
        max_trades: int, 
        target_wins: int, 
        min_bet: float = 0.35,
        # --- Solución A: Early Exit ---
        use_early_exit: bool = False,
        early_exit_threshold: float = 0.70,
        # --- Solución B: Trailing Stop de Ronda ---
        use_trailing_stop: bool = False,
        trailing_stop_pct: float = 0.50,
        # --- Solución C: Viabilidad Mínima del Trade ---
        use_min_viability: bool = False,
        min_viability_pct: float = 0.10
    ):
        self.payout = payout_pct / 100.0
        self.max_trades = max_trades
        self.target_wins = target_wins
        self.min_bet = min_bet
        
        # Parámetros de Protección (Opt-in)
        self.use_early_exit = use_early_exit
        self.early_exit_threshold = early_exit_threshold
        
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct
        
        self.use_min_viability = use_min_viability
        self.min_viability_pct = min_viability_pct
        
        self.matrix = self._build_matrix()

    def _build_matrix(self) -> np.ndarray:
        n = self.max_trades
        e = self.target_wins
        profit = self.payout
        
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

    def calculate_next_bet(self, history: List[TradeResult], current_round_capital: float) -> Dict[str, Union[float, str, int]]:
        wins = sum(1 for x in history if x == 'win')
        losses = sum(1 for x in history if x == 'loss')
        trades_done = wins + losses
        trades_remaining = self.max_trades - trades_done
        wins_needed = self.target_wins - wins
        
        if wins >= self.target_wins:
            return {"status": "TARGET_REACHED", "next_bet": 0.0, "wins": wins, "losses": losses, "trades_remaining": trades_remaining, "wins_needed": 0}
        
        if trades_done >= self.max_trades:
            return {"status": "MAX_TRADES_REACHED", "next_bet": 0.0, "wins": wins, "losses": losses, "trades_remaining": 0, "wins_needed": wins_needed}
        
        if wins_needed > trades_remaining:
            return {"status": "ROUND_LOST_IMPOSSIBLE", "next_bet": 0.0, "wins": wins, "losses": losses, "trades_remaining": trades_remaining, "wins_needed": wins_needed}
        
        if current_round_capital < self.min_bet:
            return {"status": "INSUFFICIENT_CAPITAL", "next_bet": 0.0, "wins": wins, "losses": losses, "trades_remaining": trades_remaining, "wins_needed": wins_needed}
        
        r = trades_done
        c = wins

        if r + 1 >= self.max_trades:
            next_bet = current_round_capital
        else:
            r1c0 = self.matrix[r + 1, c]
            r1c1 = self.matrix[r + 1, c + 1]
            
            if not np.isnan(r1c0) and not np.isnan(r1c1) and (r1c0 + self.payout * r1c1) != 0:
                a = 1 - ((self.payout + 1) * r1c1) / (r1c0 + self.payout * r1c1)
                next_bet = a * current_round_capital
            else:
                next_bet = current_round_capital
            
        if np.isnan(next_bet) or next_bet <= 0:
            next_bet = current_round_capital
        
        if next_bet < self.min_bet:
            next_bet = self.min_bet
        elif next_bet > current_round_capital:
            next_bet = current_round_capital
            
        # ✅ VALIDACIÓN 6: Solución C - Viabilidad Mínima del Trade
        if self.use_min_viability and current_round_capital > 0:
            viability_ratio = next_bet / current_round_capital
            if viability_ratio < self.min_viability_pct:
                return {
                    "status": "TRADE_NOT_VIABLE", 
                    "next_bet": 0.0, 
                    "wins": wins, 
                    "losses": losses,
                    "trades_remaining": trades_remaining, 
                    "wins_needed": wins_needed,
                    "viability_ratio": round(viability_ratio, 4)
                }
            
        return {
            "status": "ACTIVE",
            "next_bet": round(next_bet, 2),
            "wins": wins,
            "losses": losses,
            "trades_remaining": trades_remaining,
            "wins_needed": wins_needed
        }
    
    def should_close_round_early(self, current_pnl: float, target_profit: float) -> bool:
        """Solución A: Cierra si se alcanza un % del objetivo."""
        if not self.use_early_exit:
            return False        
        if target_profit <= 0:
            return False
        return (current_pnl / target_profit) >= self.early_exit_threshold

    def should_activate_trailing_stop(self, current_pnl: float, max_pnl: float) -> bool:
        """
        Solución B: Cierra si el PnL cae un % determinado desde su máximo histórico en la ronda.
        """    
        if not self.use_trailing_stop:
            return False
        if max_pnl <= 0:
            return False # No hay ganancias que proteger aún
        
        drawdown_from_max = (max_pnl - current_pnl) / max_pnl
        return drawdown_from_max >= self.trailing_stop_pct

    def should_close_round(
        self,
        current_pnl: float,
        target_profit: float,
        max_pnl: float
    ) -> dict[str, bool | str]:
        """
        Evalúa todas las condiciones de cierre temprano de la ronda.
        """

        if self.should_close_round_early(current_pnl, target_profit):
            return {
                "close": True,
                "reason": "EARLY_EXIT"
            }

        if self.should_activate_trailing_stop(current_pnl, max_pnl):
            return {
                "close": True,
                "reason": "TRAILING_STOP"
            }

        return {
            "close": False,
            "reason": "NONE"
        }

    def get_round_target_profit(self, initial_round_capital: float) -> float:
        """
        Devuelve la ganancia objetivo de la ronda completa.
        
        Se basa en matrix[0, 0], que representa el multiplicador de capital
        necesario desde el estado inicial (0 trades, 0 wins) para cumplir
        el objetivo de la ronda.
        
        Args:
            initial_round_capital: Capital inicial asignado a la ronda.
            
        Returns:
            Ganancia objetivo de la ronda completa.
        """
        if initial_round_capital <= 0:
            return 0.0
        
        if self.matrix.size == 0:
            return 0.0
        
        # matrix[0, 0] es el multiplicador de capital necesario desde el estado inicial
        initial_multiplier = self.matrix[0, 0]
        
        if np.isnan(initial_multiplier):
            return 0.0
        
        # El target profit es el incremento sobre el capital inicial
        target_profit = (initial_multiplier - 1.0) * initial_round_capital
        
        return max(0.0, float(target_profit))