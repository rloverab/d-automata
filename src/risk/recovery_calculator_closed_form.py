import math
from typing import List, Literal, Dict, Union

TradeResult = Literal['win', 'loss', 'draw']

class RecoveryCalculatorClosedForm:
    """
    Calculadora stateless de recuperación mediante fórmula cerrada.
    Todos los métodos son estáticos. Sin estado mutable, thread-safe.
    """
    
    @staticmethod
    def _binomial_cdf_tail(k: int, j: int, q: float) -> float:
        """
        Calcula P(X >= j) donde X ~ Binomial(k, q).
        """
        if j <= 0:
            return 1.0
        if j > k:
            return 0.0
        
        prob_less_than_j = 0.0
        for i in range(j):
            coef = math.comb(k, i)
            prob_i = coef * (q ** i) * ((1 - q) ** (k - i))
            prob_less_than_j += prob_i
        
        return 1.0 - prob_less_than_j
    
    @staticmethod
    def _get_multiplier(trades_remaining: int, wins_needed: int, payout: float) -> float:
        """
        Obtiene M[r,c] directamente mediante la fórmula cerrada.
        """
        if wins_needed <= 0:
            return 1.0
        
        if wins_needed > trades_remaining:
            return float('inf')
        
        q = 1.0 / (1.0 + payout)
        prob_success = RecoveryCalculatorClosedForm._binomial_cdf_tail(trades_remaining, wins_needed, q)
        
        if prob_success <= 0:
            return float('inf')
        
        return 1.0 / prob_success
    
    @staticmethod
    def calculate_next_bet(
        history: List[TradeResult],
        current_round_capital: float,
        payout_pct: float,
        max_trades: int,
        target_wins: int,
        min_bet: float = 0.35
    ) -> Dict[str, Union[float, str, int]]:
        """
        Calcula el monto a apostar basado en el historial de resultados.
        
        :param history: Lista de resultados previos en la ronda
        :param current_round_capital: Capital restante asignado a esta ronda
        :param payout_pct: Porcentaje de ganancia del broker (ej. 92.0 para 92%)
        :param max_trades: Número máximo de intentos permitidos
        :param target_wins: Número de victorias necesarias
        :param min_bet: Monto mínimo permitido por el broker
        :return: Diccionario con el monto a apostar y estado de la ronda
        """
        payout = payout_pct / 100.0
        wins = sum(1 for x in history if x == 'win')
        losses = sum(1 for x in history if x == 'loss')
        trades_done = wins + losses
        trades_remaining = max_trades - trades_done
        wins_needed = target_wins - wins
        
        # Validaciones de estado
        if wins >= target_wins:
            return {"status": "TARGET_REACHED", "next_bet": 0.0, "wins": wins, "losses": losses,
                    "trades_remaining": trades_remaining, "wins_needed": 0}
        
        if trades_done >= max_trades:
            return {"status": "MAX_TRADES_REACHED", "next_bet": 0.0, "wins": wins, "losses": losses,
                    "trades_remaining": 0, "wins_needed": wins_needed}
        
        if wins_needed > trades_remaining:
            return {"status": "ROUND_LOST_IMPOSSIBLE", "next_bet": 0.0, "wins": wins, "losses": losses,
                    "trades_remaining": trades_remaining, "wins_needed": wins_needed}
        
        if current_round_capital < min_bet:
            return {"status": "INSUFFICIENT_CAPITAL", "next_bet": 0.0, "wins": wins, "losses": losses,
                    "trades_remaining": trades_remaining, "wins_needed": wins_needed}
        
        # Calcular el bet usando la fórmula cerrada
        k_next = trades_remaining - 1
        j_lose = wins_needed
        j_win = wins_needed - 1
        
        M_if_lose = RecoveryCalculatorClosedForm._get_multiplier(k_next, j_lose, payout)
        M_if_win = RecoveryCalculatorClosedForm._get_multiplier(k_next, j_win, payout)
        
        denominator = M_if_lose + payout * M_if_win
        
        if denominator == 0 or math.isinf(M_if_lose) or math.isinf(M_if_win):
            next_bet = current_round_capital
        else:
            a = 1.0 - ((1.0 + payout) * M_if_win) / denominator
            next_bet = a * current_round_capital
        
        # Validaciones finales
        if math.isnan(next_bet) or next_bet <= 0:
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