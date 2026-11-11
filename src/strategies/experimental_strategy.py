
from core.models import Signal
from strategies.base import Strategy


class ExperimentalStrategy(Strategy):
    def get_signal(self, data, duration = 10, duration_unit = "t", **kwargs):
        n = 7        
        full_ticks = list(data)
        ticks = full_ticks[-n:]
        len_ticks = len(ticks)
        directions: list[int] = []

        if len(ticks) < n:
            return None

        for i in range(1, len_ticks):
            if ticks[i-1].quote < ticks[i].quote:
                directions.append(1)
            elif ticks[i-1].quote > ticks[i].quote:
                directions.append(-1)
            else:
                directions.append(0)

        trend = abs(ticks[-4].quote - ticks[-2].quote) > abs(ticks[-2].quote - ticks[-1].quote)
        
        if all([d == 1 for d in directions[:-1]]) and directions[-1] == -1: # CALL
            direction = "CALL" if trend else "PUT"
            # print(direction, trend)
            return Signal(data=full_ticks, direction=direction, duration=duration, duration_unit=duration_unit)
        if all([d == -1 for d in directions[:-1]]) and directions[-1] == 1: # PUT
            direction = "PUT" if trend else "CALL"
            # print(direction, trend)
            return Signal(data=full_ticks, direction=direction, duration=duration, duration_unit=duration_unit)

        return None

        
