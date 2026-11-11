import asyncio
from datetime import datetime

from derivpy import MarketDataClient, Tick
from utils.tick_exporter import TickExporter

SYMBOL="1HZ100V"
async def test():
    mdc = MarketDataClient()
    await mdc.connect()
    end: str | int = "latest"
    ticks: list[Tick] = []

    for i in range(10):
        _ticks, _ = await mdc.ticks_history(
            symbol=SYMBOL,
            end=end
        )
        ticks += _ticks
        end = ticks[0].epoch - 1
        await asyncio.sleep(1.0)
    timestamp = int(datetime.now().timestamp())
    TickExporter.to_jsonl(ticks=ticks, filepath=f"ticks-{timestamp}.json")
    TickExporter.to_csv(ticks=ticks, filepath=f"ticks-{timestamp}.csv")
    await mdc.disconnect()

asyncio.run(test())