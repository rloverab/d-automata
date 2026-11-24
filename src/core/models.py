from typing import Dict, Literal, Any
from derivpy import Candle, Tick
from pydantic import BaseModel, ConfigDict, Field


class Signal(BaseModel):
    data: list[Candle|Tick] | None = None
    direction: Literal[
        "ACCU", "ASIAND", "ASIANU", 
        "CALL", "CALLE", "DIGITDIFF", 
        "DIGITEVEN", "DIGITMATCH", "DIGITODD", 
        "DIGITOVER", "DIGITUNDER", "EXPIRYMISS", 
        "EXPIRYMISSE", "EXPIRYRANGE", "EXPIRYRANGEE", 
        "HIGHER", "LOWER", "MULTDOWN", 
        "MULTUP", "NOTOUCH", "ONETOUCH", 
        "PUT", "PUTE", "RANGE", 
        "RESETCALL", "RESETPUT", "RUNHIGH", 
        "RUNLOW", "TICKHIGH", "TICKLOW", 
        "TURBOSLONG", "TURBOSSHORT", "UPORDOWN", 
        "VANILLALONGCALL", "VANILLALONGPUT"
    ]
    duration: int = Field(gt=0)
    duration_unit: Literal["d", "m", "s", "h", "t"] = "t"
    extras: dict[str, Any] = Field(default_factory=dict)

class TradeRecord(BaseModel):
    """Representa un trade completado con contexto financiero real (Pydantic v2)."""
    model_config = ConfigDict(frozen=True)

    entry_epoch: float
    exit_epoch: float
    direction: str
    entry_price: float
    exit_price: float
    duration_ticks: int
    result: Literal["win", "loss"]
    stake: float
    pnl: float
    balance_after: float
    round_id: int
    matrix_step: int  # 0 = base, 1+ = recuperación
    signal_generation_epoch: float
    metadata: Dict[str, Any] = Field(default_factory=dict)