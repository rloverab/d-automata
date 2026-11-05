from typing import Literal, Any
from derivpy import Candle, Tick
from pydantic import BaseModel, Field


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