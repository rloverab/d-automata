import yaml
from pydantic import BaseModel, ValidationError

class RiseFallMontecarloParams(BaseModel):
    window_size: int
    simulations: int
    threshold: float
    min_data_points: int

class MarkovTickParams(BaseModel):
    estimated_window: int
    intervalo_reestimacion: int
    umbral_prob: float
    ventana_caracteristicas: int

class StrategiesConfig(BaseModel):
    version: str
    rise_fall_montecarlo: RiseFallMontecarloParams
    markov_tick: MarkovTickParams

def load_strategies_config(path: str = "strategies_config.yaml") -> StrategiesConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)
    try:
        return StrategiesConfig.model_validate(raw_data)
    except ValidationError as e:
        # Si alguien rompe el YAML, el bot NO arranca. Esto previene desastres.
        raise RuntimeError(f"Error crítico de Gobernanza en {path}: {e}")