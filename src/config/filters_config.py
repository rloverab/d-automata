import yaml
from pydantic import BaseModel, ValidationError

class VolatilityParams(BaseModel):
    min_volatility: float
    lookback: int

class FiltersConfig(BaseModel):
    version: str
    volatility: VolatilityParams

def load_filters_config(path: str = "filters_config.yaml") -> FiltersConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)
    try:
        return FiltersConfig.model_validate(raw_data)
    except ValidationError as e:
        # Si alguien rompe el YAML, el bot NO arranca. Esto previene desastres.
        raise RuntimeError(f"Error crítico de Gobernanza en {path}: {e}")