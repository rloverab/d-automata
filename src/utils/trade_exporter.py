import csv
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Set

from pydantic import BaseModel, ConfigDict, Field

from core.models import TradeRecord


# class TradeRecord(BaseModel):
#     """Representa un trade completado con contexto financiero real (Pydantic v2)."""
#     model_config = ConfigDict(frozen=True)

#     entry_epoch: float
#     exit_epoch: float
#     direction: str
#     entry_price: float
#     exit_price: float
#     duration_ticks: int
#     result: str
#     stake: float
#     pnl: float
#     balance_after: float
#     round_id: int
#     matrix_step: int
#     signal_generation_epoch: float
#     metadata: dict[str, object] = Field(default_factory=dict)


class TradeExporter:
    """
    Clase utilitaria para la serialización de TradeRecord.
    Utiliza métodos estáticos para operaciones de I/O sin estado.
    """

    _DEFAULT_EXCLUDED_FIELDS: Set[str] = set()

    @staticmethod
    def _prepare_filepath(filepath: str | Path) -> Path:
        """Convierte a Path y crea los directorios padre si no existen."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _get_dump_data(
        trade: TradeRecord,
        exclude_fields: Optional[Iterable[str]] = None,
        flatten_metadata: bool = False,
    ) -> dict[str, object]:
        """Serializa un TradeRecord aplicando exclusiones y aplanado opcional de metadata."""
        final_exclude = (
            TradeExporter._DEFAULT_EXCLUDED_FIELDS.union(set(exclude_fields))
            if exclude_fields
            else TradeExporter._DEFAULT_EXCLUDED_FIELDS
        )
        data = trade.model_dump(exclude=final_exclude)

        if flatten_metadata and "metadata" in data and isinstance(data["metadata"], dict):
            meta: dict[str, object] = data.pop("metadata")
            for key, value in meta.items():
                data[f"meta_{key}"] = value

        return data

    @staticmethod
    def to_jsonl(
        trades: Iterable[TradeRecord],
        filepath: str | Path,
        exclude_fields: Optional[Iterable[str]] = None,
    ) -> None:
        """Exporta una colección de TradeRecord a formato JSONL."""
        target = TradeExporter._prepare_filepath(filepath)
        with open(target, "w", encoding="utf-8") as f:
            for trade in trades:
                data = TradeExporter._get_dump_data(trade, exclude_fields=exclude_fields)
                f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def to_csv(
        trades: Sequence[TradeRecord],
        filepath: str | Path,
        exclude_fields: Optional[Iterable[str]] = None,
        flatten_metadata: bool = True,
    ) -> None:
        """Exporta una secuencia de TradeRecord a formato CSV."""
        if not trades:
            return

        target = TradeExporter._prepare_filepath(filepath)
        rows = [
            TradeExporter._get_dump_data(
                trade, exclude_fields=exclude_fields, flatten_metadata=flatten_metadata
            )
            for trade in trades
        ]

        headers: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    headers.append(key)

        with open(target, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)