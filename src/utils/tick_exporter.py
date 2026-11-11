import json
import csv
from typing import Iterable, Optional, Set

from derivpy import Tick

# Asumimos que tu clase Tick y FloatTolerante ya están definidas en el entorno.
# from tu_modulo import Tick 

class TickExporter:
    """
    Clase utilitaria para la serialización de datos de mercado (Ticks).
    Utiliza métodos estáticos para operaciones de I/O sin estado.
    """
    
    # Campos que la clase Tick excluye por defecto según su definición original
    _DEFAULT_EXCLUDED_FIELDS: Set[str] = {'bid', 'ask', 'subscription_id'}

    @staticmethod
    def _get_dump_data(tick: 'Tick', exclude_fields: Optional[Iterable[str]] = None) -> dict:
        """
        Helper para unificar la lógica de exclusión de Pydantic V2.
        Fusiona las exclusiones por defecto del modelo con las solicitadas por el usuario.
        """
        if exclude_fields:
            final_exclude = TickExporter._DEFAULT_EXCLUDED_FIELDS.union(set(exclude_fields))
            return tick.model_dump(exclude=final_exclude)
        # Si no se piden exclusiones extra, usamos el comportamiento por defecto de Pydantic
        return tick.model_dump()

    @staticmethod
    def to_jsonl(ticks: list['Tick'], filepath: str, exclude_fields: Optional[Iterable[str]] = None) -> None:
        """
        Exporta una lista de Ticks a un archivo JSONL (JSON Lines).
        
        :param ticks: Lista de objetos Tick.
        :param filepath: Ruta del archivo de destino.
        :param exclude_fields: Iterable con los nombres de los campos a omitir.
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            for tick in ticks:
                data = TickExporter._get_dump_data(tick, exclude_fields)
                # json.dumps es rápido, pero en producción con millones de ticks 
                # se recomienda usar `orjson.dumps` para mejorar el rendimiento.
                f.write(json.dumps(data) + '\n')

    @staticmethod
    def to_csv(ticks: list['Tick'], filepath: str, exclude_fields: Optional[Iterable[str]] = None) -> None:
        """
        Exporta una lista de Ticks a un archivo CSV.
        
        :param ticks: Lista de objetos Tick.
        :param filepath: Ruta del archivo de destino.
        :param exclude_fields: Iterable con los nombres de los campos a omitir.
        """
        if not ticks:
            return
            
        # Obtenemos los headers dinámicamente basándonos en el primer tick y las exclusiones
        first_tick_data = TickExporter._get_dump_data(ticks[0], exclude_fields)
        headers = list(first_tick_data.keys())
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for tick in ticks:
                row_data = TickExporter._get_dump_data(tick, exclude_fields)
                writer.writerow(row_data)