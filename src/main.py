import argparse
import asyncio
import logging
from typing import Literal

from derivpy import RestClient
from core.engine import TradingEngine
from utils.logger import setup_logger
from config import settings

logger = logging.getLogger(__name__)

async def run_trading(symbol: str, mode: Literal["real", "demo"] = "demo") -> None:
    """Función encargada de inicializar y arrancar el motor de trading."""
    logger.info(f"Preparando entorno de trading para el activo: {symbol}")    
    
    # Ejemplo de inicialización:
    # db_manager = CouchDBManager(...)
    # contract_store = ContractStore(db_manager)
    # client = MarketDataClient(...)
    # engine = TradingEngine(client, contract_store, symbol)
    # await engine.start()
    rest_client = RestClient(token=settings.DERIV_TOKEN, app_id=settings.DERIV_APP_ID)
    accounts = await rest_client.list_accounts()
    accounts_type = {account.account_type: account for account in accounts}
    account = accounts_type.get(mode, {})
    if account:
        engine = TradingEngine(account_id=account.account_id, symbol=symbol)
        await engine.start()
    
    print(f"[Simulación] El bot estaría operando en {symbol} ahora mismo...")

def main():
    # 1. Configurar el sistema de logs global al arrancar la CLI
    setup_logger(log_level=logging.INFO)

    # 2. Configuración de argparse para los comandos y parámetros
    parser = argparse.ArgumentParser(
        description="CLI Trading Bot para Deriv - Automatización y análisis de contratos."
    )

    # Subcomandos o argumentos principales
    parser.add_argument(
        "-t", "--trade", 
        action="store_true", 
        help="Inicia el bucle de trading en vivo con el broker."
    )
    parser.add_argument(
        "-s", "--symbol", 
        type=str, 
        default="R_100", 
        help="Símbolo del activo a operar (por defecto: R_100)."
    )
    parser.add_argument(
        "-m", "--metrics", 
        type=str, 
        metavar="TIPO",
        help="Muestra una tabla de estadísticas en consola (ej. -m winrate)."
    )

    args = parser.parse_args()

    # 3. Enrutamiento según los argumentos recibidos
    if args.trade:
        try:
            # Como el motor usa WebSockets y async, usamos asyncio.run()
            asyncio.run(run_trading(args.symbol))
        except KeyboardInterrupt:
            logger.info("Aplicación detenida manualmente por el usuario desde la consola.")
    elif args.metrics:
        # run_metrics(args.metrics)
        pass
    else:
        # Si no pasan argumentos válidos, muestra la ayuda automáticamente
        parser.print_help()


if __name__ == "__main__":
    main()