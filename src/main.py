"""
CLI Entry Point - Trading Bot para Deriv
=========================================

Soporta tres modos de reconexión:
  - conservative (STOP_ON_FATAL): Se detiene ante fallos. Recomendado para desarrollo.
  - balanced (LIMITED_RETRY): Reintenta hasta N veces. Paper trading.
  - aggressive (INFINITE_RETRY): Nunca se detiene. Solo real money supervisado.
"""

import argparse
import asyncio
import logging
import signal
import sys
from typing import Literal

from derivpy import RestClient
from derivpy.reconnection import ReconnectionMode
from core.engine import TradingEngine
from utils.logger import setup_logger
from config.config import settings

logger = logging.getLogger(__name__)

# Mapeo de nombres amigables a Enum
MODE_MAPPING = {
    "conservative": ReconnectionMode.STOP_ON_FATAL,
    "balanced": ReconnectionMode.LIMITED_RETRY,
    "aggressive": ReconnectionMode.INFINITE_RETRY,
}


async def run_trading(
    symbol: str,
    mode_name: Literal["conservative", "balanced", "aggressive"],
    account_mode: Literal["real", "demo"] = "demo"
) -> None:
    """Función encargada de inicializar y arrancar el motor de trading."""
    logger.info(f"Preparando entorno de trading para el activo: {symbol}")
    logger.info(f"Modo de reconexión: {mode_name} ({MODE_MAPPING[mode_name].name})")
    logger.info(f"Tipo de cuenta: {account_mode}")
    
    # Advertencias de riesgo según el modo
    if mode_name == "aggressive" and account_mode == "real":
        logger.warning(
            "🚨 ADVERTENCIA: Estás ejecutando en modo AGRESSIVE con cuenta REAL. "
            "Asegúrate de tener sistema de alertas activo y kill switch disponible."
        )
    elif mode_name == "aggressive" and account_mode == "demo":
        logger.info(
            "ℹ️ Modo AGRESSIVE en cuenta DEMO: ideal para validar estabilidad "
            "del sistema de reconexión antes de pasar a real."
        )
    
    # Obtener cuenta del broker
    rest_client = RestClient(token=settings.DERIV_TOKEN, app_id=settings.DERIV_APP_ID)
    accounts = await rest_client.list_accounts()
    accounts_type = {account.account_type: account for account in accounts}
    account = accounts_type.get(account_mode)
    
    if not account:
        logger.error(f"❌ No se encontró cuenta de tipo '{account_mode}'")
        return
    
    # Convertir nombre de modo a Enum
    reconnection_mode = MODE_MAPPING[mode_name]
    
    # Instanciar motor con TODOS los parámetros requeridos
    engine = TradingEngine(
        account_id=account.account_id,
        symbol=symbol,
        mode=reconnection_mode  # 🆕 Parámetro crítico
    )
    
    # Configurar shutdown graceful vía señales del sistema
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    
    def _signal_handler():
        logger.info("🛑 Señal de apagado recibida. Deteniendo motor...")
        shutdown_event.set()
        # Crear tarea para detener el engine limpiamente
        asyncio.create_task(engine.stop())
    
    # Registrar handlers para SIGINT (Ctrl+C) y SIGTERM
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:
            # Windows no soporta add_signal_handler
            pass
    
    # Ejecutar motor con protección contra cancelación
    try:
        await engine.start()
    except asyncio.CancelledError:
        logger.info("⏹️ Motor cancelado por el event loop")
    except KeyboardInterrupt:
        logger.info("⏹️ Interrupción manual (Ctrl+C)")
        await engine.stop()
    except Exception as e:
        logger.error(f"❌ Error fatal en motor: {e}", exc_info=True)
        await engine.stop()
        raise


def parse_arguments() -> argparse.Namespace:
    """Configuración y parseo de argumentos CLI."""
    parser = argparse.ArgumentParser(
        description="CLI Trading Bot para Deriv - Automatización y análisis de contratos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Modo seguro para desarrollo (se detiene si falla)
  python main.py -t -s R_100 -m-mode conservative
  
  # Modo equilibrado para paper trading
  python main.py -t -s 1HZ100V -m-mode balanced --account demo
  
  # Modo agresivo 24/7 (requiere supervisión)
  python main.py -t -s R_100 -m-mode aggressive --account real
  
Modos de reconexión:
  conservative  Se detiene inmediatamente ante fallos (STOP_ON_FATAL)
  balanced      Reintenta hasta 5 veces antes de parar (LIMITED_RETRY)
  aggressive    Nunca se detiene, reintentos infinitos (INFINITE_RETRY)
        """
    )

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
        "--reconnection-mode",
        type=str,
        choices=list(MODE_MAPPING.keys()),
        default="balanced",
        help="Estrategia de reconexión ante fallos (por defecto: balanced)."
    )
    parser.add_argument(
        "--account",
        type=str,
        choices=["real", "demo"],
        default="demo",
        help="Tipo de cuenta a usar (por defecto: demo)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Activa logging detallado (DEBUG)."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        metavar="TIPO",
        help="Muestra una tabla de estadísticas en consola (ej. --metrics winrate)."
    )

    return parser.parse_args()


def main():
    # 1. Parsear argumentos PRIMERO (para saber si verbose está activo)
    args = parse_arguments()
    
    # 2. Configurar logging con nivel adecuado
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(log_level=log_level)

    # 3. Enrutamiento según los argumentos recibidos
    if args.trade:
        try:
            # Validación previa de seguridad
            if args.reconnection_mode == "aggressive" and args.account == "real":
                print("\n" + "="*70)
                print("⚠️  ADVERTENCIA DE SEGURIDAD")
                print("="*70)
                print("Vas a ejecutar en modo AGRESSIVE con cuenta REAL.")
                print("Este modo NUNCA se detiene automáticamente ante fallos.")
                print("\nAsegúrate de:")
                print("  ✓ Tener sistema de alertas configurado (Telegram/email)")
                print("  ✓ Haber probado previamente en modo 'balanced' con demo")
                print("  ✓ Disponer de kill switch remoto")
                print("  ✓ Monitorear activamente durante las primeras horas")
                print("="*70)
                
                confirm = input("\n¿Confirmas que deseas continuar? (escribe 'SI' para confirmar): ")
                if confirm != "SI":
                    logger.info("Operación cancelada por el usuario.")
                    sys.exit(0)
                print()
            
            asyncio.run(run_trading(
                symbol=args.symbol,
                mode_name=args.reconnection_mode,
                account_mode=args.account
            ))
            
        except KeyboardInterrupt:
            logger.info("Aplicación detenida manualmente por el usuario.")
            sys.exit(0)
        except Exception as e:
            logger.critical(f"Error no manejado: {e}", exc_info=True)
            sys.exit(1)
            
    elif args.metrics:
        # TODO: Implementar módulo de métricas
        logger.warning("Módulo de métricas aún no implementado")
        
    else:
        # Si no pasan argumentos válidos, muestra la ayuda automáticamente
        parse_arguments.__wrapped__ = argparse.ArgumentParser
        # Truco para reusar el parser: simplemente lo ejecutamos de nuevo
        # con --help para mostrar la ayuda
        import sys
        sys.argv.append("--help")
        parse_arguments()


if __name__ == "__main__":
    main()