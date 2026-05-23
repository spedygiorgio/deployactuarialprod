import logging
import time
import functools
from typing import Callable, Any


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Crea e configura un logger con formattazione standardizzata.
    
    Args:
        name: Nome del logger (tipicamente __name__)
        level: Livello di logging (default: INFO)
    
    Returns:
        Logger configurato
    
    Example:
        logger = get_logger(__name__)
        logger.info("Messaggio di info")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evita di aggiungere handler duplicati
    if not logger.handlers:
        # Handler per console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Formattazione
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger


def timer(func: Callable) -> Callable:
    """
    Decoratore per misurare il tempo di esecuzione di una funzione.
    
    Stampa il tempo di esecuzione in console dopo l'esecuzione della funzione.
    
    Args:
        func: Funzione da decorare
    
    Returns:
        Funzione decorata
    
    Example:
        @timer
        def my_function():
            time.sleep(1)
            return "Done"
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger(__name__)
        start_time = time.time()
        
        logger.info(f"⏱️  Inizio esecuzione: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            logger.info(
                f"✅ Fine esecuzione: {func.__name__} - "
                f"Tempo impiegato: {elapsed_time:.2f}s"
            )
    
    return wrapper
