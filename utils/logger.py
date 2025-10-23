import logging
import os
from datetime import datetime


class CarrotLogger:
    """Centralized logging utility for CARROT project."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CarrotLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.setup_logging()
            CarrotLogger._initialized = True
    
    def setup_logging(self, 
                     log_level: str = "INFO",
                     log_dir: str = "logs",
                     log_to_file: bool = True,
                     log_to_console: bool = True):
        """Setup logging configuration for CARROT project."""
        
        # Create logs directory if it doesn't exist
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)
        
        # Add file handler
        if log_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"carrot_{timestamp}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance for a specific module."""
        return logging.getLogger(name)
    
    @staticmethod
    def set_level(level: str):
        """Set logging level globally."""
        logging.getLogger().setLevel(getattr(logging, level.upper()))


def get_carrot_logger(name: str) -> logging.Logger:
    """Convenience function to get a CARROT logger."""
    CarrotLogger()  # Ensure logging is initialized
    return logging.getLogger(name)