import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
        }
    },
    # "loggers": {
    #     "": {  # root logger
    #         "handlers": ["console"],
    #         "level": "DEBUG",
    #         "propagate": True,
    #     }
    # },
}

# Setup the logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

from .app import App
from .interface import Interface
from .interface_gui import Interface_GUI
from . import utils
from . import config
from .run_gui import exe