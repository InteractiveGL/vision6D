import os
import pathlib
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

CWD = pathlib.Path(os.path.abspath(__file__)).parent
STYLES_FILE = CWD / 'data' /'style.qss'

from .mainwindow import MyMainWindow
from . import tools
from . import widgets
from . import components
from . import containers

all = [
    'MyMainWindow',
    'Interface',
    'tools',
    'widgets',
    'components',
    'containers'
]
