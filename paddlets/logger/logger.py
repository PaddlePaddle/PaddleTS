#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
logger file
"""

import logging
import inspect
import functools
from enum import Enum

LOG_LEVEL = logging.INFO
DEFAULT_LOGGER = "paddlets"


def log_decorator(f):
    """Add logging for method"""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        """wrapper"""
        module_name = inspect.getmodule(f).__name__
        func_name = f.__qualname__
        logger = Logger(module_name)
        logger.debug("function:%s" % func_name)
        result = f(*args, **kwargs)
        return result

    return wrapper


class Logger(object):
    """
    log class

    Args:
        name(str, optional): module name

    Attributes:
        logger(Logger) : a logger with the specified name which created  by python logging
        lvl(Enum): log level

    """
    level = Enum('level',
                 {'debug': logging.DEBUG,
                  'info': logging.INFO,
                  'warning': logging.WARNING,
                  'error': logging.ERROR,
                  'critical': logging.CRITICAL
                  }
                 )
    logger = None
    lvl = None

    def __init__(self, name=DEFAULT_LOGGER):
        FORMAT = '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s'
        # Do basic configuration(level and format) for the logging system
        logging.basicConfig(level=LOG_LEVEL, format=FORMAT)
        self.logger = logging.getLogger(name)

    def __getattr__(self, name):
        """
        Args:
            name(str): log level name

        Returns
            Logger: returned logger

        Raises:
            AttributeError: An error occurred whern the Attr not Correct

        """
        if name in ('debug', 'info', 'warning', 'error', 'critical'):
            self.lvl = self.level[name].value
            return self
        else:
            raise AttributeError('Attr not Correct')

    def __call__(self, msg):
        """
        Args:
            msg(str):  message to be printed

        Returns
            None

        """
        self.logger.log(self.lvl, msg)


def raise_log(exception: Exception, logger: Logger = Logger(DEFAULT_LOGGER)):
    """
    Can be used to replace "raise" when throwing an exception to ensure the logging
    of the exception. After logging it, the exception is raised.

    Args:
        exception: The exception instance to be raised.
        logger: The logger instance to log the exception type and message.

    Returns:
        None

    Raises
        exception

    """
    exception_type = str(type(exception)).split("'")[1]
    message = str(exception)
    logger.error(exception_type + ": " + message)

    raise exception


def raise_if_not(
        condition: bool,
        message: str = "",
        logger: Logger = Logger(DEFAULT_LOGGER),
):
    """
    Args:
        condition(bool): The boolean condition to be checked.
        message(str): The message of the ValueError.
        logger(Logger): The logger instance to log the error message if 'condition' is False.

    Returns
        None

    Raises:
        ValueError

    """
    if not condition:
        logger.error("ValueError: " + message)
        raise ValueError(message)


def raise_if(
        condition: bool,
        message: str = "",
        logger: Logger = Logger(DEFAULT_LOGGER),
):
    """
    Args:
        condition(bool): The boolean condition to be checked.
        message(str): The message of the ValueError.
        logger(Logger): The logger instance to log the error message if 'condition' is True.

    Returns
        None

    Raises:
        ValueError

    """
    raise_if_not(not condition, message, logger)
