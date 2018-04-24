#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -*- Python version: 3.6 -*-

import os
import time
import logging
import logging.handlers


def get_local_time_string(format="%b %d %H:%M:%S"):
    return time.strftime(format, time.localtime())


class ExpLogger(object):
        """
        Logging class.
        !! This class must be instantiated before use.
        @pythonVersion: 3.5.2
        @methods:
                __init__        Initiate logger instance. Change log level with
                                'loggername, loglevel = logging.LEVEL'.
                                LEVEL can be INFO/DEBUG/WARNING(default)/ERROR/CRITICAL.
                clean           Cleanup logfile for current logger.
        @author: X.Huang
        @creation: 2016-12-17
        @modified: 2016-12-17
        @version: 0.1
        """
        log = None
        logfile = None

        def __init__(self, loggername, loglevel=logging.WARNING):
            self.logfile = loggername + ".log"
            # Init logging handler
            handler = logging.handlers.RotatingFileHandler(self.logfile, maxBytes=1024 * 1024, backupCount=5,
                                                           encoding="utf-8")
            # Set up formatter for handler
            fmt = '%(asctime)s\t%(message)s'
            formatter = logging.Formatter(fmt)
            handler.setFormatter(formatter)

            # Set up logger
            self.log = logging.getLogger(loggername)
            self.log.addHandler(handler)
            self.log.setLevel(loglevel)

        def tee(self, msg, on_screen=True):
            if on_screen:
                print(msg)
            self.log.warning(msg)

        def clean(self):
            try:
                os.remove(self.logfile)
            except Exception as _Eall:
                self.log.error("Cannot clean up logfile: %s" % _Eall)