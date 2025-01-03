import logging
import json
import sys
import traceback
import os

from settings import settings

logging.basicConfig(
        level=settings.LOG_LEVEL,
        format='%(message)s'
    )


class JSONAdapter(logging.LoggerAdapter):
    def __init__(self, logger, extra=None):
        super().__init__(logger, extra or {})

    def process(self, msg, kwargs):
        level = kwargs.pop('level', self.logger.getEffectiveLevel())
        level_name = logging.getLevelName(level).lower()

        log_data = {
            "message": str(msg),
            "level": level_name
        }

        if level_name.upper() in ['ERROR', 'CRITICAL', 'EXCEPTION']:
            exc_info = kwargs.get('exc_info')
            if exc_info:
                if isinstance(exc_info, bool):
                    exc_info = sys.exc_info()
                if isinstance(exc_info, tuple) and len(exc_info) == 3:
                    exc_type, exc_value, exc_traceback = exc_info
                    tb = traceback.extract_tb(exc_traceback)
                    if tb:
                        filename, lineno, _, _ = tb[-1]  # Get the last frame in the traceback
                        log_data["file"] = f"{os.path.basename(filename)}:{lineno}"
                    log_data["stack"] = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                kwargs['exc_info'] = False
            output_key = "error"
            if kwargs.pop('is_exception', False):
                log_data["level"] = "exception"
        else:
            output_key = level_name
            log_data["file"] = self.get_caller_file()

        log_data.update(self.extra)
        if "extra" in kwargs:
            log_data.update(kwargs["extra"])

        output = {output_key: log_data, "level": log_data["level"]}
        return json.dumps(output), kwargs

    def get_caller_file(self):
        current_file = os.path.normcase(os.path.abspath(__file__))
        for frame_info in traceback.extract_stack():
            filename, lineno, _, _ = frame_info
            if os.path.normcase(os.path.abspath(filename)) != current_file:
                return f"{os.path.basename(filename)}:{lineno}"
        
        frame = sys._getframe(1)
        return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"

    def log(self, level, msg, *args, **kwargs):
        if self.isEnabledFor(level):
            kwargs['level'] = level
            msg, kwargs = self.process(msg, kwargs)
            self.logger._log(level, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if 'exc_info' not in kwargs:
            kwargs['exc_info'] = sys.exc_info()
        self.log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        kwargs['exc_info'] = True
        kwargs['is_exception'] = True
        self.log(logging.ERROR, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.log(logging.WARNING, msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self.log(logging.CRITICAL, msg, *args, **kwargs)


def setup_logger(name):
    logger = logging.getLogger(name)
    
    logger.debug(f"Init logger {name} with logging level {settings.LOG_LEVEL}")
    
    return JSONAdapter(logger)