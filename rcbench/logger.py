import logging
import sys

OUTPUT_LEVEL = 25
logging.addLevelName(OUTPUT_LEVEL, "OUTPUT")

# Define color ANSI codes
COLOR_RESET = "\033[0m"
COLORS = {
    logging.DEBUG: "\033[36m",     # Cyan
    logging.INFO: "\033[34m",      # Blue
    OUTPUT_LEVEL: "\033[32m",      # Green 
    logging.WARNING: "\033[33m",   # Yellow
    logging.ERROR: "\033[31m",     # Red
    logging.CRITICAL: "\033[41m",  # Red Background
}

def output(self, message, *args, **kwargs):
    if self.isEnabledFor(OUTPUT_LEVEL):
        self._log(OUTPUT_LEVEL, message, args, **kwargs)

logging.Logger.output = output

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        level_color = COLORS.get(record.levelno, COLOR_RESET)
        levelname = f"{level_color}[{record.levelname}]{COLOR_RESET}"

        formatter = logging.Formatter(
            f"%(asctime)s {levelname} %(name)s — %(message)s",
            datefmt="%H:%M:%S"
        )

        return formatter.format(record)

def get_logger(name: str = "rcda"):
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(ColoredFormatter())
        logger.addHandler(handler)

        logger.propagate = False
        # Keep level unset here to control externally
        # logger.setLevel(logging.INFO)

    return logger
