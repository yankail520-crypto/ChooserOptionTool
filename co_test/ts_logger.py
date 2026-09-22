import logging

from common import logger


logg = logger.get_logger(__name__, logging.DEBUG)


logg.error("test error")
logg.info("test info")
logg.debug("test debug")