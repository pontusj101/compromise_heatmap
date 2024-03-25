import logging

debug = False

def setup_logging(name):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    fmt = '%(asctime)s - %(levelname)s %(name)s [%(filename)s:%(lineno)d] (%(funcName)s) %(message)s'

    logging.basicConfig(level=logging.WARNING,
                        format=fmt,
                        datefmt = '%Y-%m-%d %H:%M:%S',
                        handlers=[
                            cli_handler := logging.StreamHandler(),
                            logging.FileHandler('heatmap.log')
                        ])
    if debug:
        for logger_name in ['malpzsim', __name__]:
            logger = logging.getLogger('malpzsim')
            logger.setLevel(logging.DEBUG)
            logger.addHandler(cli_handler)

    return logging.getLogger(name)
