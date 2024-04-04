import logging
import warnings

debug = False


def setup_logging(name=None):
    warnings.filterwarnings(
        "ignore",
        message="Tight layout not applied. tight_layout cannot make axes height small enough to accommodate all axes decorations",
        module="pyRDDLGym.Visualizer.ChartViz",
    )
    warnings.filterwarnings(
        "ignore",
        message="Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all axes decorations.",
        module="pyRDDLGym.Visualizer.ChartViz",
    )
    warnings.filterwarnings(
        "ignore",
        message="Attempting to set identical low and high ylims makes transformation singular; automatically expanding.",
        module="pyRDDLGym.Visualizer.ChartViz",
    )

    name = name or __package__

    root_logger = logging.getLogger()

    for handler in root_logger.handlers.copy():
        root_logger.removeHandler(handler)

    fmt = "%(asctime)s - %(levelname)s [%(module)s.%(funcName)s:%(lineno)d]: %(message)s"

    logging.basicConfig(
        level=logging.WARNING,
        format=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            cli_handler := logging.StreamHandler(),
            logging.FileHandler("heatmap.log"),
        ],
    )

    logging.getLogger('malpzsim').setLevel(logging.WARN)
    for logger_name in [name]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG if debug else logging.INFO)
        logger.addHandler(cli_handler)
        logger.propagate = False

    return logging.getLogger(name)
