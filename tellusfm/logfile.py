import logging 
import sys 


def local_print_log(statement, level = 'info'):
    '''print and log statments to a file

    Parameters
    ---------
        statement : string
            the print/log statement

        level : string
            the log level, info, debug, warning, error, critical. Default is info

    Returns
    --------
        None

    Notes
    -------
    print statments in pydfnworks should generally be replaced with this print_log function. Use local_print_log if function is not in refernce to DFN object
    '''

    if level == 'info':
        print(statement)
        logging.info(statement)
    elif level == 'debug':
        print(statement)
        logging.debug(statement)
    elif level == 'warning':
        print(statement)
        logging.warning(statement)
    elif level == 'error':
        logging.error(statement)
        sys.stderr.write(statement)
        sys.exit(1)
    elif level == 'critical':
        logging.critical(statement)
        sys.stderr.write(statement)
        sys.exit(1)
    else:
        tmp_statement = f"Unknown logging level requested: {level}. Using warning"
        print(tmp_statement)
        print(statement)
        logging.warning(tmp_statement)
        logging.warning(statement)
        
def initialize_log_file(args):
    """
    Initialize and configure the experiment log file.

    This function sets up Python's built-in ``logging`` module to
    write logs to a specified file. If the provided filename does
    not end with ``.log``, the extension is automatically appended.
    Once initialized, all log messages will be written to the file
    and can also be routed through ``local_print_log`` for consistent
    console and file output.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments. Must include the attribute
        ``log_filename`` specifying the path/name of the log file.
        Example: ``args.log_filename = "experiment"`` will create
        ``experiment.log``.

    Returns
    -------
    None
        This function only configures logging; no values are returned.

    Notes
    -----
    - Log level is set to ``INFO`` by default.
    - The log file is opened in write mode (``filemode="w"``), meaning
      previous logs with the same filename will be overwritten.
    - Log format is ``<LEVEL> <MESSAGE>``, but can be extended to include
      timestamps or module names if needed.
    - A startup message confirming the log file initialization is written
      via ``local_print_log``.
    """

    logging.getLogger(__name__)
    if not args.log_filename.endswith('.log'):
        args.log_filename += '.log'

    logging.basicConfig(level = logging.INFO, filename=args.log_filename, filemode="w"
                        , format="%(levelname)s %(message)s" )
    statement = f"Initializing logfile: {args.log_filename}"
    local_print_log(statement)