import logging

# logger config function
def configure_logger(log_filename='data_collection/default_application.log', level=logging.INFO):
    logging.basicConfig(
        filename=log_filename,
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

# log record function
def log_status(module_name, status_message, level='info'):
    logger = logging.getLogger(module_name)
    
    if level.lower() == 'info':
        logger.info(status_message)
    elif level.lower() == 'warning':
        logger.warning(status_message)
    elif level.lower() == 'error':
        logger.error(status_message)
    elif level.lower() == 'debug':
        logger.debug(status_message)
    else:
        logger.info(status_message)