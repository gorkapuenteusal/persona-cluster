import logging
import os


def setup_logging():
    log_filename = os.path.join(os.path.dirname(__file__), '..', 'text_clustering.log')
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_filename)
                        ])
