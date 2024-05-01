from pathlib import Path

import logging
import os

import yaml


def get_handler(path, log_name):
    log_file_path = os.path.join(path, log_name)
    try:
        if not os.path.exists(path):
            print("We are creating the logger files")
            os.makedirs(path)
    except:
        pass
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    return file_handler, stream_handler
