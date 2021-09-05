import json
import logging
import numpy as np

class WeightedRunningAverage:
    """
    Class that maintains the weighted running average of a scalar quantity
    """

    def __init__(self):
        self.weights = 0
        self.total = 0

    def update(self, val, w):
        if w >= 0.:
            self.total += val * w
            self.weights += w
        else:
            raise ValueError("Weights should be nonnegative")

    def __call__(self):
        if self.weights == 0.:
            return 0.  # If weights are zero, update has never been called (by assumption that weights >= 0)
        else:
            return self.total / float(self.weights)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)