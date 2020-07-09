import os
import random

import numpy as np


class UUIDIterator:
    """Helper class, providing a iterator to go over the available training data"""

    def __init__(self, storage_path: str):
        """Initialize UUID Iterator class.

        Parameters
        ----------
        storage_path : str
            Path to data storage
        """

        self.path = storage_path

    def __call__(self, batch_size=None):
        """Returns list of uuids in specified storage. If a batch_size is given,
        the uuids get separtead into batches.
        By default, only single uuids are returned, which corresponds to a batch size of 1.

        Parameters
        ----------
        batch_size : None, optional
            Number of uuids in each batch

        Returns
        -------
        uuids : List of lists [[batch_1],[batch_2], .... [batch_n]]
            Each batch consists of `batch_size` uuids. By default, batch size is set
            to one.
        """

        uuids = [img.split(".")[0] for img in os.listdir(self.path + "/images")]
        random.shuffle(uuids)

        if batch_size and (batch_size > 1):
            num_batches = len(uuids) // batch_size
            uuid_batches = [
                uuids[(_ * batch_size) : (_ * batch_size + batch_size)]
                for _ in range(num_batches)
            ]
            return np.array(uuid_batches)

        return np.array([[uuid] for uuid in uuids])
