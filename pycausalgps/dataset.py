"""
dataset.py
==================================
The core module for the Dataset class.
"""

import pandas as pd

class Dataset:
    """ Dataset class """

    def __init__(self, data=None):
        
        if data is None:
            self.data = None
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise RuntimeError("Data type is not supported")

        
    def read_from_disk(self, path, format="csv"):
        """
        Reads data from disk based on the provided format.

        Inputs:
            | path: path to the file. It should include the file name. 
            | format: format of the file. Supported formats: csv
        """
        if format == "csv":
            self.data = pd.read_csv(path)


    def write_to_disk(self, path, format="csv"):
        """
        Writes data from disk based on the provided format.

        Inputs:
            | path: path to the file. It should include the file name. 
            | format: format of the file. Supported formats: csv
        """
        if format == "csv":
            self.data.to_csv(path)

    def read_from_database(self):
        pass


    def write_to_database(self):
        pass


    def compute_covariate_balance(self):
        pass

    def __str__(self):
        return self.data.info()