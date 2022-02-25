"""
study_data.py
================================================
The core module for the StudyData class.
"""

import hashlib
from logging import Logger
import pickle

from pycausalgps.log import LOGGER

class StudyData:
    """ StudyData Class"""

    pr_db = None
    processing_labels = dict()
    label_types = {
        'gps_xgb': {'ntrees': 10}
    }


    def __init__(self, exp_data, conf_data, outcome_data):
        self._exposure = None
        self._confounder = None
        self._outcome = None
        self.hash_value = None
        self.parent_node = None
        self.st_d_name = None
        self.meta_data = dict()
        self.gps_values = list()
        self._add_exposure_data(exp_data)
        self._add_confounder_data(conf_data)
        self._add_outcome_data(outcome_data)
        self._compute_hash_value()


    def _add_exposure_data(self, exp_data):
        # TODO: sanity checks
        self._exposure = exp_data


    def _add_confounder_data(self, conf_data):
        # TODO: sanity checks
        self._confounder = conf_data


    def _add_outcome_data(self, outcome_data):
        # TODO: sanity checks
        self._outcome = outcome_data

    
    def set_parent_node(self, pnode_hash):
        # Parent nodes hash.
        self.p_node_hash = pnode_hash


    def add_meta_data(self, meta_data):
        self.meta_data.update(meta_data)
        self.st_d_name = self.meta_data['data_name']


    def _compute_hash_value(self):

        try:
            data_tuple = pickle.dumps((self._exposure, self._confounder,
                                       self._outcome))
            
            self.hash_value =  hashlib.sha256(data_tuple).hexdigest()
        except Exception as e:
            print(e)

    @classmethod
    def add_processing_label(cls, label_name, label_type, params):
        """ Creates a processing label"""

        if label_name in cls.processing_label:
            LOGGER.warning(f"Label name: {label_name} has been used."+\
                           f"Try another name.")
            return

        if label_type not in cls.label_types:
            LOGGER.warning(f"Label type is not supported.")
            return

        for ak in params.keys():
            if ak not in list(cls.label_types[label_type].keys()):
                LOGGER.warning(f" '{ak}' is not a valid argument for "
                               f" {label_type}."
                               f" List of valid arguments:"
                               f" {list(cls.label_types[label_type].keys())}")
                return

        for rak in list(cls.label_types[label_type].keys()):
            if rak not in params.keys():
                LOGGER.warning(f" '{rak}' is not provided"
                               f" List of arguments: "
                               f" {list(cls.label_types[label_type].keys())}")
                return

        
        #TODO Sanity check for provided parameters.

        cls.processing_labels[label_name] = [label_type, params]



    @staticmethod
    def _apply(study_data, label_name):
        """ Applies the requested processing label on the record. Returns a 
        GPS object."""

        pass

          





    
 






