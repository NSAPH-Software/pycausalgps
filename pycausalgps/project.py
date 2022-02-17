"""
project.py
================================================
The core module for the Project class.
"""

import yaml
from os import path

from .database import DataBase

class Project:
    """ Project Class
    p1 = Project('project1')
    """
    
    def __init__(self, pr_name):
        self.pr_name = pr_name
        self.pr_db = None
        self.study_data = list()
        self._connect_to_database()

    # _instance = None

    # # There should be able to have more than one project. 
    # # convert it into an object. 
    # def __new__(cls, name):
    #     if cls._instance is None:
    #         cls._instance = super(Project,cls).__new__(cls)
    #         cls._instance.name = name
    #         cls._connect_to_database()
    #         cls._input_data = {}


    # @classmethod
    # def _connect_to_database(cls):
    #     """ Connect to the database or create one."""
    #     # All classes has one connection to database.
    #     # We may need to change it to include multi-thread access. 
    #     cls._instance.pr_db = DataBase(cls._instance.name+"_db",
    #                                    cache_size=2000)
        
    #     # Connect database to InputData class.
    #     #InputData.pr_db = cls._instance.pr_db

    def _connect_to_database(self):
        """ Connect to the database or create one."""
        # All classes has one connection to database.
        # We may need to change it to include multi-thread access. 
        self.pr_db = DataBase(self.pr_name+"_db",
                                       cache_size=2000)
        
        # Connect database to InputData class.
        #InputData.pr_db = cls._instance.pr_db




    def add_study_data(self, path_to_folder):
        # Adds an instance of input data.
        
        # Read metadata, and create a hash value.
        # Check database to see if it is available, 
        #      - if it is, retireve it.
        #      - if not, create a new instance and add to data.base. 
        

        # Read description.yml file.
        with open(path.join(path_to_folder, "description.yml"),
                  "r") as stream:
            try:
                self.data_path = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # extract info from description.yml file.
        # TODO

        # read exposure data
        # TODO

        # read confounders 
        # TODO

        # read outcome data
        # TODO

        # Check blob + description file hash values.
        # TODO

        # If the object is in the list of study_data retireve it from database.
        # TODO

        # If not, create an object of StudyData and put it inside:
        # 1) List of study_data
        # 2) database
        # 3) Any other controller list  and graph. 


    def __str__(self) -> str:
        return f"Project: {self.pr_name}"


    def __repr__(self) -> str:
        return "Is not implemented."


    @staticmethod
    def projects_list(db):
        pass
    
    @staticmethod
    def save_project(db, pr_name):
        pass
    
    @staticmethod
    def load_project(db, pr_name):
        pass
 