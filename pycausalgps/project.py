"""
project.py
================================================
The core module for the Project class.
"""

from matplotlib import projections
import yaml
from os import path
import pandas as pd
import hashlib

from pycausalgps.log import LOGGER

from .database import Database
from .study_data import StudyData

class Project:
    """ Project Class
    p1 = Project('project1')
    """

  
    def __init__(self, pr_name, db_path):
        
        self.pr_name = pr_name
        self.pr_db_path = db_path
        self.hash_value = None
        self.study_data = list()
        self._add_hash()
        self._connect_to_database()

    def _connect_to_database(self):
        print(f"Projects sqlite database name: {self.pr_db_path}")
        if self.pr_db_path is None:
            raise Exception("Database is not defined.")
            
        self.db = Database(self.pr_db_path)

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

        # collect path

        exp_data_path = self.data_path.get("exposure")
        con_data_path = self.data_path.get("confounder")
        out_data_path = self.data_path.get("output")

        # read data from disk
        try: 
            exp_data = pd.read_csv(path.join(path_to_folder, exp_data_path))
            conf_data = pd.read_csv(path.join(path_to_folder, con_data_path))
            out_data = pd.read_csv(path.join(path_to_folder, out_data_path))
        except TypeError as e:
            print(f"Possibly wrong path: " + getattr(e, 'message', repr(e)))
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(e)

        # create StudyData object
        stdata_obj = StudyData(exp_data=exp_data, conf_data=conf_data,
                               outcome_data=out_data)


        if stdata_obj.hash_value in self.study_data:
            print("Data has been already loaded to the project object."+\
                  "The command is ignored.")
        else:
            stdata_obj.add_meta_data(self.data_path.get("metadata"))
            self.study_data.append(stdata_obj.hash_value)
            stdata_obj.set_parent_node = self.hash_value
            self.db.set_value(stdata_obj.hash_value, stdata_obj)
            LOGGER.info(f"Study data has been added.")

    
        # TODO: This should be also added to Database controller. 

        # Check blob + description file hash values.
        # TODO

        # If the object is in the list of study_data retireve it from database.
        # TODO

        # If not, create an object of StudyData and put it inside:
        # 1) List of study_data
        # 2) database
        # 3) Any other controller list  and graph. 
    
    
    def remove_study_data(self, st_data_name):
        pass


    def __str__(self) -> str:
        
        pr_details = f"Project name: {self.pr_name} \n" +\
                     f"Project database: {self.pr_db_path}"
        
        return pr_details


    def __repr__(self) -> str:
        return (f"Project({self.pr_name})")



    def _add_hash(self):
        try:            
            self.hash_value =  hashlib.sha256(
                self.pr_name.encode('utf-8')).hexdigest()
        except Exception as e:
            print(e) 

    def summary_study_data(self):

        if len(self.study_data) == 0:
            print ("The project does not have any study data.")
        else:
            print(f"The project has {len(self.study_data)} study data: ")
            for item in self.study_data:
                st_data = self.db.get_value(item)
                print(st_data.st_d_name)