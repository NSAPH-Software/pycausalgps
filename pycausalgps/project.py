"""
project.py
================================================
The core module for the Project class.

"""

import os
import yaml
import hashlib
from os import path
import pandas as pd

from pycausalgps.log import LOGGER
from pycausalgps.database import Database
from pycausalgps.gps import GeneralizedPropensityScore

class Project:
    """ Project Class
    The Project class generates a project object with collecting the project's
    details. 

    Parameters
    ----------
    project_params : dict
    The parameters of the project. It should contain the following mandotary keys:
    | name: str
    | project_id: int
    | data.outcome_path: str
    | data.exposure_path: str
    | data.covariate_path: str

    Notes
    -----
    The project object does not load the data. It only stores the paths to the 
    data. Other than mandatory keys, the project_params can contain other keys. 

    Examples
    --------

    >>> from pycausalgps.project import Project
    >>> project_params = {"name": "test_project", "project_id": 1,
                          "data": {"outcome_path": "data/outcome.csv", 
                                   "exposure_path": "data/exposure.csv", 
                                   "covariate_path": "data/covariate.csv"}}
    >>> project = Project(project_params = project_params, db_path = "test.db")
    """

    def __init__(self, project_params, db_path):
        
        self.project_params = project_params
        self._check_params()
        self.pr_name = self.project_params.get("name")
        self.pr_id = self.project_params.get("project_id")
        self.pr_db_path = db_path
        self.hash_value = None
        self.gps_list = list()
        self._add_hash()
        self.db = None
        self._connect_to_database()

    def _check_params(self):
        #TODO: In case of raising exceptions, refer the users to the documentation.
        
        required_keys = ["name", "project_id", "data"]
        required_data_keys = ["outcome_path", "exposure_path", "covariate_path"]

        for key in required_keys:
            if key not in self.project_params.keys():
                raise Exception(f"In the project.yaml file, " \
                                f"please provide the '{key}' field.")
        for key in required_data_keys:
            if key not in self.project_params.get("data").keys():
                raise Exception(f"In the project.yaml file, "\
                                f"under the 'data' field, " \
                                f"please provide the '{key}' field.")


    def _connect_to_database(self):
        if self.pr_db_path is None:
            raise Exception("Database is not defined.")
            
        self.db = Database(self.pr_db_path)

    def check_input_data_quality(self):
        # This includes checking if the data is still accessible. 
        # if the format is supported.
        # if each data comes with id column.
        # if the id column matches the id column in the other data.
        pass

    def compute_gps(self, gps_params_path):
        """ Compute GPS
        This function computes the GPS for the project.

        Parameters
        ----------
        | gps_params_path : str 

        """
        # This includes loading a yaml file with gps parameters.
        # and creating a gps object.
        # The gps object will have access to the data (exposure and covariates.).  
        # and adding it to the database.
        # in the gps_param file, if we give more than one gps, it should create different gps objects.

        # load gps parameters
                # safe read yaml file
        if gps_params_path is not None:
            try:
                with open(gps_params_path, "r") as f:
                    gps_params = yaml.safe_load(f)
            except Exception as e:
                print(e)
                return
        else:
            print("Please provide a yaml file path for the project.")
            return

        
        # compute the combination of providied gps parameters
        # TODO: If a range of parameters are given we should make decision on how to compute the gps.
        
        
        # check if they are valid
    
        # create gps objects based on the parameters. 
        # TODO

        gps = GeneralizedPropensityScore(self.project_params, 
                                         gps_params, 
                                         db_path=self.pr_db_path)
        
        # check if the gps is already in the database.
        if gps.hash_value in self.gps_list:
            LOGGER.info(f"GPS is already computed, retireving from the database.")
            try:
                gps = self.db.get_value(gps.hash_value)
            except Exception as e:
                print(e)
                return
        else:
            gps.compute_gps()
            self.gps_list.append(gps.hash_value)
            self.db.set_value(gps.hash_value, gps)
            self.db.set_value(self.hash_value, self)
        


        # You can add the gps object to the database, with or without GPS value. We develop the GPS object with all available data and parameters but we do not compute it. 
        # Or we can have a plan to efficiently compute the GPS values. 

       

    # def add_study_data(self, path_to_folder):
    #     # Adds an instance of input data.
        
    #     # Read metadata, and create a hash value.
    #     # Check database to see if it is available, 
    #     #      - if it is, retireve it.
    #     #      - if not, create a new instance and add to data.base. 
        
    #     # Read yaml file.
    #     with open(path.join(path_to_folder, "parameters.yml"),
    #               "r") as stream:
    #         try:
    #             self.data_path = yaml.safe_load(stream)
    #         except yaml.YAMLError as exc:
    #             print(exc)

    #     # collect path

    #     exp_data_path = self.data_path.get("exposure")
    #     con_data_path = self.data_path.get("confounder")
    #     out_data_path = self.data_path.get("output")

    #     # read data from disk
    #     try: 
    #         exp_data = pd.read_csv(path.join(path_to_folder, exp_data_path))
    #         conf_data = pd.read_csv(path.join(path_to_folder, con_data_path))
    #         out_data = pd.read_csv(path.join(path_to_folder, out_data_path))
    #     except TypeError as e:
    #         print(f"Possibly wrong path: " + getattr(e, 'message', repr(e)))
    #     except FileNotFoundError as e:
    #         print(e)
    #     except Exception as e:
    #         print(e)

    #     # create StudyData object
    #     stdata_obj = StudyData(exp_data=exp_data, conf_data=conf_data,
    #                            outcome_data=out_data)


    #     if stdata_obj.hash_value in self.study_data:
    #         print("Data has been already loaded to the project object."+\
    #               "The command is ignored.")
    #     else:
    #         stdata_obj.add_meta_data(self.data_path.get("metadata"))
    #         self.study_data.append(stdata_obj.hash_value)
    #         stdata_obj.set_parent_node = self.hash_value
    #         self.db.set_value(stdata_obj.hash_value, stdata_obj)
    #         LOGGER.info(f"Study data has been added.")

    
        # TODO: This should be also added to Database controller. 

        # Check blob + description file hash values.
        # TODO

        # If the object is in the list of study_data retireve it from database.
        # TODO

        # If not, create an object of StudyData and put it inside:
        # 1) List of study_data
        # 2) database
        # 3) Any other controller list  and graph. 
    

    def __str__(self) -> str:

        return f"Project name: {self.pr_name} \n" +\
               f"Project id: {self.pr_id} \n" +\
               f"Project database: {self.pr_db_path} \n" +\
               f"Number of gps objects: {len(self.gps_list)} \n"


    def __repr__(self) -> str:
        return (f"Project({self.pr_name})")

    def _add_hash(self):
        
        # check the yaml file --------------------------------------------------
        if "name" not in self.project_params.keys() or self.project_params.get("name") is None:
            print("Please provide a project name.")
            return

        if "project_id" not in self.project_params.keys() or self.project_params.get("project_id") is None:
            print("Please provide a project id.")
            return

        if "data" not in self.project_params.keys() or \
                         self.project_params.get("data").get("outcome_path") is None or \
                         self.project_params.get("data").get("covariate_path") is None or \
                         self.project_params.get("data").get("exposure_path") is None:
            print("Please provide a path to the outcome, covariates and "+\
                  "treatment data.")
            return

        # create a hash string 
        outcome_path = self.project_params.get("data").get("outcome_path")
        exposure_path = self.project_params.get("data").get("exposure_path")
        covariate_path = self.project_params.get("data").get("covariate_path")

        outcome_name = path.basename(outcome_path).split('/')[-1]
        exposure_name = path.basename(exposure_path).split('/')[-1]
        covariate_name = path.basename(covariate_path).split('/')[-1]

        hash_string = "-".join([str(self.project_params.get("name")), str(self.project_params.get("id")), outcome_name, exposure_name, covariate_name])

        try:            
            self.hash_value =  hashlib.sha256(
                hash_string.encode('utf-8')).hexdigest()
        except Exception as e:
            print(e) 

        self.project_params["hash_value"] = self.hash_value

    def summary(self):
        if len(self.gps_list) == 0:
            print ("The project does not have any computed GPS object.")
        else:
            print(f"The project has {len(self.gps_list)} GPS object(s): ")
            for item in self.gps_list:
                gps = self.db.get_value(item)
                print(gps)


    def get_gps(self, gps_id):
        """
        Get a project object from the database.

        Parameters:
        -----------
        gps_id: str

        Returns:
        --------
        | gps: GeneralizedPropensityScore

        """

        gps_id_dict = {}
        for gps_hash in self.gps_list:
               gps_obj = self.db.get_value(gps_hash)
               gps_id_dict[gps_obj.gps_id] = gps_hash

        if gps_id in gps_id_dict.keys():
            return self.db.get_value(gps_id_dict[gps_id])
        else:
            print(f"A GPS object with id:{gps_id} is not defined.")
        


if __name__ == "__main__":
    project_params = { 'project_id': 20221027,
                       'name': 'cms_kidney_failure',
                       'details': {'description': 'Computing the effect of longterm pm2.5 exposure on kidney failure.', 'version': '1.0.0', 'authors': {'name': 'Naeem Khoshnevis', 'email': 'nkhoshnevis@g.harvard.edu'}}, 
                       'data': {'outcome_path': '/Users/nak443/Documents/Naeem_folder_mac_h/Research_projects/F2022_003_Harvard/p20221104_gps/code_package/pycausalgps/notebooks/project_abc/data/outcome.csv', 
                                'exposure_path': '/Users/nak443/Documents/Naeem_folder_mac_h/Research_projects/F2022_003_Harvard/p20221104_gps/code_package/pycausalgps/notebooks/project_abc/data/exposure.csv', 
                                'covariate_path': '/Users/nak443/Documents/Naeem_folder_mac_h/Research_projects/F2022_003_Harvard/p20221104_gps/code_package/pycausalgps/notebooks/project_abc/data/covariate.csv'}}

    current_dir = os.getcwd()
    db_path = path.join(current_dir, "database.sqlite")
    print(current_dir)
    pr = Project(project_params, db_path)
    pr.compute_gps( os.path.join(current_dir,"notebooks/project_abc", "gps_params_1.yaml"))

