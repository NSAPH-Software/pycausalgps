"""
project_controller.py
================================================
The core module for the ProjectController class.
"""


from logging import warning

import os
import yaml

from project import Project
from database import Database
from pycausalgps.log import LOGGER


class ProjectController:

    _instance = None

    def __new__(cls, db_path):
        if not cls._instance:
            cls._instance = super(ProjectController, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path):

            self.db_path = db_path
            self.db = Database(db_path=self.db_path)
            self.projects_list = list()
            self._update_project_list()
            # self._update_reserved_keys()

    def _update_project_list(self):
        # TODO: How to protect this key from others access?
        if self.db.get_value("PROJECTS_LIST") is None:
            self.db.set_value("PROJECTS_LIST", list())
        else:
            self.projects_list = self.db.get_value("PROJECTS_LIST")

    # def _update_reserved_keys(self):
    #     self.db.set_value("RESERVED_KEYS", ["RESERVED_KEYS", "PROJECTS_LIST"])

    def connect_to_project(self, folder_path=None):
        """
        Connect to a project. If the project is not defined, it will be created.

        Parameters
        ----------
        folder_path : str
            The path to the project folder.
  
        """

        # safe read yaml file
        if folder_path is not None:
            try:
                with open(os.path.join(folder_path, "project.yaml"), "r") as f:
                    project_params = yaml.safe_load(f)
            except Exception as e:
                print(e)
                return
        else:
            print("Please provide a yaml file path for the project.")
            return
        
        p_obj = Project(project_params=project_params, db_path=self.db_path)
   
        c_pr = self.db.get_value("PROJECTS_LIST")

        if p_obj.hash_value in c_pr:
            print("Project has been already submitted to the database. ")
            p_obj = self.db.get_value(p_obj.hash_value)
        else:
            c_pr.append(p_obj.hash_value)     
            self.db.set_value("PROJECTS_LIST", c_pr)
            self._update_project_list()

            # Add project to the database
            self.db.set_value(p_obj.hash_value, p_obj)

            # This step is not efficient. It adds set value back into cache. 
            # However, I keep it until a better solution.

            # if self.db.get_value(p_obj.hash_value).hash_value != p_obj.hash_value:
            #     print("Something is wrong with storing object on database.") 

            print(f"Project {project_params.get('name')} has been successfully added to the database.")

    def remove_project(self, project_name):
        self._update_project_list()

        
        # retrerive project hash value
        pr_name_dict = {}
        for project_hash in self.projects_list:
               pr = self.db.get_value(project_hash)
               pr_name_dict[pr.pr_name] = project_hash

        if project_name in pr_name_dict.keys():
            p_obj = self.db.get_value(pr_name_dict[project_name])
        else:
            print("Project is not defined.")
            return
        
        if p_obj.hash_value in self.projects_list:
            # Project is already defined and should be retireved from db.
            p_obj = self.db.get_value(p_obj.hash_value)
            if len(p_obj.gps_list) > 0:
                LOGGER.warning(f"Project {p_obj.pr_name} has "+\
                               f"{len(p_obj.study_data)} GPS object and "+\
                               f"cannot be deleted. First GPS objects.")
                return
            self.projects_list.remove(p_obj.hash_value)
            self.db.delete_value(p_obj.hash_value)
            self.db.set_value("PROJECTS_LIST", self.projects_list)
            del p_obj
            print(f"Project '{project_name}' has been successfully deleted.")
        else:
            print(f"Project '{project_name}' is not defined.")

    def summary(self):
        
        try:
           self._update_project_list()
           print(f"Number of projects in the database: "+ 
                 f"{len(self.projects_list)}\n")
           for project in self.projects_list:
               pr = self.db.get_value(project)
               print(f"  {pr.pr_name}")
        except Exception as e:
            print(e)


    def get_project(self, pr_name):
        
        self._update_project_list()
        pr_name_dict = {}
        for project_hash in self.projects_list:
               pr = self.db.get_value(project_hash)
               pr_name_dict[pr.pr_name] = project_hash

        if pr_name in pr_name_dict.keys():
            return self.db.get_value(pr_name_dict[pr_name])
        else:
            print("Project is not defined.")



if __name__ == "__main__":
    
   cwd = os.getcwd()
   script2 = os.path.join(cwd, "scripts2")
   db_path = os.path.join(script2, "database.sqlite")

   # start an instance of the project controller
   pc = ProjectController(db_path=db_path)

   # add a project
   pc.connect_to_project(folder_path=os.path.join(script2, "project_abc"))
   
   # add another project
   pc.connect_to_project(folder_path=os.path.join(script2, "project_efg"))
   pc.connect_to_project(folder_path=os.path.join(script2, "project_hij"))
   pc.connect_to_project(folder_path=os.path.join(script2, "project_101"))
   
   # This input file has problem with covariate field. 
   #pc.connect_to_project(folder_path=os.path.join(script2, "project_102"))
   # look at projects list
   pc.summary()

   #remove a project
   pc.remove_project("cms_kidney_failure")

   pc.summary()

   # get a project
   pr = pc.get_project("cms_kidney_failure")

   print(pr.params)

   # print current working dir
    # print(os.getcwd())

# pc = ProjectController(db_path="test.db") 
# pc.summary()
# current_dir = os.path.dirname(os.path.abspath(__file__))
# folder_path = os.path.join(current_dir, "myproject_folder")
# pc.connect_to_project(folder_path=folder_path)
# pc.summary()