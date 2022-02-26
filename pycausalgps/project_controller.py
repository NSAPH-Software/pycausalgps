"""
project_controller.py
================================================
The core module for the ProjectController class.
"""


from logging import warning

from pycausalgps.log import LOGGER
from .database import Database
from .project import Project

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
            self._update_reserved_keys()

    def _update_project_list(self):
        # TODO: How to protect this key from others access?
        if self.db.get_value("PROJECTS_LIST") is None:
            self.db.set_value("PROJECTS_LIST", [])
        else:
            self.projects_list = self.db.get_value("PROJECTS_LIST")

    def _update_reserved_keys(self):
        self.db.set_value("RESERVED_KEYS", ["RESERVED_KEYS", "PROJECTS_LIST"])

    def add_project(self, project_name):

        p_obj = Project(pr_name=project_name, db_path=self.db_path)
   
        c_pr = self.db.get_value("PROJECTS_LIST")

        if p_obj.hash_value in c_pr:
            print("Project has been already submitted to the database. "+\
                  "Ignoring this command.")
        else:
            c_pr.append(p_obj.hash_value)      
            self.db.set_value("PROJECTS_LIST", c_pr)
            self._update_project_list()

            # with pickle.dump
            self.db.set_value(p_obj.hash_value, p_obj)

            # This step is not efficient. It adds set value back into cache. 
            # However, I keep it until a better solution.
            if self.db.get_value(p_obj.hash_value).hash_value != p_obj.hash_value:
                print("Something is wrong with storing object on database.") 
            print(f"Project {project_name} has been successfully added to the database.")

    def remove_project(self, project_name):
        self._update_project_list()
        p_obj = Project(pr_name=project_name, db_path=self.db_path)
        if p_obj.hash_value in self.projects_list:
            # Project is already defined and should be retireved from db.
            p_obj = self.db.get_value(p_obj.hash_value)
            if len(p_obj.study_data) > 0:
                LOGGER.warning(f"Project {p_obj.pr_name} has "+\
                               f"{len(p_obj.study_data)} study data and "+\
                               f"cannot be deleted. First remove study data.")
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
               print(pr.pr_name)
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