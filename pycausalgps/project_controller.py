"""
project_controller.py
================================================
The core module for the ProjectController class.
"""


from .database import Database
from .project import Project

import pickle


class ProjectController:

    _instance = None

    def __new__(cls, dbname):
        if not cls._instance:
            cls._instance = super(ProjectController, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_path):

            self.db_path = db_path
            self.db = Database(dbname=self.db_path, cache_size=1000)
            self.connected = True
            self.projects_list = list()
            self.update_project_list()


    def update_project_list(self):
        # TODO: How to protect this key from others access?
        if self.db.get_value("PROJECTS_LIST") is None:
            self.db.set_value("PROJECTS_LIST", [])
        else:
            self.projects_list = self.db.get_value("PROJECTS_LIST")

    def add_project(self, project_name):

        p_obj = Project(pr_name=project_name, db_path=self.db_path)
        c_pr = self.db.get_value("PROJECTS_LIST")
        if p_obj.hash_value in c_pr:
            print("Project has been already submitted to the database. "+\
                  "Ignoring this command.")
        else:
            c_pr.append(p_obj.hash_value)      
            self.db.set_value("PROJECTS_LIST", c_pr)
            self.projects_list.append(p_obj.hash_value)
            self.db.set_value(p_obj.hash_value, p_obj)
            print(f"Project {project_name} has been successfully added to the database.")

    def remove_project(self, project_name):
        c_pr = self.db.get_value("PROJECTS_LIST")
        p_obj = Project(pr_name=project_name, db_path=self.db_path)
        if p_obj.hash_value in c_pr:
            c_pr.remove(p_obj.hash_value)
            self.db.set_value("PROJECTS_LIST", c_pr)
            del p_obj
            print(f"Project '{project_name}' has been successfully deleted.")
        else:
            print(f"Project '{project_name}' is not defined. Ignoring this command.")

    def summary(self):
        
        try:
           self.update_project_list()
           print(f"Number of projects in the database: "+ 
                 f"{len(self.projects_list)}\n")
           for project in self.projects_list:
               pr = self.db.get_value(project)
               print(pr.pr_name)
        except Exception as e:
            print(e)


    def get_project(self, pr_name):
        
        self.update_project_list()
        pr_name_dict = {}
        for project_hash in self.projects_list:
               pr = self.db.get_value(project_hash)
               pr_name_dict[pr.pr_name] = project_hash


        if pr_name in pr_name_dict.keys():
            return self.db.get_value(pr_name_dict[pr_name])
        else:
            print("Project is not defined.")


