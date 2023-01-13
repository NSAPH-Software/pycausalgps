"""
project_controller.py
================================================
The core module for the ProjectController class.
"""

import os
import yaml

from pycausalgps.log import LOGGER
from pycausalgps.project import Project
from pycausalgps.database import Database


class ProjectController:
    """ ProjectController class   

    The ProjectController class manages the projects. It provides suite of
    methods to add, remove, and connect to projects. It also provides a summary
    method to print the list of projects. Each project is defined by a folder 
    with a project.yaml file. 

    Parameters
    ----------
    db_path: str
        Path to the database file.
    """

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

    def _update_project_list(self):
        """ The PROJECTS_LIST is a list of available projects' hash values."""
        # TODO: How to protect this key from others access?
        if self.db.get_value("PROJECTS_LIST") is None:
            self.db.set_value("PROJECTS_LIST", list())
        else:
            self.projects_list = self.db.get_value("PROJECTS_LIST")

    def connect_to_project(self, folder_path=None):
        """
        Connect to a project. If the project is not defined, it will be created.
        The project is defined by a folder with a `project.yaml` file inside. 

        Parameters
        ----------
        folder_path : str
            A path to the project folder.
        """

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
        
        # sanity check for the project parameters are defined inside the 
        # Project class.
        p_obj = Project(project_params=project_params, db_path=self.db_path)
   
        c_pr = self.db.get_value("PROJECTS_LIST")

        if p_obj.hash_value in c_pr:
            print("The project has been already submitted to the database. "
                  + "Retrieving the project object from the database.")
            p_obj = self.db.get_value(p_obj.hash_value)
        else:
            c_pr.append(p_obj.hash_value)     
            self.db.set_value("PROJECTS_LIST", c_pr)
            self._update_project_list()

            # Add project to the database
            self.db.set_value(p_obj.hash_value, p_obj)
            print(f"Project {project_params.get('name')} " 
                  + f"has been successfully added to the database.")

    def remove_project(self, project_name):
        """
        Remove a project from the database, the list of projects, and the
        in-memory cache. Run pc.summary() to see the list of projects.  

        Parameters
        ----------
        project_name: str
            Name of the project to be removed.
        """
        self._update_project_list()

        # retrerive project hash value
        pr_name_dict = {}
        for project_hash in self.projects_list:
               pr = self.db.get_value(project_hash)
               pr_name_dict[pr.pr_name] = project_hash

        if project_name in pr_name_dict.keys():
            p_obj = self.db.get_value(pr_name_dict[project_name])
        else:
            print("Project is not defined." 
                  + "Use pc.summary() to see the list of projects.")
            return
        
        if p_obj.hash_value in self.projects_list:
            # Project is already defined and should be retireved from db.
            p_obj = self.db.get_value(p_obj.hash_value)
            if len(p_obj.gps_list) > 0:
                LOGGER.warning(
                    f"Project {p_obj.pr_name} has "
                    + f"{len(p_obj.study_data)} GPS object(s) and "
                    + f"cannot be deleted. First remove GPS object(s).")
                return
            self.projects_list.remove(p_obj.hash_value)
            self.db.delete_value(p_obj.hash_value)
            self.db.set_value("PROJECTS_LIST", self.projects_list)
            del p_obj
            print(f"Project '{project_name}' has been successfully deleted.")
        else:
            print(f"Project '{project_name}' is not defined.")

    def summary(self):
        """
        Print the number of available projects with project names. 

        """
        
        try:
           self._update_project_list()
           print(f"Number of projects in the database: " 
                 + f"{len(self.projects_list)}\n")
           for project in self.projects_list:
               pr = self.db.get_value(project)
               print(f"  {pr.pr_name}")
        except Exception as e:
            print(e)


    def get_project(self, pr_name):
        """
        Get a project object from the database.

        Parameters
        ----------
        pr_name: str
            Name of the project to be retrieved.

        Returns
        -------
        project: Any
            The project object.

        """
        
        self._update_project_list()
        pr_name_dict = {}
        for project_hash in self.projects_list:
            pr = self.db.get_value(project_hash)
            pr_name_dict[pr.pr_name] = project_hash

        if pr_name in pr_name_dict.keys():
            return self.db.get_value(pr_name_dict[pr_name])
        else:
            print("Project is not defined.")

    def __str__(self) -> str:
        return (f"A project controller connected to the database: "
                + f"{self.db_path}, "
                + f"with {len(self.projects_list)} projects. "
                + f"Use pc.summary() to see the list of projects.")

    def __repr__(self) -> str:
        return f"ProjectController(db_path='{self.db_path}')"