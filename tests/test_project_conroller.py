import unittest
import os
import shutil
import yaml
from pycausalgps.project_controller import ProjectController

class TestProjectController(unittest.TestCase):

    def setUp(self):
        self.pc = ProjectController(db_path="test.db")
        self.test_folder = "test_folder"

        if not os.path.exists(self.test_folder):
            os.makedirs(self.test_folder)

        # create a project json file. 
        project_params = {
            'name': 'cms_heart_failure',
            'project_id': 20221025,
            'details': {
                'description': 'Computing the effect of longterm pm2.5 exposure on lung cancer.',
                'version': '1.0.0',
                'authors': {
                    'name': 'Naeem Khoshnevis',
                    'email': 'nkhoshnevis@g.harvard.edu'
                }
            },
            'data': {
                'outcome_path': 'project_abc/data/outcome.csv',
                'exposure_path': 'project_abc/data/exposure.csv',
                'covariate_path': 'project_abc/data/covariate.csv'
            }
        }

        yaml_content = yaml.dump(project_params, default_flow_style=False)

        with open(os.path.join(self.test_folder, "project.yaml"), "w") as f:
            f.write(yaml_content)

    def tearDown(self):
        if os.path.exists(self.test_folder):
            shutil.rmtree(self.test_folder)

        if os.path.exists("test.db"):
            os.remove("test.db")

    def test_create_project(self):
        self.pc.create_project(folder_path=self.test_folder)
        projects_list = self.pc.db.get_value("PROJECTS_LIST")
        self.assertEqual(len(projects_list), 1)

    def test_connect_to_project(self):
        self.pc.create_project(folder_path=self.test_folder)
        self.pc.connect_to_project(folder_path=self.test_folder)
        projects_list = self.pc.db.get_value("PROJECTS_LIST")
        self.assertEqual(len(projects_list), 1)

    def test_remove_project(self):
        self.pc.create_project(folder_path=self.test_folder)
        self.pc.remove_project("cms_heart_failure")
        projects_list = self.pc.db.get_value("PROJECTS_LIST")
        self.assertEqual(len(projects_list), 0)



if __name__ == '__main__':
    unittest.main()