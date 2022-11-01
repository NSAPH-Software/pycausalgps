import unittest

from pycausalgps.project import Project

class TestGPS(unittest.TestCase):


    def test_create_instance(self):
        project_params = {"name": "test_project", "id": 136987,
                          "data": {"outcome_path": "data/outcome.csv", 
                                   "exposure_path": "data/exposure.csv", 
                                   "covariate_path": "data/covariate.csv"}}
        project = Project(project_params = project_params, db_path = "test.db")

        self.assertEqual(project.hash_value, 
        "80d8dae0c7b8af4d0a09bdd3587967c2882a02cd8a9865510600b7c41af22794")        