import unittest

from pycausalgps.project import Project

class TestGPS(unittest.TestCase):


    def test_create_instance(self):
        project_params = {"name": "test_project", "project_id": 136987,
                          "data": {"outcome_path": "data/outcome.csv", 
                                   "exposure_path": "data/exposure.csv", 
                                   "covariate_path": "data/covariate.csv"}}
        project = Project(project_params = project_params, db_path = "test.db")

        self.assertEqual(project.hash_value, 
        "259b62ca9caa789921d14bb0cc8959975270925ce23f30345d6db379f770bc5b")        