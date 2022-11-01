import unittest

from pycausalgps.project_controller import ProjectController

class TestGPS(unittest.TestCase):

    def test_create_instance(self):

        pc = ProjectController(db_path="test.db")
        self.assertEqual(pc.db_path, "test.db")
        self.assertEqual(pc.db.get_value("PROJECTS_LIST"), [])
