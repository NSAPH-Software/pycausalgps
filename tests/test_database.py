import unittest

from pycausalgps.database import Database

class TestGPS(unittest.TestCase):

    def test_add_retrieve_data(self):

        db = Database(db_path="test.db")
        db.set_value("test_key", "test_value")
        self.assertEqual(db.get_value("test_key"), "test_value")

        db.set_value("second_key", [1,2,3,4,5,6,7,8,9,10])
        self.assertEqual(db.get_value("second_key"), [1,2,3,4,5,6,7,8,9,10])
        db.close_db()
        
    def test_remove_data(self):

        db = Database(db_path="test.db")
        db.set_value("key1", "test_value")

        db.delete_value("key1")
        self.assertEqual(db.get_value("key1"), None)
        
        # trying to remove a data that does not exist.
        self.assertEqual(db.delete_value("key1"), None)

        db.close_db()

    def test_cache(self):

        db = Database(db_path="test.db")

        db.update_cache_size(10)

        # putting value in database does not change the cache.
        for i in range(10):
            db.set_value(f"key{i}", f"value{i}")

        self.assertEqual(len(Database._cache), 0)

        for i in range(10):
            _ = db.get_value(f"key{i}")
        
        # now the cache is full.
        self.assertEqual(len(Database._cache), 10)

        # Adding 10 new values to the database.
        for i in range(10, 20):
            db.set_value(f"key{i}", f"value{i}")
        
        # retrieving 3 more items.
        for i in range(10, 13):
            _ = db.get_value(f"key{i}")

        # cache size is still 10.
        self.assertEqual(len(Database._cache), 10)
