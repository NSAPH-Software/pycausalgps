import unittest
import os
import sys
import io

from pycausalgps.database import Database
from pycausalgps.base.utils import human_readible_size
from sqlitedict import SqliteDict
from unittest.mock import MagicMock

class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.db = Database(db_path="test.db")

    def tearDown(self):
        self.db.close_db()
        if os.path.exists("test.db"):
            os.remove("test.db")

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


    def test_summary(self):
        db_name = "test.db"
        db = Database(db_path=db_name)
        
        db.set_value("test_key", "test_value")

        # redirecting standard output to a StringIO object.
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # calling summary function.
        db.summary()

        # restoring standard output.
        sys.stdout = sys.__stdout__

        # Get the captured output as a string.
        summary_output = captured_output.getvalue()

        # Define the expected output
        _db_c_count = len(db._cache)
        _db_c_limit = db._cache_size
        _db_size_hr = human_readible_size(sys.getsizeof(db.name))
        _db_c_size_hr = human_readible_size(sys.getsizeof(db._cache))

        expected_output = (f"Cache: \n"
                         + f"  length: {_db_c_count}\n"
                         + f"  limit: {_db_c_limit} \n"
                         + f"  size: {_db_c_size_hr}\n" 
                         + f"Database: \n"
                         + f"  name: {db_name}\n"
                         + f"  size: {_db_size_hr}\n\n")
        
        self.assertEqual(summary_output, expected_output)
        db.close_db()

    def test_remove_reserved_keys(self):
        db = Database(db_path="test.db")

        # trying to remove a reserved key.
        db.delete_value("RESERVED_KEY")
        db.delete_value("PROJECTS_LIST")

        # Check if the reserved keys still exist in the database
        reserved_keys_value = db.get_value("RESERVED_KEYS")

        # Assert that the reserved keys still exist
        self.assertIsNotNone(reserved_keys_value)

        db.close_db()

    def test_update_cache_size(self):
        db = Database(db_path="test.db")

        # updating cache size to 10.
        new_cache_size = 5
        db.update_cache_size(new_cache_size)

        # putting value in database does not change the cache.
        for i in range(new_cache_size + 5):
            db.set_value(f"key{i}", f"value{i}")

        self.assertEqual(len(Database._cache), 0)

        # access enough data to fill the cache.
        for i in range(new_cache_size):
            _ = db.get_value(f"key{i}")
        
        # now the cache is full.
        self.assertEqual(len(Database._cache), new_cache_size)

        # Access one more item, triggering cache eviction
        _ = db.get_value(f"key{new_cache_size}")

        # Check that the cache size is still equal to the new limit
        self.assertEqual(len(Database._cache), new_cache_size)

        db.close_db()

    def test_set_value_database_unreachable(self):
        sqlite_dict_mock = MagicMock(side_effect=RuntimeError("Database unreachable"))

        with unittest.mock.patch("pycausalgps.database.SqliteDict",
                                  sqlite_dict_mock):
            db = Database(db_path="test.db")

            try:
                db.set_value("test_key", "test_value")
            except RuntimeError as e:
                self.fail(f"set_Value raised RuntimeError unexpectedly: {e}")

        db.close_db()


if __name__ == "__main__":
    unittest.main()