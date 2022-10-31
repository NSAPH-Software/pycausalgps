"""
database.py
===========
The core module for the Database class.
"""

import sys

from itertools import islice
from sqlitedict import SqliteDict
from collections import OrderedDict

from pycausalgps.log import LOGGER
from pycausalgps.utils import human_readible_size

class Database:
    """ Database class
    
    Parameters:
    -----------

    :db_path: str
        Path to the database file.  

    The Database class takes care of the access to the database. It is a wrapper class for the sqlitedict package. The Database class has a in-memory cache to speed up the access to the recentely used values. User can update the cache size. 

    Examples:  
    ---------

    >>> from pycausalgps.database import Database  
    >>> db = Database("test.db")  
    >>> db.set_value("key1", "value1")  
    >>> db.get_value("key1")  
    'value1'

    >>> db.delete_value("key1")  
    >>> db.get_value("key1")  
    >>> db.update_cache_size(100)  
    >>> db.close_db()  

    """

    _cache = None
    _cache_size = None 

    def __init__(self, db_path):
        """ 
        Initializes the database.
        Inputs:
            | db_path: path to the database file.
        """

        self.name = db_path
        self._init_reserved_keys()

        if Database._cache is None:       
            Database._cache_size = 1000
            Database._cache = OrderedDict()
            LOGGER.debug(f"In memory cache has been initiated "+\
                         f"with size: {Database._cache_size}.")

    def __str__(self):
        return f"SQLitedict Database: {self.name}"

    def __repr__(self):
        return f"Database({self.name}, {Database._cache_size})"

    def set_value(self, key, value):
        """ 
        Sets the key and given value in the database. If the key exists,
        it will override the value. In that case, it will remove the key from
        the in-memory dictionary. It will be loaded again with the get_value
        command if needed.
        Inputs:
        | key: hash value (generated by the package)
        | value: Any python object
        """
        try:
            db = SqliteDict(self.name, autocommit=True)
            db[key] = value
            del Database._cache[key]
        except KeyError:
            LOGGER.debug(f"Tried to delete non-existing {key} on the cache.")
        except Exception:
            LOGGER.warning(f"Tried to set {key} on the database."
             "Something went wrong.")
        finally:
            db.commit()
            db.close() 

    def delete_value(self,key):
        """ Deletes the key, and its value from both in-memory dictionary and
        on-disk database. If the key is not found, simply ignores it.
        
        Inputs:
        | key: hash value (generated by the package)
        """
        try:
            db = SqliteDict(self.name, autocommit=True)
            reserved_keys = db["RESERVED_KEYS"]
            if key in reserved_keys:
                LOGGER.debug(f"An attempt to remove {key} was recorded.")
                print(f"{key} is a Reserved key. Reserved keys are not removable.")
                return
            
            del db[key]   
            try: 
                del Database._cache[key]
            except Exception as e:
                print(e)         
            LOGGER.debug(f"Value {key} is removed from database.")
        except KeyError:
            LOGGER.warning(f"Tried to delete '{key}' on the database."
             " No such keys on the database.")
        except Exception as e:
            print(e)
        finally:
            db.commit()
            db.close()

    def get_value(self, key):
        """ Returns the value in the following order:
        
        | 1) It will look for the value in the cache and return it, if not found
        | 2) will look for the value in the disk and return it, if not found
        | 3) will return None.
        
        Inputs:
        | key: hash value (generated by the package)

        Outputs:
        | If found, value, else returns None.         
        """
        value = None
        try:
            value = Database._cache[key]
            LOGGER.debug(f"Key: {key}. Value is loaded from the cache.")
            LOGGER.debug(f"In memory cache size: {len(Database._cache)}")
        except:
            LOGGER.debug(f"Key: {key}. Value is not found in the cache.")

        if not value:
            try:
                db = SqliteDict(self.name, autocommit=True)
                tmp = db[key]
                if len(Database._cache) >  Database._cache_size - 1:
                    Database._cache.popitem(last=False)
                    LOGGER.debug(f"cache size is more than limit"
                     f"{Database._cache_size}. An item removed, and new item added.")
                Database._cache[key] = tmp
                return tmp
            except Exception:
                LOGGER.debug(f"The requested key ({key}) is not in the"
                 " database. Returns None.")
                return None
            finally:
                db.commit()
                db.close()
        else:
            return value

    def cache_summary(self):
        """ 
        Returns the summary of the cache. It includes the length, limit and 
        human readible cache size.  
        """
        return f"Cache length: {len(Database._cache)}\n" \
               f"Cache limit: {Database._cache_size} \n" \
               f"Cache size: {human_readible_size(sys.getsizeof(Database._cache))}"
      


    def update_cache_size(self, new_size):
        """
        Updates the cache size. If the new size is smaller than the current size,
        it will remove the oldest items from the cache.
        Inputs:
        | new_size: new cache size (this is the number of items, not the size on the disk)
        """
        Database._cache_size = new_size
        if Database._cache_size > new_size:
            keys = list(islice(Database._cache, new_size))
            tmp_cache = OrderedDict()
            for key in keys:
                tmp_cache[key] = Database._cache[key]
            Database._cache = tmp_cache

    def close_db(self):
        """ Commits changes to the database, closes the database, clears the 
        cache.
        """
        Database._cache = None
        LOGGER.info(f"Database ({self.name}) is closed.")

    def _init_reserved_keys(self):
        """ Initializes the reserved keys in the database. """
        try:
            db = SqliteDict(self.name, autocommit=True)
            db["RESERVED_KEYS"] = ["RESERVED_KEYS", "PROJECTS_LIST"]
            LOGGER.debug(f"Reserved keys are initialized.")
        except Exception:
            LOGGER.debug(f"Reserved keys are already initialized.")
        finally:
            db.commit()
            db.close()


if __name__ == "__main__":
    db = Database("test.db")
    db.set_value("myvalue_1", [1,2,3,4,5])
    print(db.get_value("test"))
    db.delete_value("myvalue_1")
    print(db.get_value("myvalue_1"))
    db.cache_summary()
    db.update_cache_size(10)
    db.cache_summary()
    db.close_db()