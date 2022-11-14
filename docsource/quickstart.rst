Quickstart
==========

Installing the Package
----------------------

1. Install the package using pip::

   *TBD*

2. Install the package from Github::

   *TBD*


The Whole Game
--------------

A processing is described under a project. Although users can use classes individually, we recommend using the project controller because it is the only way to generate a flow of processing and keep track of the results in the database. Each processing session requires an instance of project controller. 

.. code-block:: python

    >>> from pycausalgps.project_controller import ProjectController
    >>> pc = ProjectController(db_path = 'test_database.sqlite')   


The project controller manages the projects in the database. The project controller will connect to the database if the provided database already exists. Otherwise, it will create a new database. We assume that the database did not exist.



