Development Environment
=======================

The development environment requires some extra packages to support documentation. `pycausalgps` package can be installed in any environment with Python 3.7 or higher. Please download the package from the `pycausalgps <https://github.com/fasrc/pycausalgps>`_ Github repository.  Anaconda virtual environment is highly recommended. Please make sure that you have `anaconda <https://docs.anaconda.com/anaconda/install/index.html>`_ installed on your system. You can also install `miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, which is the lightweight version of anaconda. 

Setting up a new environment
----------------------------

If you have not set up pycausalgps development environment, you can use the following command to create a new environment and install dependencies.


.. code-block:: console

    $ conda env create --name your_env_name --file environment_dev.yml 


Updating an existing environment
--------------------------------

If you want to make sure that the environment is the latest version, you can update the environment using the following command:

.. code-block:: console

    $ conda env update --name your_env_name --file environment_dev.yml 


Installing pycausalgps
----------------------

Navigate into the package folder and run the following command:

.. code-block:: console

    $ pip3 install -e .

``-e`` flag installs the package in the development mode. As a result, you do not need to reintall the package by chaning the code.


Registering the kernel
----------------------

If you plan on testing the package using Jupyter Notebook, you need to register the package using the following command:

.. code-block:: console

    $ python3 -m ipykernel install --user --name your_env_name   

You can remove the kernle from registered kernel list using:

.. code-block:: console

    $ jupyter kernelspec remove 'your_env_name'   

Use the following command to completely remove a conda env from your system:

.. code-block:: console

    $ conda env remove -n your_env_name    


Updating the environment recipe
-------------------------------

If you added a new feature to the package that is unavailable in the current environment, you need to update the environment recipe, while environment is activated, using the following command:


.. code-block:: console

    $ conda env export > environment_dev.yml   

Make sure to manually remove `name` and `prefix` sections as well as `nsaphx` from the dependencies section in the `.yml` file. 