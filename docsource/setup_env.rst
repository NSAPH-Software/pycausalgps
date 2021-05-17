Setting Up Environment
======================

Creating a virtual environment can separate your package management steps 
from other projects. Here are the steps to set up a virtual environment 
and install the package. Please download the package from 
`pycausalgps <https://github.com/fasrc/pycausalgps>`_ repository.


Conda
-----
Here are the steps for running the code with anaconda virtual environment. 
Please make sure that you have
`anaconda <https://www.anaconda.com/products/individual>`_ installed on your
system. You can also install
`miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ which is the 
lightweight version of anaconda.

- Step 1: Create a virtual environment

.. code-block:: console

    $ conda create --name your_venv python=3.7

Type 'y' (yes) for popup questions.

- Step 2: Activate the virtual environment

.. code-block:: console

    $ conda activate your_venv

Install **PyCausalGPS** (Step 3 or Step 4)

- Step 3: Directly from PyPI
TBD

- Step 4: Install from Github package in developement mode
You can install the package in a developement mode according to the
following commands: 

Navigate to the downloaded ``pycausalgps`` folder and install the package requirements.

.. code-block:: console

    $ pip3 install -r requirements.txt

Navigate one folder up and install the package.

.. code-block:: console

    $ pip3 install -e pycausalgps

That's it. You should be able to *import pycausalgps* and use it. However, if you 
want to use the code inside `Jupyter Notebook <https://jupyter.org>`_ 
(or Jupyter Lab) please follow the next steps.

- Step 5: Register the kernel on Jupyter Notebook:

Make sure that you have activated the virtual environment (Please see Step 2). 
Next, install *ipykernel*:

.. code-block:: console

    $ conda install -c anaconda ipykernel

Type 'y' (yes) for popup questions.
The last step is adding the kernel into Jupyter Notebook. 


.. code-block:: console

    $ python3 -m ipykernel install --user --name your_venv_name

- Step 6: Open your notebook and start processing

In your working directory (any arbitrary directory that you work on your data), 
open terminal and fire up notebook:

.. code-block:: console

    $ jupyter notebook (or jupyter lab)

At the top right corner, there is a button labeled `New` key. Choose your 
recently created kernel (in this example: your_venv_name). Choosing a kernel 
will open a new tab that you can work on.

pipenv
------
TBD