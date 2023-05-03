Quickstart
==========

Installing the Package
----------------------

1. Install the package using pip::

.. code-block:: bash

    $ pip install pycausalgps

See the `Setting Up Environment <installation.html>`_ for more details.


Generating Synthetic Data
-------------------------

The package provides a function to generate synthetic data. 

.. code-block:: python

    >>> from pycausalgps.base.utils import generate_syn_pop
    >>> data = generate_syn_pop(sample_size=1000, 
                                seed_val=456, 
                                outcome_sd=0.25, 
                                gps_spec=1, 
                                cova_spec=2)

