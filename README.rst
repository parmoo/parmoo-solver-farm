=======================================================================
ParMOO Solver Farm: Sample Simulation Optimization Problems and Solvers
=======================================================================

This repository contains samples of multiobjective simulation optimization
(MOSO) problems that have been solved with ParMOO.
All of the models and solver files needed to reproduce the results are
provided within each subdirectory.

*Note: this public repository is intended to be lightweight and contains
only publication-ready results. Use a private repository for development
of solvers and testing to avoid overfilling the git history.*

Setup and Installation
----------------------

To use or compare against any of the sample problem, clone the appropriate
directory and ``pip``-install the ``REQUIREMENTS`` file for that example.

.. code-block:: bash

    python3 -m pip install -r REQUIREMENTS

Instructions and Structure
--------------------------

For further instructions, including details on the specifics of each
problem, see the nested ``README.rst`` file for each of the internal
subdirectories.

Resources
---------

For more information on the ParMOO library

 * visit our GitHub_ page,
 * view our documentation on ReadTheDocs_

Or contact our e-mail/support

 * ``parmoo@mcs.anl.gov``

For potential contributors to the ParMOO Solver Farm, please see our CONTRIBUTING_ files.

Please check each individual example for its own license files.
If none is found, then you may assume that each directory carries
the ParMOO Solver Farm's default LICENSE_ file.

Citing
------

Many of these problems are associated with a publication.
If so, please cite the reference provided for the example that you
are running.

Otherwise, to cite the ParMOO library, use one of the following:

.. code-block:: bibtex

    @article{parmoo-joss,
        author={Chang, Tyler H. and Wild, Stefan M.},
        title={{ParMOO}: A {P}ython library for parallel multiobjective simulation optimization},
        year = {2023},
        journal = {Journal of Open Source Software},
        volume = {8},
        number = {82},
        pages = {4468},
        doi = {10.21105/joss.04468}
    }

    @techreport{parmoo-docs,
        title       = {{ParMOO}: {P}ython library for parallel multiobjective simulation optimization},
        author      = {Chang, Tyler H. and Wild, Stefan M. and Dickinson, Hyrum},
        institution = {Argonne National Laboratory},
        number      = {Version 0.4.0},
        year        = {2024},
        url         = {https://parmoo.readthedocs.io/en/latest}
    }


.. _CONTRIBUTING: https://github.com/parmoo/parmoo-solver-farm/blob/main/CONTRIBUTING.rst
.. _GitHub: https://github.com/parmoo/parmoo
.. _LICENSE: https://github.com/parmoo/parmoo-solver-farm/blob/main/LICENSE
.. _ReadTheDocs: https://parmoo.readthedocs.org
