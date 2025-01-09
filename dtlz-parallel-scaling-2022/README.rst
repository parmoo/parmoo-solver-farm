===============================================================
Timing ParMOO + libEnsemble parallel runs on DTLZ2 test problem
===============================================================

This directory contains several scripts for evaluating the parallel
performance of ParMOO v0.2.2 with libEnsemble.

Note that real-world performance will vary based on simulation expense,
problem type, choice of solver, available hardware, and many other
factors.

In this example, we use the DTLZ2 test problem from the multiobjective
test problem library DTLZ by Deb et al. (2005). In order to simulate
problem expense, we have created artificial simulation runtimes that
are uniformly distributed in the range [1, 3] seconds using the Python
``time.sleep(t)`` command.

These scripts correspond to the results described in Section 6 of
*Chang and Wild (2023)
Designing a Framework for Solving Multiobjective Simulation Optimization
Problems, Preprint ArXiv:2304.06881.*
Further details on the problem are also available in that paper.

This represents an older slower version of ParMOO, note that performance
has been greatly improved in newer versions.

Setup and Installation
----------------------

The requirements for this directory are:

 - parmoo_,
 - libensemble_, and
 - pymoo_ (only for calculating hypervolumes and for comparison against
   NSGA-II).

To try running these solvers yourself, clone this directory
and install the requirements, or install the included ``REQUIREMENTS.txt``
file.

.. code-block:: bash

    python3 -m pip install -r REQUIREMENTS.txt

Note that several major libraries in ``REQUIREMENTS.txt`` have been pinned to
very specific versions.  Since these runs were used for timing, any change in
the version numbers could result in different results than those that we have
previously recorded.

The scripts in this directory will attempt to store results in the
subdirectory ``parmoo-dtlz2``. You must create this directory yourself.

.. code-block:: bash

    mkdir -p parmoo-dtlz2

Alternatively, the ``run_timings.sh`` script will create them for you.

Instructions and Structure
--------------------------

This particular directory contains three Python files.

 - ``parmoo_solve_dtlz2_serial.py`` provides scripts for solving the
   DTLZ2 problem with ParMOO, with serial simulation evaluations;
 - ``parmoo_solve_dtlz2_parallel.py`` provides scripts for solving the
   DTLZ2 problem with ParMOO, with parallel simulation evaluations; and
 - ``pymoo_solve_dtlz2_serial.py`` provides scripts for solving the DTLZ2
   problem with pymoo, with serial simulation evaluations.

And one additional run file:

 - ``run_timings.sh``, which can be used to reproduce our experiments. It
   will create and store results in the new folders ``parmoo-dtlz2`` and
   ``pymoo-dtlz2``.

.. code-block:: bash

    ./run_timings.sh

To most closely reproduce our results, make sure you are using
Python v3.8.3, and the library versions specified in the ``REQUIREMENTS.txt``
file. The results of our experiments are saved in:

 - ``parmoo-dtlz2-results-2024/size{SIZE}_seed{SEED}.csv``,
   where ``SIZE`` and ``SEED`` represent the batch size and random seed
   number from each experiment, respectively.
   Within each file, each line contains values: ``walltime, hypervolume``,
   and the lines are ordered in increasing number of parallel threads,
   speficially 1, 2, 4, 8.

Resources
---------

For more reading on the ParMOO library and its other options

 * visit the parmoo_GitHub_page_, or
 * view the parmoo_readthedocs_page_

Citing this work and ParMOO
---------------------------

To cite this work, use the following:

.. code-block:: bibtex

    @techreport{parmoo-design,
        title   = {Designing a Framework for Solving Multiobjective Simulation Optimization Problems},
        author  = {Chang, Tyler H. and Wild, Stefan M.},
        year    = {2023},
        note    = {Preprint \url{https://arxiv.org/abs/2304.06881}}
    }

Use the following to credit or read more about the DTLZ test suite and the
DTLZ2 problem:

.. code-block:: bibtex

    @incollection{dtlz,
        title       = {Scalable test problems for evolutionary multiobjective optimization},
        author      = {Deb, Kalyanmoy and Thiele, Lothar and Laumanns, Marco and Zitzler, Eckart},
        booktitle   = {Evolutionary Multiobjective Optimization, Theoretical Advances and Applications},
        chapter     = {6},
        editors     = {Abraham, Jain, and Goldberg},
        year        = {2005},
        address     = {London, UK},
        publisher   = {Springer}
    }

To specifically cite the ParMOO library, use one of the following:

.. code-block:: bibtex

    @article{parmoo-joss,
        author={Chang, Tyler H. and Wild, Stefan M.},
        title={{ParMOO}: A {P}ython library for parallel multiobjective simulation optimization},
        journal = {Journal of Open Source Software},
        volume = {8},
        number = {82},
        pages = {4468},
        year = {2023},
        doi = {10.21105/joss.04468}
    }

    @techreport{parmoo-docs,
        title       = {{ParMOO}: {P}ython library for parallel multiobjective simulation optimization},
        author      = {Chang, Tyler H. and Wild, Stefan M. and Dickinson, Hyrum},
        institution = {Argonne National Laboratory},
        number      = {Version 0.2.2},
        year        = {2023},
        url         = {https://parmoo.readthedocs.io/en/latest}
    }


.. _libensemble: https://github.com/libensemble/libensemble
.. _parmoo: https://github.com/parmoo/parmoo
.. _parmoo_github_page: https://github.com/parmoo/parmoo
.. _parmoo_readthedocs_page: https://parmoo.readthedocs.org
.. _pymoo: https://pymoo.org
