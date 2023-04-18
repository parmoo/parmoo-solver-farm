=============================================================
Identifying CFR Material Manufacturing Conditions with ParMOO
=============================================================

This directory contains several scripts for calibrating a continuous flow
reactor (CFR) in order to maximize the production TFMC at high temperatures,
based on a model of of the expected products and byproducts, which was fit
using previously-collected experimental data.

The significance of the problem and collection of the dataset used to train
our model is fully described in
*Chang et al. (2023) ICLR 23, Workshop on ML4Materials*.
To reduce dependencies, the original ``sklearn`` model has been ported into
basic ``python`` and ``sklearn`` is not a dependency for this project.

These scripts correspond to the results described in Section 5 of
*Chang and Wild (2023)
Designing a Framework for Solving Multiobjective Simulation Optimization
Problems, Preprint ArXiv:2304.06881.*
Further details on the problem are also available in that paper.

Setup and Installation
----------------------

The requirements for this directory are:

 - parmoo_ and
 - plotly_.

To try running these solvers yourself, clone this directory
and install the requirements, or install the included ``REQUIREMENTS.txt``
file.

.. code-block:: bash

    python3 -m pip install -r REQUIREMENTS.txt

Then, to verify the installation, try running the faster test version of
the problem in ``parmoo_cfr_test.py``, to verify that ParMOO is working
correctly.

.. code-block:: bash

    python3 parmoo_cfr_test.py

Instructions and Structure
--------------------------

This particular directory contains three Python files.

 - ``cfr_model.py`` defines a model of the product and byproduct NMR integrals
   and provides several ParMOO compatible objective and constraint functions;
 - ``parmoo_cfr_structured_solver.py`` provides scripts for solving the
   problem with ParMOO, while exploiting the heterogeneous objective structure
   of the problem; and
 - ``parmoo_cfr_blackbox_solver.py`` provides an implementation where the
   heterogeneous problem structure is not available to ParMOO, thus allowing
   for comparison between exploiting vs. not exploiting the problem structure.
 - ``parmoo_cfr_test.py`` is a test case, to make sure that ParMOO and the
   model definitions are working.

If ``name.py`` is one of

 - ``parmoo_cfr_structured_solver.py``
 - ``parmoo_cfr_blackbox_solver.py``

then you can reproduce our results by using the command:

.. code-block:: bash

    python3 name.py [--iseed I]

or, if you are working on a Linux or MacOS system, run all of the experiments
with our ``run_all.sh`` script:

.. code-block:: bash

    ./run_all.sh

where ``I`` is the random seed, which can be fixed to any integer for
reproducability (when omitted, it is assigned by the system clock).

In the associated paper, we used the seed values ``I = 0, 1, 2, 3, 4``.

After running, the complete function-value database is saved to a file
``parmoo_cfr_structured_results_seed_I.csv`` or
``parmoo_cfr_blackbox_results_seed_I.csv``, depending on the method run
where ``I`` is as defined above.

To recreate the plots in the paper, run either of the plotting scripts in
the ``./plots`` subdirectory.

Resources
---------

For more reading on the ParMOO library and its other options

 * visit the parmoo_GitHub_page_, or
 * view the parmoo_readthedocs_page_

To read about the CFR product/byproduct models and how the data was collected,
see

  https://openreview.net/forum?id=8KJS7RPjMqG

Citing this work and ParMOO
---------------------------

To cite this work, use the following:

.. code-block:: bibtex

    @techreport{parmoo-design,
        title   = {Designing a Framework for Solving Multiobjective Simulation Optimization Problems},
        author  = {Chang, Tyler H. and Wild, Stefan M.},
        year    = {2023},
        note    = {Preprint: \url{https://arxiv.org/abs/2304.06881}}
    }

If you use our pre-trained CFR product/byproduct models, consider also citing
the data collection:

.. code-block:: bibtex

    @inproceedings{chang2023,
        author  = {Chang, Tyler H. and Elias, Jakob R. and Wild, Stefan M. and Chaudhuri, Santanu and Libera, Joseph A.},
        title   = {A framework for fully autonomous design of materials via multiobjective optimization and active learning: challenges and next steps},
        year    = {2023},
        booktitle = {ICLR 2023, Worshop on Machine Learning for Materials (ML4Materials)}, 
        url     = {https://openreview.net/forum?id=8KJS7RPjMqG}
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
        number      = {Version 0.2.1},
        year        = {2023},
        url         = {https://parmoo.readthedocs.io/en/latest}
    }


.. _parmoo: https://github.com/parmoo/parmoo
.. _parmoo_github_page: https://github.com/parmoo/parmoo
.. _parmoo_readthedocs_page: https://parmoo.readthedocs.org
.. _plotly: https://plotly.com/python/
