============================================
Calibrating the Fayans EDF Model with ParMOO
============================================

This directory contains several scripts for calibrating a neural-network
replica of the Fayans Energy Density Functional (EDF) model residuals based
on experimental data.

The significance of the problem and collection of the dataset used to train
our ``keras`` residual model is fully described in
*Bollapragada et al. (2020)
Journal of Physics G: Nuclear and Particle Physics 48(2):024001*.
For compatibility reasons, the ``keras`` residual model has been converted
into a ``torch`` model.
This means that ``pytorch``, not ``tensorflow``, is required for usage
of this model.

These scripts correspond to the results described in Section 4 of
*Chang and Wild (2023)
Designing a Framework for Solving Multiobjective Simulation Optimization
Problems, under review.*
Further details on the problem are also available in that paper.

Setup and Installation
----------------------

The requirements for this directory are:

 - parmoo_,
 - libensemble_,
 - plotly_, and
 - torch_.

To try running these solvers yourself, clone this directory
and install the requirements, or install the included ``REQUIREMENTS.txt``
file.

.. code-block:: bash

    python3 -m pip install -r REQUIREMENTS.txt

Then, to verify the installation, try running the faster test version of
the problem in ``parmoo_fayans_test.py``, to verify that ParMOO,
libEnsemble, and PyTorch are all working together correctly.

.. code-block:: bash

    python3 parmoo_fayans_test.py --comms local --nworkers 4

Instructions and Structure
--------------------------

This particular directory contains three Python files.

 - ``fayans_model.py`` defines a model of the residuals of the Fayans EDF
   calibration problem and provides several ParMOO compatible objective
   and constraint functions;
 - ``parmoo_fayans_structured_solver.py`` provides scripts for solving the
   problem with ParMOO, while exploiting the sum-of-squares structure in
   how the (blackbox) residuals are used to define the problem; and
 - ``parmoo_fayans_blackbox_solver.py`` provides an implementation where
   the problem structure is not available to ParMOO, thus allowing for
   comparison between exploiting vs. not exploiting the problem structure.
 - ``parmoo_fayans_test.py`` is a test case, to make sure that ParMOO,
   libEnsemble, and pytorch are working together.

An additional file:

 - ``fayans_weights.h5`` contains the weights for the pre-trained
   ``torch`` model of the  Fayans EDF residuals.

If ``name.py`` is one of

 - ``parmoo_fayans_structured_solver.py``
 - ``parmoo_fayans_blackbox_solver.py``

then you can reproduce our results by using the command:

.. code-block:: bash

    python3 name.py --comms C --nworkers N [--iseed I]

or, if you are working on a Linux or MacOS system, run all of th experiments
with our ``run_all.sh`` script:

.. code-block:: bash

    ./run_all.sh

where ``C`` is the communication protocol (``local`` or ``tcp``);
``N`` is the number of libE workers (i.e., number of concurrent simulation
evaluations); and
``I`` is the random seed, which can be fixed to any integer for
reproducability (when omitted, it is assigned by the system clock).

In the associated paper, we used the seed values ``I = 0, 1, 2, 3, 4``.

After running, the complete function-value database is saved to a file
``parmoo_fayans_structured_results_seed_I.csv`` or
``parmoo_fayans_blackbox_results_seed_I.csv``, depending on the method run
where ``I`` is as defined above.

To recreate the plots in the paper, run either of the plotting scripts in
the ``./plots`` subdirectory.

Resources
---------

For more reading on the ParMOO library and its other options

 * visit the parmoo_GitHub_page_, or
 * view the parmoo_readthedocs_page_

To read about the Fayans EDF model and how the data was collected, see

  https://arxiv.org/abs/2010.05668

Citing this work and ParMOO
---------------------------

To cite this work, use the following:

.. code-block:: bibtex

    @techreport{parmoo-design,
        title   = {Designing a Framework for Solving Multiobjective Simulation Optimization Problems},
        author  = {Chang, Tyler H. and Wild, Stefan M.},
        year    = {2023},
        note    = {Under review, preprint \url{https://arxiv.org/abs/2304.06881}}
    }

If you use our pre-trained Fayans EDF residual model, consider also citing
the authors who collected the Fayans EDF model's training data:

.. code-block:: bibtex

    @article{bollapragada2020,
        author  = {Bollapragada, Raghu and Menickelly, Matt and Nazarewicz, Witold and O'Neal, Jared and Reinhard, Paul-Gerhard and Wild, Stefan M.},
        title   = {Optimization and supervised machine learning methods for fitting numerical physics models without derivatives},
        year    = {2020},
        journal = {Journal of Physics G: Nuclear and Particle Physics}, 
        volume  = {48},
        number  = {2}, 
        pages   = {024001},
        doi     = {10.1088/1361-6471/abd009}
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


.. _libensemble: https://github.com/libensemble/libensemble
.. _parmoo: https://github.com/parmoo/parmoo
.. _parmoo_github_page: https://github.com/parmoo/parmoo
.. _parmoo_readthedocs_page: https://parmoo.readthedocs.org
.. _plotly: https://plotly.com/python/
.. _torch: https://pytorch.org/
