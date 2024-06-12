MINDS
=====

|Pythonv| |ASCL| |DOI| |License|

.. |Pythonv| image:: https://img.shields.io/badge/Python-3.10%2C%203.11-brightgreen.svg
            :target: https://github.com/VChristiaens/MINDS
.. |ASCL| image:: https://img.shields.io/badge/ascl-2403.007-blue.svg?colorB=262255
            :target: https://ascl.net/2403.007
.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.11121180.svg
            :target: https://zenodo.org/records/11121180
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
            :target: https://github.com/VChristiaens/MINDS/blob/master/LICENSE

This repository contains a hybrid pipeline based on the ``jwst`` pipeline and routines from the ``VIP`` package for the reduction of JWST MIRI-MRS data.
The pipeline was developed by the MINDS - MIRI mid-INfrared Disk Survey - GTO team in an attempt to compensate for some of the known weaknesses of the official jwst pipeline to improve the quality of spectrum extracted from MIRI-MRS data. This is done by leveraging the capabilities of VIP, another large data reduction package used in the field of high-contrast imaging.
The front end of the pipeline is a highly automated Jupyter notebook. Parameters are typically set in one cell at the beginning of the notebook, and the rest of the notebook can be run without any further modification. The Jupyter notebook format provides flexibility, enhanced visibility of intermediate and final results, more straightforward troubleshooting, and the possibility to easily incorporate additional codes by the user to further analyse or exploit their results.


Documentation
-------------
3 PDF files are available in the docs folder:

- Guidelines.pdf: provides guidelines to use the notebook, and answers to FAQ.
- Flow_charts.pdf: provides troubleshooting tips in the form of flow charts.
- Summary.pdf: summarises the structure of the pipeline, and the results obtained with different options on a given dataset.


TL;DR setup guide
-----------------
.. code-block:: bash

    pip install git+https://github.com/VChristiaens/MINDS.git

Then launch and run the MINDS_reduction.ipynb notebook, after adapting the input path to your data.


Installation and dependencies
-----------------------------
The benefits of using a Python package manager (distribution), such as
(ana)conda, are many. Mainly, it brings easy and robust package
management and avoids messing up with your system's default python. 
We recommend using
`Miniconda <https://conda.io/miniconda>`_.

Before installing the package, it is **highly recommended to create a dedicated
conda environment** to not mess up with the package versions in your base
environment. This can be done easily with (replace ``minds_env `` by the name you want
for your environment):

.. code-block:: bash

  conda create -n minds_env python=3.11

Then, to activate it (assuming you named it as above):

.. code-block:: bash

  conda activate minds_env


The pipeline depends on two major packages: ``jwst`` and ``vip_hci``, which both come
with their own set of dependencies from the Python ecosystem, such as
``numpy``, ``scipy``, ``matplotlib``, ``pandas``, ``astropy``, ``scikit-learn``,
``scikit-image``, ``photutils`` and others. There are 2 ways to get it all install at once.

1. The most convenient way is simply to run, once in the environment:

.. code-block:: bash

  pip install git+https://github.com/VChristiaens/MINDS.git

If later on, you want to upgrade to the latest MINDS pipeline version, it would be a matter of:

.. code-block:: bash

  pip install git+https://github.com/VChristiaens/MINDS.git --upgrade

2. Alternatively, if you also want the documentation, flow charts and example notebooks, clone the repository first and pip install locally:

.. code-block:: bash

  # cd where you want your local repository to be located
  git clone https://github.com/VChristiaens/MINDS.git
  # cd in your local repository
  pip install -e .

In the latter case, you can benefit from the latest changes made to the repository any time, with:

.. code-block:: bash

  git pull

In most cases, you should *not* select the option to use point-source specific reference files, as these are not up-to-date regarding temporal evolution of the photometric calibration (see flow charts).
However, if you do, you will first have to download them here: https://dox.uliege.be/index.php/s/h4MM95IqFt8Gvce
Place the psff_ref folder in the same directory as the Jupyter notebook.


Usage
-----

After downloading locally the raw data into a folder called "Stage0", it is only a matter of adapting a couple of parameters in the second cell of the Jupyter notebook and let it run entirely (no need to modify subsequent cells).

The point of preserving the notebook is added flexibility, visibility, and easier debugging.

Detailed instructions are provided in the Guidelines pdf.


Publications that made use of the MINDS pipeline:
-------------------------------------------------

- `Perotti et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023Natur.620..516P/abstract>`_
- `Schwarz et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024ApJ...962....8S/abstract>`_
- `Temmink et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024arXiv240313591T/abstract>`_


Attribution
-----------

If the pipeline is useful to your science, we kindly ask you to cite:

`Christiaens, Samland, Gasman, Temmink & Perotti (2024), Astrophysics Source CodeLibrary, ascl:2403.007 <https://ui.adsabs.harvard.edu/abs/2024ascl.soft03007C/abstract>`_

As well as some of the following publications:

- `Bushouse et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023zndo...7795697B/abstract>`_ for the jwst pipeline (or similar reference, depending on the exact jwst pipeline version you use);
- `Gomez Gonzalez et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154....7G/abstract>`_ and `Christiaens et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023JOSS....8.4774C/abstract>`_ for VIP routines used in the pipeline;
- `Gasman et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023A%26A...673A.102G/abstract>`_ if you set the option psff=True (i.e. point-source specific reference files);
- `Temmink et al. (2024) <https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.117T/abstract>`_ for continuum subtraction.

We sincerely thank David Law and Patrick Kavanagh, whose notebook and script shared with us allowed us to kickstart this projet. We also thank Yannis Argyriou for very useful feedback throughout the development of this hybrid pipeline.