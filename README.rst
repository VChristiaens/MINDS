MINDS
=====


.. image:: https://img.shields.io/badge/Python-3.9%2C%203.10%2C%203.11-brightgreen.svg
    :target: https://github.com/VChristiaens/MINDS
.. image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/VChristiaens/MINDS/blob/master/LICENSE
::


This repository contains the hybrid pipeline based on the `jwst` pipeline and leveraging routines from the VIP package that was developed by the MINDS - MIRI INfrared Disk Survey - GTO team for the reduction of MIRI-MRS data.


Documentation
-------------
3 PDF files are available:

- Guidelines.pdf: provides guidelines to use the notebook, and answers to FAQ.
- Flow_charts.pdf: provides troubleshooting tips in the form of flow charts.
- Summary.pdf: summarises the structure of the pipeline, and the results obtained with different options on a given dataset.


TL;DR setup guide
-----------------
.. code-block:: bash

    $ pip install -r requirements.txt

Then launch and run the MINDS_reduction.ipynb notebook, after adapting the input path to the raw data.


Installation and dependencies
-----------------------------
The benefits of using a Python package manager (distribution), such as
(ana)conda, are many. Mainly, it brings easy and robust package
management and avoids messing up with your system's default python. 
We recommend using
`Miniconda <https://conda.io/miniconda>`_.

Before installing the package, it is **highly recommended to create a dedicated
conda environment** to not mess up with the package versions in your base
environment. This can be done easily with (replace ``vipenv`` by the name you want
for your environment):

.. code-block:: bash

  $ conda create -n minds_env python=3.10 jupyter

Then, to activate it (assuming you named it as above):

.. code-block:: bash

  $ conda activate minds_env


The pipeline depends on two major packages: jwst and vip_hci, which both come
with their own set of dependencies from the Python ecosystem, such as
``numpy``, ``scipy``, ``matplotlib``, ``pandas``, ``astropy``, ``scikit-learn``,
``scikit-image``, ``photutils`` and others. The most convenient way to install 
all required dependencies is simply, once in the environment:

.. code-block:: bash

  $ pip install -r requirements.txt

In most cases, you should not select the option to use point-source specific reference files (see flow charts).
However, if you do, you will first have to download them here: https://dox.uliege.be/index.php/s/h4MM95IqFt8Gvce
Place the psff_ref folder in the same directory as the Jupyter notebook. 


Usage
-----

After downloading locally the raw data into a folder called "Stage0", it is only a matter of adapting a couple of parameters in the second cell of the Jupyter notebook and let it run entirely (no need to modify subsequent cells).

The point of preserving the notebook is added flexibility, visibility, and easier debugging.

Detailed instructions are provided in the Guidelines pdf.


Attribution
-----------

Christiaens, Samland, Gasman, Temmink & Perotti, subm. to ASCL (DOI link to be provided soon).

We sincerely thank David Law and Patrick Kavanagh, whose notebook and script shared with us allowed us to kickstart this projet. We also thank Yannis Argyriou for very useful feedback throughout the development of this hybrid pipeline.

We also kindly to the following publications:
- `Bushouse et al. (2023)<https://ui.adsabs.harvard.edu/abs/2023zndo...7795697B/abstract>`_ for the jwst pipeline (or similar reference, depending on the exact jwst pipeline version you use);
- `Gomez Gonzalez et al. (2017) <https://ui.adsabs.harvard.edu/abs/2017AJ....154....7G/abstract>`_ and `Christiaens et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023JOSS....8.4774C/abstract>`_ for VIP routines;
- `Gasman et al. (2023)<https://ui.adsabs.harvard.edu/abs/2023A%26A...673A.102G/abstract>`_ (if you use the psff=True option);
- Temmink et al. (2024) for continuum subtraction [link to be provided soon].
