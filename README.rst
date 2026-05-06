NikaMap
=======
|pypi| |license| |wheels| |format| |pyversions| |rtd|

`nikamap` is a Python package built around `ContMap`, a general-purpose class
for processing and analysing continuum maps from radio/millimetre
observatories.  It provides beam handling, match-filtering, point-source
detection and photometry, and cutout / simultaneous stacking.

For NIKA2 users, `nikamap` offers first-class support for the FITS data
products of both the **IDL** and **PIIC** NIKA2 reduction pipelines through
`NikaMap` and `NikaFits`, thin subclasses of `ContMap` that handle
multi-band files and scan-level jackknife realisations.

Quick start with generic continuum data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from nikamap import ContMap

    cm = ContMap.read('mymap.fits')
    cm.plot()

Quick start with NIKA2 data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from nikamap import NikaMap

    nm = NikaMap.read('map.fits', band='1mm')
    nm.plot()

or using the multi-band container:

.. code:: python

    from nikamap import NikaFits

    data = NikaFits.read('map.fits')
    data['1mm'].plot()

Read the documentation on `readthedoc <http://nikamap.readthedocs.io>`_.

Features
--------

- reading, slicing, plotting
- match filtering, point source detection and photometry
- powspec spectra estimation
- bootstraping and jackknife

Requirements
------------
You need python 3.10 or later to run `nikamap`. You will also need `numpy`, `scipy`, `matplotlib`, `astropy>=2.0` and `photutils>=1.13`. The master branch is based on newert photutils API>=1.8, while the `tag:0.6` branch follow the old API and python up to 3.8

Installation
------------
`nikamap` can be installed with 

.. code:: bash

    pip install nikamap

But if you prefer to work on the master branch, you can clone the git repository, and install it

.. code:: bash

    git clone https://gitlab.lam.fr/N2CLS/nikamap.git
    cd nikamap
    pip install -e .


or directly install it from git

.. code:: bash

    pip install git+https://gitlab.lam.fr/N2CLS/nikamap.git


License
-------

This project is licensed under the MIT license.

|build-travis| |appveyor| |codeclimate| |codehealth| |sonarqube|


.. |pypi| image:: https://img.shields.io/pypi/v/nikamap.svg?maxAge=2592000
    :alt: Latest Version
    :target: https://pypi.python.org/pypi/nikamap


.. |license| image:: https://img.shields.io/pypi/l/nikamap.svg?maxAge=2592000
    :alt: License


.. |wheels| image:: https://img.shields.io/pypi/wheel/nikamap.svg?maxAge=2592000
   :alt: Wheels


.. |format| image:: https://img.shields.io/pypi/format/nikamap.svg?maxAge=2592000
   :alt: Format


.. |pyversions| image:: https://img.shields.io/pypi/pyversions/nikamap.svg?maxAge=25920001;5002;0c
   :alt: pyversions


.. |build-travis| image:: https://travis-ci.org/abeelen/nikamap.svg?branch=master
    :alt: Travis Master Build
    :target: https://travis-ci.org/abeelen/nikamap


.. |codeclimate| image:: https://api.codeclimate.com/v1/badges/708805538fddec5ef127/maintainability
   :target: https://codeclimate.com/github/abeelen/nikamap/maintainability
   :alt: Maintainability


.. |codehealth| image:: https://landscape.io/github/abeelen/nikamap/master/landscape.svg?style=flat
   :alt: Code Health
   :target: https://landscape.io/github/abeelen/nikamap/master


.. |sonarqube| image:: https://sonarcloud.io/api/project_badges/measure?project=nikamap&metric=alert_status
   :alt: SonarQube
   :target: https://sonarcloud.io/dashboard/index/nikamap


.. |rtd| image:: https://readthedocs.org/projects/nikamap/badge/?version=latest
    :alt: Read the doc
    :target: http://nikamap.readthedocs.io/

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/abeelen/nikamap?branch=master&svg=true
    :alt: AppVeoyr
    :target: https://ci.appveyor.com/project/abeelen/nikamap

Contributing
------------

See ``CONTRIBUTING.rst`` for the development setup, test commands, documentation build,
and contribution expectations.
