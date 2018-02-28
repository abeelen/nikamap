NikaMap
=======
|pypi| |license| |wheels| |format| |pyversions| |rtd|

`nikamap` is a python package to manipulate data produced by the IDL NIKA2 pipeline.

.. code:: python
    from nikamap import NikaMap

    nm = NikaMap.read('map.fits')
    nm.plot()

Read the documentation on `readthedoc <http://nikamap.readthedocs.io>`_.

Features
--------

- reading, slicing, plotting
- match filtering, point source detection and photometry
- powspec spectra estimation
- bootstraping and jackknife

Requirements
------------
You need python 3.4 or later to run `nikamap`. You will also need `numpy`, `scipy`, `matplotlib`, `astropy>=2.0` and `photutils`.

Installation
------------
`nikamap` will soon be on `PyPI`, and thus you will be able to install it with

.. code:: bash

    pip install nikamap

In the meantime you can clone the git repository, and install it

.. code:: bash

    git clone https://gitlab.lam.fr/N2CLS/nikamap.git
    cd NikaMap
    pip install -e .


or directly install it from git

.. code:: bash

    pip install https://gitlab.lam.fr/N2CLS/nikamap.git


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


.. |codeclimate| image:: https://codeclimate.com/github/abeelen/nikamap/badges/gpa.svg
   :alt: Code Climate
   :target: https://codeclimate.com/github/abeelen/nikamap


.. |codehealth| image:: https://landscape.io/github/abeelen/nikamap/master/landscape.svg?style=flat
   :alt: Code Health
   :target: https://landscape.io/github/abeelen/nikamap/master


.. |sonarqube| image:: https://sonarcloud.io//api/badges/gate?key=nikamap&metric=code_smells
   :alt: SonarQube
   :target: https://sonarcloud.io/dashboard/index/nikamap


.. |rtd| image:: https://readthedocs.org/projects/nikamap/badge/?version=latest
    :alt: Read the doc
    :target: http://nikamap.readthedocs.io/

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/abeelen/nikamap?branch=master&svg=true
    :alt: AppVeoyr
    :target: https://ci.appveyor.com/project/abeelen/nikamap
