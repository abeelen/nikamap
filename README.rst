NikaMap
=======
|pypi| |license| |wheels| |format| |pyversions| |rtd|

`nikamap` is a python package to manipulate data produced by the IDL NIKA2 pipeline.

.. code:: python

    from nikamap import NikaMap

    nm = NikaMap.read('map.fits', band='1mm')
    nm.plot()

or alternatively

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

We love contributions! cruft_openastro is open source,
built on open source, and we'd love to have you hang out in our community.

**Imposter syndrome disclaimer**: We want your help. No, really.

There may be a little voice inside your head that is telling you that you're not
ready to be an open source contributor; that your skills aren't nearly good
enough to contribute. What could you possibly offer a project like this one?

We assure you - the little voice in your head is wrong. If you can write code at
all, you can contribute code to open source. Contributing to open source
projects is a fantastic way to advance one's coding skills. Writing perfect code
isn't the measure of a good developer (that would disqualify all of us!); it's
trying to create something, making mistakes, and learning from those
mistakes. That's how we all improve, and we are happy to help others learn.

Being an open source contributor doesn't just mean writing code, either. You can
help out by writing documentation, tests, or even giving feedback about the
project (and yes - that includes giving feedback about the contribution
process). Some of these contributions may be the most valuable to the project as
a whole, because you're coming to the project with fresh eyes, so you can see
the errors and assumptions that seasoned contributors have glossed over.

Note: This disclaimer was originally written by
`Adrienne Lowe <https://github.com/adriennefriend>`_ for a
`PyCon talk <https://www.youtube.com/watch?v=6Uj746j9Heo>`_, and was adapted by
cruft_openastro based on its use in the README file for the
`MetPy project <https://github.com/Unidata/MetPy>`_.
