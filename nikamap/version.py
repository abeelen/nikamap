from __future__ import absolute_import, division, print_function

from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 6
_version_micro = ""  # use '' for first of series, number for 1 and above
_version_extra = "dev0"
#_version_extra = ""  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = ".".join(map(str, _ver))

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Scientific/Engineering",
]

# Description should be a one-liner:
description = "nikamap: a package to manipulate data produced by the IDL NIKA2 pipeline"
# Long description will go up on the pypi page
long_description = """

Nikamap
========
Nikamap is a  python package to manipulate data produced by the IDL NIKA2 pipeline.

To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/abeelen/nikamap/blob/master/README.md

License
=======
``nikamap`` is licensed under the terms of the MIT license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2017--, Alexandre Beelen, Laboratoire d'Astrophysique
de Marseille.

"""

NAME = "nikamap"
MAINTAINER = "Alexandre Beelen"
MAINTAINER_EMAIL = "alexandre.beelen@lam.fr"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/abeelen/nikamap"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Alexandre Beelen"
AUTHOR_EMAIL = "alexandre.beelen@lam.fr"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {"nikamap": [pjoin("data", "*")]}
INSTALL_REQUIRES = [
    "numpy",
    "scipy",
    "matplotlib",
    "astropy>5.2",
    "photutils<=1.8",
    "scikit_image",
    "powspec>=0.2",
]
SETUP_REQUIRES = ["pytest-runner"]
EXTRA_REQUIRE = {"stacking": ['reproject']}
TESTS_REQUIRE = ["pytest", "pytest-cov", "pytest-mpl", "reproject"]
