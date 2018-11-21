
Welcome to NikaMap's documentation!
====================================

:mod:`nikamap` is a python module to manipulate data produced by the IDL NIKA2 pipeline.

.. code:: python

  from nikamap import NikaMap

  nm = NikaMap.read('map.fits', band='1mm')
  nm.plot()

or alternatively

.. code:: python

  from nikamap import NikaFits

  data = NikaFits.read('map.fits')
  data['1mm'].plot()


Features
--------

- reading, slicing, plotting
- match filtering, point source detection and photometry
- power spectra estimation
- bootstraping and jackknife


Please refer to the `README file
<https://gitlab.lam.fr/N2CLS/nikamap/blob/master/README.rst>`_ in the Git repository.


Contents:

.. toctree::
   :maxdepth: 2

   auto_examples/index
   api
