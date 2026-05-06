
Welcome to NikaMap's documentation!
====================================

:mod:`nikamap` is a Python package built around :class:`~nikamap.ContMap`, a
general-purpose class for processing and analysing continuum maps from
radio/millimetre observatories.  :class:`~nikamap.ContMap` extends
:class:`~astropy.nddata.NDDataArray` with beam handling, match-filtering,
point-source detection and photometry, and stacking, making it a versatile
tool for any single-dish continuum survey.

For NIKA2 data specifically, :mod:`nikamap` provides :class:`~nikamap.NikaMap`
and :class:`~nikamap.NikaFits`, thin subclasses of :class:`~nikamap.ContMap`
that add first-class support for the FITS data products of both the IDL and
PIIC NIKA2 reduction pipelines, including multi-band files and scan-level
jackknife realisations.  All :class:`~nikamap.ContMap` features are
naturally inherited by these NIKA2-specific classes.

Quick start with NIKA2 data
---------------------------

.. code:: python

   from nikamap import NikaMap

   nm = NikaMap.read('map.fits', band='1mm')
   nm.plot()

or using the multi-band container:

.. code:: python

   from nikamap import NikaFits

   data = NikaFits.read('map.fits')
   data['1mm'].plot()

Quick start with generic continuum data
---------------------------------------

Any continuum map can be loaded directly into :class:`~nikamap.ContMap`:

.. code:: python

   from nikamap import ContMap, ContBeam
   import astropy.units as u

   cm = ContMap.read('mymap.fits')
   cm.beam = ContBeam(major=18 * u.arcsec, pixscale=3 * u.arcsec)
   cm.plot()


Features
--------

- Reading, slicing and plotting NIKA2 and generic continuum FITS maps
- Beam / PSF description via :class:`~nikamap.ContBeam`
- Match filtering, point source detection (:meth:`~nikamap.ContMap.detect_sources`)
  and PSF photometry (:meth:`~nikamap.ContMap.phot_sources`)
- Power spectra estimation
- **Cutout stacking** — weighted average of stamps centred on a source catalogue
  (:meth:`~nikamap.ContMap.stack`)
- **Simultaneous stacking (simstack)** — linear regression over source populations
  (:meth:`~nikamap.ContMap.simstack`), following `Viero et al. 2013
  <https://arxiv.org/abs/1304.0446>`_
- Bootstrap and jackknife noise realisations


Please refer to the `README file
<https://gitlab.lam.fr/N2CLS/nikamap/blob/master/README.rst>`_ in the Git
repository for installation instructions.


Contents:

.. toctree::
   :maxdepth: 2

   auto_examples/index
   contmap
   api
