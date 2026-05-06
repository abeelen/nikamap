.. _contmap_guide:

ContMap — continuum map analysis and stacking
=============================================

:class:`~nikamap.ContMap` is the central object in *nikamap*.  It inherits
from :class:`~astropy.nddata.NDDataArray` and therefore stores the map data,
associated uncertainty, mask, WCS, unit and arbitrary metadata in a single
object.  On top of the nddata interface, ``ContMap`` adds beam handling,
match-filtering, point-source detection and photometry, and the two stacking
methods described in this page.

For NIKA2 data specifically, :class:`~nikamap.NikaMap` is a thin subclass of
``ContMap`` that adds NIKA2-specific I/O (multi-band FITS files, scan-level
jackknife blocks, etc.).  **Everything documented here applies equally to
NikaMap.**


Loading a map
-------------

NIKA2 pipeline FITS file
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   from nikamap import NikaFits, NikaMap

   # Full multi-band container
   data = NikaFits.read('map.fits')
   nm1 = data['1mm']   # ContMap at 1 mm
   nm2 = data['2mm']   # ContMap at 2 mm

   # Or directly, single band
   nm = NikaMap.read('map.fits', band='1mm')

Generic continuum FITS file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any 2-D FITS image can be loaded.  The beam must be described either by
``BMAJ``/``BMIN``/``BPA`` FITS header keywords, or set manually:

.. code:: python

   from nikamap import ContMap, ContBeam
   import astropy.units as u

   cm = ContMap.read('mymap.fits')

   # Override beam if needed
   cm.beam = ContBeam(major=18 * u.arcsec, pixscale=3 * u.arcsec)

From arrays (programmatic construction)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

   import numpy as np
   import astropy.units as u
   from astropy.wcs import WCS
   from astropy.nddata import StdDevUncertainty
   from nikamap import ContMap, ContBeam

   shape = (256, 256)
   pixscale = 3 * u.arcsec

   wcs = WCS(naxis=2)
   wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
   wcs.wcs.cdelt = [-pixscale.to('deg').value, pixscale.to('deg').value]
   wcs.wcs.crval = [0.0, 0.0]
   wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']

   data = np.random.normal(0, 1e-3, size=shape)   # Jy/beam
   noise = np.full(shape, 1e-3)

   cm = ContMap(
       data,
       uncertainty=StdDevUncertainty(noise),
       wcs=wcs,
       unit=u.Jy / u.beam,
       beam=ContBeam(major=18 * u.arcsec, pixscale=pixscale),
   )


The beam — :class:`~nikamap.ContBeam`
--------------------------------------

:class:`~nikamap.ContBeam` describes the instrument PSF.  It is derived from
:class:`~astropy.convolution.Kernel2D` and supports the same convolution
interface.  By default a circular Gaussian is used, but an arbitrary array
can be provided.

.. code:: python

   from nikamap import ContBeam
   import astropy.units as u

   # Circular Gaussian at 12.5" FWHM, 2" pixels
   beam = ContBeam(major=12.5 * u.arcsec, pixscale=2 * u.arcsec)
   print(beam)   # ContBeam: BMAJ=… BMIN=… BPA=…

   # Elliptical beam
   beam_ell = ContBeam(major=12.5 * u.arcsec, minor=10 * u.arcsec,
                       pa=30 * u.deg, pixscale=2 * u.arcsec)

   # From beam area only
   beam_area = ContBeam(area=120 * u.arcsec**2, pixscale=2 * u.arcsec)

Key attributes: ``beam.major``, ``beam.minor``, ``beam.pa``, ``beam.sr``
(solid angle in steradians).


Map analysis
------------

Match filtering
^^^^^^^^^^^^^^^^

:meth:`~nikamap.ContMap.match_filter` convolves the map with a kernel
normalised to produce an optimal signal-to-noise filtered map (a.k.a.
matched filter):

.. code:: python

   mf = cm.match_filter(cm.beam)
   mf.plot_SNR(cbar=True)

Point source detection
^^^^^^^^^^^^^^^^^^^^^^^

:meth:`~nikamap.ContMap.detect_sources` runs :func:`~photutils.detection.find_peaks`
on the signal-to-noise map and stores the result in ``cm.sources``:

.. code:: python

   cm.detect_sources(threshold=3)
   print(cm.sources)

PSF photometry
^^^^^^^^^^^^^^^

:meth:`~nikamap.ContMap.phot_sources` performs PSF photometry at the
positions stored in ``cm.sources`` (or any user-supplied catalogue):

.. code:: python

   cm.detect_sources(threshold=3)
   cm.phot_sources()
   print(cm.sources['flux_fit', 'flux_unc'])


Stacking
--------

.. rubric:: Cutout stacking — :meth:`~nikamap.ContMap.stack`

The simplest stacking mode extracts a stamp of angular size ``size`` centred
on each input coordinate and computes the inverse-variance weighted mean.

.. code:: python

   from astropy.coordinates import SkyCoord
   import astropy.units as u

   # Prior catalogue (e.g. from an optical/IR survey)
   coords = SkyCoord(ra=..., dec=..., unit='deg')

   stacked = cm.stack(coords, size=60 * u.arcsec)
   stacked.plot(cbar=True)

The return value is a new :class:`~nikamap.ContMap` whose uncertainty is the
inverse-variance weight map of the stack.

**Bootstrap uncertainty estimate** — pass ``n_bootstrap`` to replace the
analytic weight-based uncertainty with a bootstrap standard deviation:

.. code:: python

   stacked_bs = cm.stack(coords, size=60 * u.arcsec, n_bootstrap=200)

**Reprojection backend** — for large pixel scales or maps with significant
astrometric distortion, use ``method='reproject'`` to re-grid each stamp onto
a common WCS before averaging:

.. code:: python

   stacked_repr = cm.stack(coords, size=60 * u.arcsec,
                           method='reproject', type='interp')


.. rubric:: Simultaneous stacking (simstack) — :meth:`~nikamap.ContMap.simstack`

Simstack (`Viero et al. 2013 <https://arxiv.org/abs/1304.0446>`_) solves for
the mean flux density of one or more source *populations* simultaneously by
fitting a linear combination of beam-convolved hit maps to the data.  Because
all populations are fitted jointly, flux cross-contamination between
spatially correlated populations is removed.

.. code:: python

   from astropy.coordinates import SkyCoord
   import astropy.units as u

   # Single population
   coords = SkyCoord(ra=..., dec=..., unit='deg')
   flux, err = cm.simstack(coords)
   print(f"Mean flux: {flux[0]:.3f} ± {err[0]:.3f} {cm.unit}")

**Multiple populations** — pass a list of :class:`~astropy.coordinates.SkyCoord`
objects, one per population:

.. code:: python

   coords_bright = SkyCoord(...)
   coords_faint  = SkyCoord(...)

   fluxes, errs = cm.simstack([coords_bright, coords_faint])
   # fluxes[0] → mean flux of bright population
   # fluxes[1] → mean flux of faint population

**Offset term** — add a constant sky offset regressor:

.. code:: python

   fluxes, errs = cm.simstack(coords, add_offset=True)
   # fluxes[-1] is the fitted offset

**Bootstrap uncertainty** — same ``n_bootstrap`` keyword as for ``stack``:

.. code:: python

   fluxes, errs = cm.simstack(coords, n_bootstrap=200)

.. note::

   The low-level :func:`nikamap.stack.simstack` function accepts raw arrays
   (``data``, ``wcs``, ``kernel``, ``coords``) and can be used independently
   of :class:`~nikamap.ContMap`.


Noise realisation classes
--------------------------

When several independent scan maps are available, ``nikamap`` provides two
estimators for the map noise that are free of common-mode artefacts.

.. autosummary::

   ~nikamap.HalfDifference
   ~nikamap.Jackknife
   ~nikamap.Bootstrap

See the :doc:`API reference <api>` and the Sphinx-Gallery examples for usage.


See also
--------

- :doc:`API reference <api>`
- :ref:`sphx_glr_auto_examples_fake_data_plot_stacking.py`
- :ref:`sphx_glr_auto_examples_fake_data_plot_simstack.py`
