master
======

Added
-----

Fixed
-----

Version 1.2
===========

Switch to photutils v1.13, incompatible with previous version, implies python > 3.10

Added
-------
* Photometry can use global or local background
* `phot_source` option `fixed_psf` renamed `fixed_positions`
* `phot_source` option `fixed_sigma` to allow for varying sigma in psf fiting

Fixed
-----
* matched filter hits must be smooth by the kernerl squared !
* fix non standard fits header value as json dump
* switch to photutils >= 1.13, which implies python > 3.10
* switch from deprecated `photutils.datasets.make_model_sources_image` to `photutils.datasets.make_model_image`

Version 1.0
===========

Switch to photutils v1.9, incompatible with previous version

Added
-----
* photometric background can now be optionnal
* photometric background is done locally
* psf fitting can fit for source position

Version 0.6
===========

Added
-----
* Last version compatible with photutils < 0.9

Fixed
-----
* hits are save as uint32 to avoid overflow

Version 0.5
===========

Added
-----
* beam option in plot* methods
* Stacking Methods for ContMap

Fixed
-----
* default n_boostrap increased
* Bootstrap and Jackknife now applies to ContMap as well
* Using powspec package instead of internal function

Version 0.4
===========

Added
-----
* First release for Zenodo
* Early CITATION.cff

Fixed
-----
* Documentation is running again


Version 0.3.8
=============

Added
-----

* Add `range` and `return_mean` to `check_SNR()`
* Re-introducing `NikaBeam` with old API for back compatibility

Fixed
-----

* PIIC support is fixed

Version 0.3.6
=============

Added
-----
* `find_sources` save pixels coordinates for peak and centroids
* `phot_sources` with `psf=True` save residual map

Fixed
-----

* Proper Handling of all NDUncertainty
* Proper Handling for history and comment in meta/header
* Better Handling of custom WCS ctype in find_peak

Version 0.3
===========

Added
-----
* Full refactoring splitting NikaMap into ContMap (general) & NikaMap
* Better handling of beams with ContBeam, same API as radio_beam
* Removed NikaBeam, replaced by ContBeam

Fixed
-----
* Better PIIC map handling
* Some newer version of photutils centroids would have crashed `find_sources`

Version 0.2.2
=============

Added
-----
* Jackknife class renamed as HalfDifference
* Jackknife now provide a proper jackknife resampling

Vesion 0.2
==========
Added
-----
* Boostrap sampling

Fixed
-----
* use the new astropy low_level_wcs interface when present (astropy 4 compatibility)
* general piic format (rgw instead of snr)
* Refactor Jackknife

Version 0.1.2
=============

Added
-----
* `PIIC_fits_nikamap_reader` to read fits file produced by PIIC

Fixed
-----
* resolve astropy v4 unit handling issue in powspec_k

Version 0.1.1
=============

Added
-----

Fixed
-----
* photutils API change in v0.7



Version 0.1
===========

First published version. See documentation
