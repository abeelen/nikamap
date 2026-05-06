"""
===============
Cutout stacking
===============

Stack continuum map stamps centred on a prior source catalogue.

This example shows how to use :meth:`~nikamap.ContMap.stack` to produce a
weighted-average stacked map from a set of source positions.  No real data is
needed — we build a synthetic :class:`~nikamap.ContMap` with known injected
sources and then stack on their positions.
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.table import Table
from astropy.wcs import WCS
from photutils.datasets import make_model_image

from nikamap import ContBeam, ContMap

rng = np.random.default_rng(42)

###############################################################################
# Build a synthetic map
# ---------------------
#
# We create a 128 × 128 pixel map at 2″/pixel with a 12.5″ FWHM beam
# and inject 30 point sources at uniform random positions.

shape = (256, 256)
pixscale = 2 * u.arcsec
fwhm = 12.5 * u.arcsec
noise_level = 1e-3  # Jy/beam

wcs = WCS(naxis=2)
wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
wcs.wcs.cdelt = [-pixscale.to("deg").value, pixscale.to("deg").value]
wcs.wcs.crval = [0.0, 0.0]
wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

# Random source positions (pixel, integer for exact centering)
nsources = 30
margin = 15  # pixels — keep sources away from edges
x_pix = rng.integers(margin, shape[1] - margin, nsources)
y_pix = rng.integers(margin, shape[0] - margin, nsources)
peak_flux = 5e-3  # Jy/beam

beam_std_pix = (fwhm / pixscale).decompose().value * gaussian_fwhm_to_sigma
source_table = Table(
    {
        "x_mean": x_pix.astype(float),
        "y_mean": y_pix.astype(float),
        "amplitude": np.full(nsources, peak_flux),
        "x_stddev": np.full(nsources, beam_std_pix),
        "y_stddev": np.full(nsources, beam_std_pix),
        "theta": np.zeros(nsources),
    }
)

source_map = (
    make_model_image(shape, models.Gaussian2D(), source_table,
                     model_shape=shape, x_name="x_mean", y_name="y_mean")
)
noise_map = rng.normal(0, noise_level, size=shape)

cm = ContMap(
    source_map + noise_map,
    uncertainty=StdDevUncertainty(np.full(shape, noise_level)),
    wcs=wcs,
    unit=u.Jy / u.beam,
    beam=ContBeam(major=fwhm, pixscale=pixscale),
)

###############################################################################
# Recover the sky coordinates of the injected sources

world = wcs.pixel_to_world(x_pix, y_pix)
coords = SkyCoord(world)

print(f"Number of stacked positions: {len(coords)}")

###############################################################################
# Stack the cutouts
# -----------------
#
# :meth:`~nikamap.ContMap.stack` extracts a stamp of angular size ``size``
# around each coordinate and returns an inverse-variance weighted average.
# The uncertainty of the output map is derived from the combined weights.

stacked = cm.stack(coords, size=40 * u.arcsec)

print(stacked)

###############################################################################
# The peak of the stacked map should recover the injected peak flux

peak = float(np.nanmax(stacked.data))
print(f"Injected peak flux : {peak_flux * 1e3:.1f} mJy/beam")
print(f"Stacked peak flux  : {peak * 1e3:.1f} mJy/beam")

###############################################################################
# Plot
# ----

fig, axes = plt.subplots(1, 2, figsize=(10, 4),
                         subplot_kw={"projection": cm.wcs})
cm.plot(ax=axes[0], cbar=True)
axes[0].set_title("Synthetic map")
axes[0].scatter([c.ra.deg for c in coords],
                [c.dec.deg for c in coords],
                transform=axes[0].get_transform("world"),
                s=10, c="red", label="sources")
axes[0].legend(loc="upper right")

stacked.plot(ax=axes[1], cbar=True)
axes[1].set_title("Stacked stamp")
plt.tight_layout()
