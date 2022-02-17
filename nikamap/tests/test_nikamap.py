from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

from astropy.nddata import StdDevUncertainty
from astropy.modeling import models
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.convolution import MexicanHat2DKernel

from photutils.datasets import make_gaussian_sources_image

import matplotlib.pyplot as plt

from ..nikamap import NikaMap, NikaFits, retrieve_primary_keys


@pytest.fixture(scope="session")
def generate_fits(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("nm_map")
    filename = str(tmpdir.join("map.fits"))
    # Larger map to perform check_SNR

    np.random.seed(0)

    shape = (256, 256)
    pixsize = 1 / 3 * u.deg
    peak_flux = 1 * u.Jy
    noise_level = 0.1 * u.Jy / u.beam
    fwhm = 1 * u.deg
    nsources = 1

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    xx, yy = np.indices(shape)
    mask = np.sqrt((xx - (shape[1] - 1) / 2) ** 2 + (yy - (shape[0] - 1) / 2) ** 2) > shape[0] / 2

    sources = Table(masked=True)
    sources["amplitude"] = np.ones(nsources) * peak_flux
    sources["x_mean"] = [shape[1] / 2]
    sources["y_mean"] = [shape[0] / 2]

    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    sources["x_stddev"] = np.ones(nsources) * beam_std_pix
    sources["y_stddev"] = np.ones(nsources) * beam_std_pix
    sources["theta"] = np.zeros(nsources)

    data = make_gaussian_sources_image(shape, sources)

    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level.to(u.Jy / u.beam).value
    data += np.random.normal(loc=0, scale=1, size=shape) * uncertainty
    data[mask] = np.nan
    hits[mask] = 0
    uncertainty[mask] = 0

    header = wcs.to_header()
    header["UNIT"] = "Jy / beam", "Fake Unit"

    primary_header = fits.header.Header()
    primary_header["f_sampli"] = 10.0, "Fake the f_sampli keyword"
    primary_header["FWHM_260"] = fwhm.to(u.arcsec).value, "[arcsec] Fake the FWHM_260 keyword"
    primary_header["FWHM_150"] = fwhm.to(u.arcsec).value, "[arcsec] Fake the FWHM_150 keyword"

    primary_header["nsources"] = 1, "Number of fake sources"
    primary_header["noise"] = noise_level.to(u.Jy / u.beam).value, "[Jy/beam] noise level per map"

    primary = fits.hdu.PrimaryHDU(header=primary_header)

    hdus = fits.hdu.HDUList(hdus=[primary])
    for band in ["1mm", "2mm"]:
        hdus.append(fits.hdu.ImageHDU(data, header=header, name="Brightness_{}".format(band)))
        hdus.append(fits.hdu.ImageHDU(uncertainty, header=header, name="Stddev_{}".format(band)))
        hdus.append(fits.hdu.ImageHDU(hits, header=header, name="Nhits_{}".format(band)))
        hdus.append(fits.hdu.BinTableHDU(sources, name="fake_sources"))

    hdus.writeto(filename, overwrite=True)

    return filename


def test_retrieve_primary_keys(generate_fits):

    filename = generate_fits

    with pytest.raises(AssertionError):
        retrieve_primary_keys(filename, band="toto")

    f_sampling, bmaj = retrieve_primary_keys(filename, band="1mm")
    assert f_sampling == 10.0 * u.Hz
    assert bmaj == 3600.0 * u.arcsec
    f_sampling, bmaj = retrieve_primary_keys(filename, band="2mm")
    assert f_sampling == 10.0 * u.Hz
    assert bmaj == 3600.0 * u.arcsec


def test_nikamap_read(generate_fits):

    filename = generate_fits
    primary_header = fits.getheader(filename, 0)

    data = NikaMap.read(filename)
    data_2mm = NikaMap.read(filename, band="2mm")
    data_1mm = NikaMap.read(filename, band="1mm")

    assert np.all(data._data[~data.mask] == data_1mm._data[~data_1mm.mask])
    assert np.all(data._data[~data.mask] == data_2mm._data[~data_2mm.mask])

    assert data.beam.major.to(u.arcsec).value == primary_header["FWHM_260"]
    assert np.all(data.time[~data.mask].to(u.h).value == ((primary_header["F_SAMPLI"] * u.Hz) ** -1).to(u.h).value)

    data_revert = NikaMap.read(filename, revert=True)
    assert np.all(data_revert._data[~data_revert.mask] == -1 * data._data[~data.mask])


def test_nikamap_write(generate_fits):
    filename = generate_fits

    data = NikaMap.read(filename)
    data_2mm = NikaMap.read(filename, band="2mm")
    data_1mm = NikaMap.read(filename, band="1mm")

    outfilename = filename.replace("map.fits", "map2.fits")
    data.write(outfilename)
    data_1mm.write(outfilename, overwrite=True)
    data_2mm.write(outfilename, append=True)


def test_nikafits_read(generate_fits):
    filename = generate_fits
    primary_header = fits.getheader(filename, 0)

    data = NikaFits.read(filename)
    assert data.primary_header == primary_header
    assert len(data) == 5
    assert isinstance(data["1mm"], NikaMap)
    assert list(data.keys()) == ["1mm", "2mm", "1", "2", "3"]
    nm = NikaMap.read(filename, band="1mm")
    assert np.nanstd((data["1mm"].subtract(nm))) == 0
