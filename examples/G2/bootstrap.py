"""
==================
Bootstraping error
==================

Simple example of bootstrap on G2 dataset from the N2CLS GTO.

"""

################################################################################
# First import the :func:`nikamap.bootstrap` function

from pathlib import Path
from nikamap import bootstrap

################################################################################
# This define the root directory where all the data ...

DATA_DIR = Path("/data/NIKA/Reduced/G2_COMMON_MODE_ONE_BLOCK/v_1")

################################################################################
# can be retrieved using a simple regular expression

filenames = list(DATA_DIR.glob('*/map.fits'))
filenames


################################################################################
# Generate a bootstrap dataset for all the maps.
#
# .. note:: At the moment, this is very memory demanding as we use
#           ```map_size * (len(filenames) + n_bootstrap))```


nm = bootstrap(filenames, n_bootstrap=200, ipython_widget=True)


#################################################################################
# In the resulting :class:`nikamap.NikaMap` object, the uncertainty as been computed using the bootstrap technique.
#
# .. warning:: The resulting uncertainty map could still be biased, see https://gitlab.lam.fr/N2CLS/NikaMap/issues/4

_ = nm.plot_SNR(cbar=True)
