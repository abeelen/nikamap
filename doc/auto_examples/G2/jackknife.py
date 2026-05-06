"""
=================
Jackknife dataset
=================

Simple example of jacknnife generation on G2 dataset from the N2CLS GTO.

"""
###############################################################################
# First import the :class:`nikamap.jackkinfe` class

from pathlib import Path
from nikamap import Jackknife

################################################################################
# This define the root directory where all the data ...

DATA_DIR = Path("/data/NIKA/Reduced/G2_COMMON_MODE_ONE_BLOCK/v_1")

################################################################################
# can be retrieved using a simple regular expression

filenames = list(DATA_DIR.glob('*/map.fits'))
filenames

################################################################################
# Create a jackknife object for future use in, for e.g., a simulation,
#
# .. note:: by default the constructor will read all the maps in memory

jacks = Jackknife(filenames, n=10)


################################################################################
# The `jacks` object can be iterated upon, each of the items is a different jackknife with noise properties corresponding to the original dataset


for nm in jacks:
    print(nm.check_SNR())
