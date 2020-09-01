from nikamap.utils import fake_data

import warnings
warnings.filterwarnings("ignore")


class TimeNikaMap:
    """
    Basic example to test timing in nikamap
    """

    def setup(self):
        self.nm = fake_data()
        self.mf_nm = self.nm.match_filter(self.nm.beam)
        self.mf_nm.sources = self.mf_nm.fake_sources
        self.nm.sources = self.nm.fake_sources

    # def time_read(self):
    #     self.nm = NikaMap.read(self.filename)

    def time_matchfilter(self):
        self.mf_nm = self.nm.match_filter(self.nm.beam)

    def time_detect(self):
        self.mf_nm.detect_sources()

    def time_phot_peak(self):
        self.mf_nm.phot_sources(peak=True, psf=False)

    def time_phot_psf(self):
        self.nm.phot_sources(self.mf_nm.sources, peak=False, psf=True)

    def track_sources(self):
        return len(self.nm.sources)
