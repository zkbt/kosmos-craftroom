from astropy.nddata import CCDData
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
from astropy.stats import mad_std
import warnings
from astropy.time import Time
from IPython.display import Image, display
import copy


def bin_std(x, y, N=50):
    """
    Calculate standard deviation in bins spanning x range.

    Parameters
    ----------
    x : array
        x values, which will be used to define which bins fall
    y : array
        y values,
    """
    grid = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), N)
    centers = grid[:-1] + np.diff(grid) / 2
    sigmas = np.array(
        [mad_std(y[(x > grid[i]) * (x < grid[i + 1])]) for i in range(len(grid) - 1)]
    )
    return centers, sigmas


def extract_constant_header(images):
    """
    From a list of CCDData or FITS images, make a header with the information that stays constant.
    """
    constant_header = fits.Header()
    headers = [i.header for i in images]
    for k in headers[0].keys():
        try:
            values = [h[k] for h in headers]
            unique_values = np.unique(values)
            if len(unique_values) == 1:
                constant_header[k] = unique_values[0]
        except KeyError:
            pass

    return constant_header


def bin_string(b):
    return f"{b[0]}x{b[1]}"


class Detector:
    def __repr__(self):
        """
        Represent this object as a string.
        """
        return f"<{self.instrument}-{bin_string(self.binning)}>"

    def __init__(self, binning=[2, 2]):
        """
        Initialize this detector, with one particular binning.
        """
        self.binning = binning
        self.hdus = fits.HDUList()

    def _verify_binning(self, image):
        """
        Confirm that the binning of a loaded image matches this detector.
        """
        image_binning = self._get_binning(image)

        try:
            assert np.all(image_binning == self.binning)
        except AssertionError:
            raise RuntimeError(
                f"""
            {bin_string(self.binning)} binning is set for {self}, but
            {image._filename}
            appears to be set to {bin_string(binning)}.
            """
            )

    def read(self, filename):
        """
        Read a filename into a CCDData object.

        Parameters
        ----------
        filename : str
            The path to the image to load.


        KOSMOS images have 1 extension, including both data + header.

        No.    Name      Ver    Type      Cards   Dimensions   Format
        0  PRIMARY       1 PrimaryHDU      78   (2148, 4096)   int32
        """

        # read raw file into a CCDData object with .data and .header
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image = CCDData.read(filename, unit="adu")
            image._filename = filename

        # make sure the binning matches this detector
        self._verify_binning(image)

        return image

    def make_reference_bias(self, bias_filenames=[]):
        """
        Stack a group of bias images together into a reference bias + read noise estimate.

        Parameters
        ----------
        bias_filenames : list
            A list of bias filenames.
        """

        # read images
        bias_exposures = [self.read(f) for f in bias_filenames]
        print(f"   found {len(bias_exposures)} bias exposures")

        # calculate bias level and read noise
        print("🌌 calculating reference bias image and read noise")
        bias = np.median(bias_exposures, axis=0)

        # make a header for this bias
        bias_header = extract_constant_header(bias_exposures)

        # store a FITS image with this reference bias
        self.hdus.append(fits.ImageHDU(data=bias, header=bias_header, name="bias"))

        # calculate read noise estimate
        read_noise_adu = np.median(mad_std(bias_exposures, axis=0))
        print(f"    the median bias level is {np.median(bias)} ADU")
        print(f"    the read noise is {read_noise_adu} ADU")
        self.hdus["bias"].header["NBIAS"] = (
            len(bias_exposures),
            "How many bias frames were combined to make this?",
        )
        self.hdus["bias"].header["UNIT"] = ("adu", "What are the units of this image?")
        self.hdus["bias"].header["RN_ADU"] = read_noise_adu
        self.bias = self.hdus["bias"].data

    def make_reference_dark(self, dark_filenames=[]):
        """
        Stack a group of dark images together into a reference dark rate estimate.

        Parameters
        ----------
        dark_filenames : list
            A list of dark filenames.
        """

        # read images
        dark_exposures = [self.read(f) for f in dark_filenames]
        print(f"   found {len(dark_exposures)} dark exposures")

        if len(dark_exposures) == 0:
            print("🌃 assuming reference dark current rate to be 0")
            dark_rate = self.bias * 0 / u.s
            dark_header = None
        else:
            # how fast does dark current accumulate (ADU/s)
            print("🌃 calculating reference dark current rate")
            b = self.hdus["bias"].data
            dark_rates = [(d - b) / self._get_exposure_time(d) for d in dark_exposures]
            dark_rate = np.median(dark_rates, axis=0)
            dark_rate = np.maximum(dark_rate, 0)

            # make a header for this dark
            dark_header = extract_constant_header(dark_exposures)

        # store a FITS image with this reference bias
        self.hdus.append(
            fits.ImageHDU(
                data=dark_rate,
                header=dark_header,
                name="dark",
            )
        )
        self.hdus["dark"].header["NDARK"] = (
            len(dark_exposures),
            "How many dark frames were combined to make this?",
        )
        self.hdus["dark"].header["UNIT"] = (
            "adu/s",
            "What are the units of this image?",
        )
        print(f"    the median dark current rate is {np.median(dark_rate):.3f} ADU/s")
        self.dark = self.hdus["dark"].data

    def reference_filename(self):
        return f"{str(self)[1:-1]}-detector-calibrations.fits"

    def make_reference_calibrations(
        self,
        bias_filenames=[],
        dark_filenames=[],
    ):
        print("🧮 constructing reference calibration images")
        self.make_reference_bias(bias_filenames=bias_filenames)
        self.make_reference_dark(dark_filenames=dark_filenames)

        print(f"💾 saving reference calibration data into {self.reference_filename()}")
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["SCOPE"] = (
            self.telescope,
            "Which telescope are these calibration for?",
        )
        primary_hdu.header["INSTRUME"] = (
            self.instrument,
            "Which detector are these calibration for?",
        )
        primary_hdu.header["DETECTOR"] = (
            self.detector,
            "Which detector are these calibration for?",
        )
        primary_hdu.header["DATE"] = (
            Time.now().iso,
            "When were these calibrations made?",
        )
        primary_hdu.header["XBINNING"] = (
            self.binning[0],
            f"How many pixels are binned together in x?",
        )
        primary_hdu.header["YBINNING"] = (
            self.binning[1],
            f"How many pixels are binned together in Y?",
        )
        primary_hdu.header["GAIN"] = (
            self.gain,
            f"What is the {bin_string(self.binning)} binned gain, in photons/ADU?",
        )
        primary_hdu.header["IGAIN"] = (
            1 / self.gain,
            f"What is 1/gain, in ADU/photons?",
        )
        primary_hdu.header["RN_ADU"] = (
            self.hdus["bias"].header["RN_ADU"],
            f"What is the read noise, in ADU?",
        )
        primary_hdu.header["RN_PHOT"] = (
            primary_hdu.header["RN_ADU"] * self.gain,
            f"What is the read noise, in photons?",
        )
        primary_hdu.header["MEANBIAS"] = (
            np.nanmedian(self.hdus["bias"].data),
            f"What is the typical bias level, in ADU?",
        )
        primary_hdu.header["MEANDARK"] = (
            np.nanmedian(self.hdus["dark"].data),
            f"What is the typical dark rate, in ADU/s?",
        )

        self.hdus.writeto(self.reference_filename(), overwrite=True)

        return self.hdus

    def load(self, filename=None):
        self.hdus = fits.open(filename or self.reference_filename())


class SBODetector(Detector):
    def _get_binning(self, image):
        """
        Get the binning of an image.
        """
        return [image.header["XBINNING"], image.header["YBINNING"]]

    def _get_exposure_time(self, image):
        """
        Get the exposure time of an image.

        This is needed to group flats.
        """
        return image.header["EXPTIME"] * u.s


class KOSMOSDetector(Detector):
    instrument = "KOSMOS"
    telescope = "ARC 3.5m"
    detector = "E2V CCD44-82"
    gain = 0.6
    rows_unbinned = 4096
    cols_unbinned = 2048
    overscan_unbinned = 100
    overscan_skip_unbinned = 0

    def read(self, filename, visualize=False):
        """
        Subtract and trim overscan.
        """
        image = super().read(filename)
        return self.subtract_and_trim_overscan(image, visualize=visualize)

    def subtract_and_trim_overscan(self, image, visualize=False):
        cols = int(self.cols_unbinned / self.binning[0])
        rows = int(self.rows_unbinned / self.binning[1])
        overscan = self.overscan_unbinned / self.binning[0]
        overscan_skip = self.overscan_skip_unbinned / 2

        trimmed = copy.deepcopy(image[:rows, :cols])
        if visualize:
            fi, ax = plt.subplots(4, 1, sharex=True, figsize=(10, 10))
            vmin, vmax = np.percentile(image, [2, 98])

            plt.sca(ax[0])
            plt.plot(np.median(image, axis=0))
            plt.ylim(vmin, vmax)
            plt.sca(ax[1])
            plt.imshow(image, vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(orientation="horizontal")

        for i in [0, 1]:
            overscan_start = int(cols + overscan * i + overscan_skip)
            overscan_end = int(cols + overscan * (i + 1))

            correction = np.median(
                image[:, overscan_start:overscan_end],
                # axis=1,
            )
            data_start = int(i * cols / 2)
            data_end = int((i + 1) * cols / 2)

            trimmed.data[:, data_start:data_end] = (
                image[:, data_start:data_end] - correction
            )
            if visualize:
                dy = 0.1
                kw = dict(
                    alpha=0.3,
                    ymin=dy * (i + 1),
                    ymax=dy * (i + 2),
                    color=["red", "yellow"][i],
                )
                plt.axvspan(overscan_start, overscan_end, **kw)
                plt.axvspan(data_start, data_end, **kw)

        if visualize:
            plt.sca(ax[3])
            vmin, vmax = np.percentile(trimmed, [2, 98])
            plt.imshow(trimmed, vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(orientation="horizontal")
            plt.xlim(0, image.shape[1])

            plt.sca(ax[2])
            plt.plot(np.median(trimmed, axis=0))
            plt.ylim(vmin, vmax)

        return trimmed

    def _get_binning(self, image):
        """
        Get the binning of an image.
        """
        return [image.header["BINX"], image.header["BINY"]]

    def _get_exposure_time(self, image):
        """
        Get the exposure time of an image.

        This is needed to group flats.
        """
        return image.header["EXPTIME"] * u.s

    # TO-DO
    # - add explicit checks that binning matches
    # - extract timing metadata of when these data were taken?
    # - estimate uncertainties on variance, to make easier to reject noisy
    # - tidy, organize, package, DOCUMENTATION!
