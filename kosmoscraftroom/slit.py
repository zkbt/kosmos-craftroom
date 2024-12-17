from .detector import *


class Slit:
    def __init__(self, detector=None):
        self.detector = detector

    def _get_optic(self, image):
        """
        Get the filter and/or disperser in front of the detector.

        This is needed to group flats.
        """
        return image.header["DISPERSR"]

    def make_reference_calibrations(self, flat_filenames={}, arc_filenames={}):
        pass

    def make_reference_flat(self, flat_filenames=[]):
        """
        Stack a group of flat images together into a reference flat estimate.

        Parameters
        ----------
        flat_filenames : list
            A list of flat filenames.
        """

        flat_exposures = [self.read(f) for f in flat_filenames]
        print(f"   found {len(flat_exposures)} flat exposures")

        # organize flat images by filter/disperser
        print("🌅 calculating reference flats and gain")
        print("    grouping flats by filter")
        optics = [self._get_optic(f) for f in flat_exposures]
        exposures_for_optics = {}
        for e in flat_exposures:
            k = self._get_optic(e)
            if k not in exposures_for_optics:
                exposures_for_optics[k] = []
            exposures_for_optics[k].append(e)

        print("    bias-subtracting and normalizing each flat image")
        flat_normalizations, flat_normalized, flats = {}, {}, {}
        for k in exposures_for_optics:
            print(f"     {k}")
            for flat in exposures_for_optics[k]:
                b = self.hdus["bias"].data
                d = self.hdus["dark"].data * self._get_exposure_time(flat)
                normalization = np.nanmedian(flat - b - d)
                if normalization > 0:
                    if k not in flat_normalizations:
                        flat_normalizations[k] = []
                    if k not in flat_normalized:
                        flat_normalized[k] = []
                    flat_normalizations[k].append(normalization)
                    flat_normalized[k].append((flat - b - d) / normalization)

        print("    calculating reference as median of normalized flat images")
        for k in flat_normalized:
            flat_data = np.nanmedian(flat_normalized[k], axis=0)
            flat_header = extract_constant_header(exposures_for_optics[k])
            flats[k] = fits.ImageHDU(
                data=flat_data,
                header=flat_header,
                name=f"flat-{k}",
            )
            flats[k].header["NFLAT"] = (
                len(exposures_for_optics[k]),
                "How many flat frames were combined to make this?",
            )
            flats[k].header["UNIT"] = (
                "unitless",
                "What are the units of this image?",
            )
        # self.estimate_gain_from_flats(flat_normalized, flat_normalizations)

        for k in flats:
            self.hdus.append(flats[k])

    def estimate_gain_from_flats(self, flat_normalized, flat_normalizations):
        """
        Estimate the gain from the per-pixel scatter in normalized flats.
        """
        print("    estimating the detector gain based on variance of normalized flats")
        fi, ax = plt.subplots(
            2, 1, constrained_layout=True, sharex=True, figsize=(10, 6)
        )
        xbinning, ybinning = self.binning
        bin_string = f"{xbinning}x{ybinning}"
        all_gain_estimates, all_sigmas, all_x = [], [], []
        for k in flat_normalized:
            differences = np.array(flat_normalized[k][1:]) - np.array(
                flat_normalized[k][:-1]
            )
            x_for_gain = (
                np.array(flat_normalized[k][1:])
                / np.array(flat_normalizations[k][1:])[:, np.newaxis, np.newaxis]
                + np.array(flat_normalized[k][:-1])
                / np.array(flat_normalizations[k])[:-1, np.newaxis, np.newaxis]
            )

            x = x_for_gain.flatten()
            y = differences.flatten()
            center, sigma = bin_std(x, y)
            gain_estimates = center / sigma**2
            all_x.extend(center)
            all_gain_estimates.extend(gain_estimates)
            all_sigmas.extend(sigma)

            plt.sca(ax[0])
            plt.plot(center, sigma**2, "o", label=k, alpha=0.25)

            plt.sca(ax[1])
            plt.plot(
                center,
                gain_estimates,
                "o",
                label=f"{k} | g={np.nanmedian(gain_estimates):.3f}",
                alpha=0.25,
            )
            plt.ylim(0, None)

        unbinned_gain_from_header = self.gain  # _get_gain(flat_exposures[0])
        gain_from_header = xbinning * ybinning * unbinned_gain_from_header

        gain_from_header_string = f"{xbinning}x{ybinning}x{unbinned_gain_from_header:.3f} = {gain_from_header:.3f} photons/ADU (from header)"
        # if self.trust_header_gain:
        #    gain = gain_from_header
        # else:
        gain = np.nanmedian(all_gain_estimates)
        plt.sca(ax[0])
        plt.title(f"{self.instrument.capitalize()} | {bin_string}")
        plt.ylabel(r"$\sf \sigma_{\Delta}^2 = \sigma_{I_{i+1} - I_{i}}^2$")
        # plt.ylim(*np.nanpercentile(all_sigmas, [5,95])**2)
        x_smooth = np.linspace(*np.nanpercentile(all_x, [0, 100]))
        plt.plot(
            x_smooth,
            x_smooth / self.gain,
            label=f"g={gain_from_header_string}",
            linestyle="--",
            color="black",
        )
        plt.plot(
            x_smooth,
            x_smooth / gain,
            label=f"g={gain:.3f} photons/ADU",
            color="black",
        )
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(bbox_to_anchor=(1, 1), frameon=False)
        plt.ylim(np.min(x_smooth) / gain, np.max(x_smooth) / gain)

        plt.sca(ax[1])

        r = np.array(all_gain_estimates) - gain

        span = np.sqrt(np.nanmedian(r[np.isfinite(r)] ** 2))

        plt.axhline(
            gain_from_header,
            label=f"g = {gain_from_header_string}",
            color="black",
            linestyle="--",
        )
        plt.axhline(gain, label=f"g = {gain:.3f} photons/ADU", color="black")
        plt.ylim(
            0.8
            * np.minimum(
                gain, gain_from_header
            ),  # np.max([0, 0.9*np.minimum(gain, gain_from_header), np.nanpercentile(all_gain_estimates, 5)]),
            1.2 * np.maximum(gain, gain_from_header),
        )  # np.max([n*1.1, gain_from_header*1.1, np.nanpercentile(all_gain_estimates, 95)]))

        plt.xlabel(
            r"$\sf x = I_{i+1}/median(E_{i+1}-E_{bias}) + I_{i}/median(E_{i}-E_{bias})$"
        )
        plt.ylabel(r"Gain ($\sf e^-/ADU$)")
        plt.legend(bbox_to_anchor=(1, 1), frameon=False)
        plt.savefig(f"{str(self)[1:-1]}-estimated-gain.pdf")
