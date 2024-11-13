from thefriendlystars import *
from astropy.time import Time
from ipywidgets import Output, AppLayout


def propagate_proper_motions(stars, epoch="now"):
    if epoch == "now":
        epoch = Time.now().decimalyear

    from copy import deepcopy

    propagated = deepcopy(stars)
    original_epoch = stars.meta["epoch"]

    dt = (epoch - stars.meta["epoch"]) * u.year
    propagated["ra"] = stars["ra"] + dt * stars["pmra"] / np.cos(stars["dec"])
    propagated["dec"] = stars["dec"] + dt * stars["pmdec"]
    propagated.meta["epoch"] = epoch

    dra = (propagated["ra"] - stars["ra"]) * np.cos(stars["dec"])
    ddec = propagated["dec"] - stars["dec"]
    motion = np.sqrt(dra**2 + ddec**2).to(u.arcsec)

    print(
        f"""
    Propagating proper motions from {original_epoch:.2f} (catalog) to {epoch:.2f} (requested).
    The largest motion was {np.max(motion):.3g}; the median was {np.median(motion):.3g}.
    """
    )
    return propagated


class Finder:
    def __init__(self, name, epoch="now", **kwargs):
        self.stars_at_gaia_epoch = get_gaia(name, **kwargs)
        self.stars = propagate_proper_motions(self.stars_at_gaia_epoch, epoch=epoch)

    def plot(self, **kwargs):
        plot_gaia(self.stars, **kwargs)

    def interact(
        self,
        filter="G_gaia",
        faintest_magnitude_to_show=20,
        faintest_magnitude_to_label=15,
        size_of_zero_magnitude=100,
        unit=u.arcmin,
        **kwargs,
    ):
        """
        Plot a finder chart using results from `get_gaia_data`.

        Use the table of positions and photometry returned by
        the `get_gaia_data` function to plot a finder chart
        with symbol sizes representing the brightness of the
        stars in a particular filter.

        Parameters
        ----------
        filter : str
            The filter to use for setting the size of the points.
            Options are "G_gaia", "RP_gaia", "BP_gaia", "g_sloan",
            "r_sloan", "i_sloan", "V_johnsoncousins", "R_johnsoncousins",
            "I_johnsoncousins". Default is "G_gaia".
        faintest_magnitude_to_show : float
            What's the faintest star to show? Default is 20.
        faintest_magnitude_to_label : float
            What's the faintest magnitude to which we should
            add a numerical label? Default is 16.
        size_of_zero_magnitude : float
            What should the size of a zeroth magnitude star be?
            Default is 100.
        unit : Unit
            What unit should be used for labels? Default is u.arcmin.
        """

        table = self.stars
        # extract the center and size of the field
        center = table.meta["center"]
        radius = table.meta["radius"]

        # find offsets relative to the center
        dra = ((table["ra"] - center.ra) * np.cos(table["dec"])).to(unit)
        ddec = (table["dec"] - center.dec).to(unit)

        # set the sizes of the points
        mag = table[f"{filter}_mag"].to_value("mag")
        size_normalization = size_of_zero_magnitude / faintest_magnitude_to_show**2
        marker_size = (
            np.maximum(faintest_magnitude_to_show - mag, 0) ** 2 * size_normalization
        )

        # handle astropy units better
        with quantity_support():
            with plt.ioff():
                fig = plt.figure(dpi=150)

            # plot the stars
            plt.scatter(
                dra,
                ddec,
                s=marker_size,
                color="black",
                picker=True,
                pickradius=5,
                **kwargs,
            )
            plt.xlabel(
                rf"$\Delta$(Right Ascension) [{unit}] relative to {center.ra.to_string(u.hour, format='latex', precision=2)}"
            )
            plt.ylabel(
                rf"$\Delta$(Declination) [{unit}] relative to {center.dec.to_string(u.deg, format='latex', precision=2)}"
            )
            plt.title(f'{self.stars.meta["epoch"]:.2f}')

            # add labels
            filter_label = filter.split("_")[0]
            to_label = np.nonzero(mag < faintest_magnitude_to_label)[0]
            for i in to_label:
                plt.text(
                    dra[i],
                    ddec[i],
                    f"  {filter_label}={mag[i]:.2f}",
                    ha="left",
                    va="center",
                    fontsize=5,
                )

            # add a grid
            plt.grid(color="gray", alpha=0.2)

            # plot a circle for the edge of the field
            circle = plt.Circle(
                [0, 0], radius, fill=False, color="gray", linewidth=2, alpha=0.2
            )
            plt.gca().add_patch(circle)

            # set the axis limits
            plt.xlim(radius, -radius)
            plt.ylim(-radius, radius)
            plt.axis("scaled")

            # set up interaction defaults
            highlight_color = "darkorchid"
            highlight_alpha = 0.5

            # simple shortcut for x + y coordinates
            x = dra
            y = ddec

            # create a space for displaying text output
            o = Output()

            # create an empty dictionary
            self.selected = {}

            def onpick(event):
                """
                When a star is picked, highlight it and add it to a self.selected dictionary.

                Parameters
                ----------
                event : PickEvent
                """

                # put text outputs in a specific spot
                with o:
                    # ignore scroll events (and others beside mouse clicks)
                    if event.mouseevent.name == "button_press_event":
                        # erase all previous output
                        o.clear_output()
                        # print(f"{event.mousevent}")

                        # extract which index of the plotted (x,y) was self.selected
                        i = event.ind[0]

                        # events.append(event)

                        # add the point into a dictionary of "self.selected" objects
                        if i not in self.selected:
                            self.selected[i] = {
                                "(x,y)": (x[i], y[i]),  # position object
                                "label": plt.text(
                                    x[i],
                                    y[i],
                                    f"[{i}]\n\n",  # text label above star
                                    fontsize=7,
                                    color=highlight_color,
                                    alpha=highlight_alpha,
                                    va="center",
                                    ha="center",
                                ),
                                "circle": plt.scatter(
                                    x[i],
                                    y[i],  # plotted circle around star
                                    s=100,
                                    facecolor="none",
                                    edgecolor=highlight_color,
                                    alpha=highlight_alpha,
                                ),
                            }
                        # if point was previously self.selected, remove it!
                        else:
                            self.selected[i]["label"].remove()
                            self.selected[i]["circle"].remove()
                            self.selected.pop(i)

                        # print summary of the self.selected stars
                        for i in self.selected:
                            G = table[i]["G_gaia_mag"]
                            BP_minus_RP = (
                                table[i]["BP_gaia_mag"] - table[i]["RP_gaia_mag"]
                            )
                            label = f"[{i}]"
                            print(f"{label:>8}, G={G:.2f}, Bp-Rp={BP_minus_RP:.2f}")

                        if len(self.selected) == 2:
                            i_A, i_B = self.selected.keys()
                            A = table[i_A]
                            B = table[i_B]

                            dra_AB = (B["ra"] - A["ra"]) * np.cos(
                                0.5 * (A["dec"] + B["dec"])
                            )
                            ddec_AB = B["dec"] - A["dec"]

                            angle = np.arctan2(ddec_AB, dra_AB).to("deg")
                            PA = 90 * u.deg - angle
                            kosmos_rotation = PA - 90 * u.deg
                            kosmos_rotation_string = (
                                f"RotType=Object; RotAng={kosmos_rotation:.2f}"
                            )

                            ra_center = 0.5 * (A["ra"] + B["ra"])
                            dec_center = 0.5 * (A["dec"] + B["dec"])
                            kosmos_center = SkyCoord(
                                ra=ra_center.filled(np.nan),
                                dec=dec_center.filled(np.nan),
                            )
                            kosmos_center_string = kosmos_center.to_string(
                                "hmsdms", sep=":"
                            )

                            print()
                            print(
                                f"To align stars [{i_A}] and [{i_B}] on a KOSMOS slit, try:"
                            )
                            print(
                                f"center-of-two-stars {kosmos_center_string} {kosmos_rotation_string}"
                            )

            # tell the figure to watch for "pick" events
            cid = fig.canvas.mpl_connect("pick_event", onpick)

            layout = AppLayout(
                center=fig.canvas,
                footer=o,
            )
            display(layout)

            # display(fig.canvas)
            # display(o)

    def show_lightcurves(self):
        """
        (Not yet really tested. Needs work + docstring + better labeling. Or maybe just use Autumn's?)
        """
        from lightkurve import search_lightcurve

        for i in self.selected:
            star = self.stars[i]
            c = SkyCoord(ra=star["ra"], dec=star["dec"])
            lcs = search_lightcurve(c)
            lc = lcs[-1].download()
            lc.normalize().plot()

    def show_slit(self, slit_center=None, rotation_angle=0, slit_width=7.1*u.arcsec, slit_length=6*u.arcmin):
        """
        Add graphical representation of slit on finder chart

        Parameters
        ----------
        slit_center : SkyCoord object
            Where to place the center of the slit on the field.
        rotation_angle : int
            Angle to rotate the slit [degree, default=0]
        slit_width : float
            Width of the slit [units]
        slit_length : float
            Length of the slit [units]
        """
        from astropy.coordinates.sky_coordinate import SkyCoord

        # remove any previous patches (slits) every time we run this function.
        patches = plt.gca().patches
        for patch in patches[1:]:
            patch.remove()

        table = self.stars
        ref_center = SkyCoord(table.meta['center'])

        if slit_center is None:
            slit_center = SkyCoord(f"{np.array(table[0]['ra'].value)} {np.array(table[0]['dec'].value)}",
                                   frame="icrs", unit=(u.deg, u.deg))
        def find_delta_radec(ra_center, ra, dec_center, dec):
            """
            Mini-function to account for the spherical projection when plottting.

            Parameters
            ----------
            ra_center : float [units]
                Right Ascension of the reference center of the field.
            ra : float [units]
                Right Ascension of object.
            dec_center : float [units]
                Declination of the reference center of the field.
            dec : float [units]
                Declination of object.
            """
            delta_ra = (ra - ra_center) * np.cos(dec)
            delta_dec = (dec - dec_center)
            try:
                return delta_ra.arcmin, delta_dec.arcmin
            except:
                return np.ma.filled(delta_ra, np.nan).to_value('arcmin'), np.ma.filled(delta_dec, np.nan).to_value('arcmin')

        # plot the slit on the finder chart
        center_pos_of_slit = np.array(find_delta_radec(ref_center.ra, slit_center.ra, ref_center.dec, slit_center.dec))
        xy = center_pos_of_slit - [0.5*slit_length.to_value("arcmin"), 0.5*slit_width.to_value("arcmin")]
        slit = plt.Rectangle(xy=xy, width = slit_length.to_value("arcmin"), height = slit_width.to_value("arcmin"),
                             alpha=0.3, angle=-rotation_angle, rotation_point='center')

        # determine which stars will be within the slit:
        ra = table['ra']
        dec = table['dec']
        dra, ddec = find_delta_radec(ref_center.ra, ra, ref_center.dec, dec)

        star_positions = [[r, d] for r, d in zip(dra, ddec)]
        stars_in_slit = slit.contains_points(star_positions)

        if np.any(stars_in_slit):
            print("The slit will overlap: ")
            for i, (s_id, gmag, r, d) in enumerate(zip(table['source_id'], table['G_gaia_mag'], dra, ddec)):
                if stars_in_slit[i]:
                    print(f"{s_id}, G-mag {gmag:.2f} at $\Delta$RA: {r:.3f}', $\Delta$Dec:{d:.3f}'")

            # highlight the stars that will overlap
            # steal colors from Zach:
            highlight_color = "darkorchid"
            highlight_alpha = 0.5
            plt.scatter(
                dra[stars_in_slit],
                ddec[stars_in_slit],  # plotted circle around star
                s=100,
                facecolor="none",
                edgecolor=highlight_color,
                alpha=highlight_alpha,
            )
        else:
            print("""
                  The slit will not overlap any stars on this finder chart. Maybe try changing the slit position,
                  rotation angle, or the magnitude limits of the chart?
                  """)

        plt.gca().add_patch(slit)