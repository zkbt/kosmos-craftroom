from .iplot import iplot
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import os
from tqdm import tqdm
from astropy.stats import median_absolute_deviation
import numpy as np


class loupe(iplot):
    """
    loupe is an interactive matplotlib imshow,
    for looking at images and clicking points,
    in a matplotlib-like plot.

    It reproduces *a few* of the features
    available from using IRAF display with imexam,
    and ds9 to look at images interactively,
    but allows slightly easier integration
    with custom Python codes.
    """

    def __init__(self, **kwargs):
        """Initialize the loupe object."""

        # populate a dictionary of available actions
        # each option has a key to press,
        #   a description of what it will do
        #   a function to be called when that key is pressed
        #   and whether the function needs to know the current mouse position
        self.options = {}

        # quit the interactive loupe
        self.options["q"] = dict(
            description="[q]uit", function=self.quit, requiresposition=False
        )

        # a crosshair will plot vertical and horizontal plots at that location
        self.options["c"] = dict(
            description="move the [c]rosshair, and plot slicey along it",
            function=self.moveCrosshair,
            requiresposition=True,
        )

        # once the crosshair is set, you can movie it up and down
        self.options["up"] = dict(
            description="nudge the crosshair [up]",
            function=self.moveCrosshair,
            requiresposition=False,
        )
        self.options["down"] = dict(
            description="nudge the crosshair [down]",
            function=self.moveCrosshair,
            requiresposition=False,
        )
        self.options["left"] = dict(
            description="nudge the crosshair [left]",
            function=self.moveCrosshair,
            requiresposition=False,
        )
        self.options["right"] = dict(
            description="nudge the crosshair [right]",
            function=self.moveCrosshair,
            requiresposition=False,
        )

    @property
    def ok_slicex(self):
        """
        Return an array of which elements are OK in the x-direction,
        given the y-location of the current crosshair slice.
        """

        if self.ok is None:
            return np.ones_like(self.xaxis).astype(bool)
        else:
            i = int(
                np.interp(self.crosshair["y"], self.yaxis, np.arange(len(self.yaxis)))
            )
            return self.ok[:, i]

    @property
    def ok_slicey(self):
        """
        Return an array of which elements are OK in the y-direction,
        given the x-location of the current crosshair slice.
        """
        if self.ok is None:
            return np.ones_like(self.yaxis).astype(bool)
        else:
            i = int(
                np.interp(self.crosshair["x"], self.xaxis, np.arange(len(self.xaxis)))
            )
            return self.ok[i, :]

    @property
    def slicey(self):
        """
        Return (z, y) of a slice in the y-direction,
        for plotting in the vertical projection plot.
        """

        # figure out the column of the image associated with the crosshair's x
        i = int(np.interp(self.crosshair["x"], self.xaxis, np.arange(len(self.xaxis))))
        return self.image[i, :], self.yaxis

    @property
    def slicex(self):
        """
        Return (x, z) of a slice in the x-direction,
        for plotting in the horizontal projection plot.
        """

        # figure out the column of the image associated with the crosshair's y
        i = int(np.interp(self.crosshair["y"], self.yaxis, np.arange(len(self.yaxis))))
        return self.xaxis, self.image[:, i]

    @property
    def imagetoplot(self):
        """
        The image for plotting (transposed).
        """

        # show the transpose of the image in imshow
        return np.transpose(self.image)

    def update(self, image, **kwargs):
        """
        Once the loupe has been set up,
        modify the image being shown
        (ideally without changing anything else.)

        Parameters
        ----------
        image : 2D array
            the image to display
        """

        # store the image
        self.image = image
        #
        try:
            # have we already created a loupe here?
            self.plotted["2d"]
        except AttributeError:
            # if not, set it up!
            self.setup(image, **kwargs)

        # update the data being imshowed
        self.plotted["2d"].set_data(self.imagetoplot)

        # update the plots for the slices along each axis
        x, y = self.crosshair["x"], self.crosshair["y"]
        self.moveCrosshair(x=x, y=y)

        # force a redraw of the plot
        plt.draw()

    def setup(
        self,
        image,
        ok=None,
        xaxis=None,
        yaxis=None,
        title="",  # title to go over the top of the plot
        figsize=(8, 4),  # kwargs for figure,
        dpi=100,
        hspace=0,
        wspace=0,  # kwargs for gridspec
        left=0.05,
        right=0.95,
        bottom=0.05,
        top=0.95,
        width_ratios=[1.0, 0.1],
        height_ratios=[0.1, 1.0],
        initialcrosshairs=[0.0, 0.0],  # where the crosshairs should start
        aspect="auto",  # kwargs for imshow
        vmin=None,
        vmax=None,
        scale="symlog",  # make a default that can be edited interactively (in Mask.py)
        labelfontsize=5,
        datacolor="darkorange",
        crosshaircolor="darkorange",
        **kwargs
    ):
        """
        Initialize the loupe
        (this has a pretty big overhead,
        so use "update_image" if you can to update
        the data being displayed)
        """

        self.ok = ok
        self.image = image
        # set the image
        if xaxis is not None:
            self.xaxis = xaxis
        else:
            xsize = self.imagetoplot.shape[1]
            self.xaxis = np.arange(xsize)
        if yaxis is not None:
            self.yaxis = yaxis
        else:
            ysize = self.imagetoplot.shape[0]
            self.yaxis = np.arange(ysize)

        self.dx = np.median(np.diff(self.xaxis))
        self.dy = np.median(np.diff(self.yaxis))

        # keep track of where the crosshair is pointing
        self.crosshair = dict(x=initialcrosshairs[0], y=initialcrosshairs[1])

        # create an interactive plot
        iplot.__init__(
            self,
            2,
            2,
            hspace=hspace,
            wspace=wspace,
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            figsize=figsize,
        )

        # a dictionary to store the axes objects
        self.ax = {}

        # for displaying the 2D image
        labelkw = dict(fontsize=labelfontsize)
        self.ax["2d"] = self.subplot(1, 0)
        plt.setp(self.ax["2d"].get_xticklabels(), **labelkw)
        plt.setp(self.ax["2d"].get_yticklabels(), **labelkw)

        # for displaying cuts along the dispersion direction
        self.ax["slicex"] = self.subplot(0, 0, sharex=self.ax["2d"])
        self.ax["slicex"].set_title(title, fontsize=8)
        plt.setp(self.ax["slicex"].get_xticklabels(), visible=False)
        plt.setp(self.ax["slicex"].get_yticklabels(), **labelkw)

        # for display cuts along the cross-dispersion direction
        self.ax["slicey"] = self.subplot(1, 1, sharey=self.ax["2d"])
        self.ax["slicey"].xaxis.tick_top()
        self.ax["slicey"].xaxis.set_label_position("top")
        plt.setp(self.ax["slicey"].get_xticklabels(), rotation=270, **labelkw)
        plt.setp(self.ax["slicey"].get_yticklabels(), visible=False)

        # set the limits of the plot
        ok = np.isfinite(self.imagetoplot)
        # self.vmin, self.vmax = np.percentile(self.imagetoplot[ok], [0,100])

        # pick a scale for the plotting
        if scale == "symlog":
            norm = colors.SymLogNorm(
                linthresh=median_absolute_deviation(self.imagetoplot),
                linscale=0.1,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            norm = None

        self.extent = [
            np.min(self.xaxis),
            np.max(self.xaxis),
            np.min(self.yaxis),
            np.max(self.yaxis),
        ]

        self.plotted = {}
        # plot the image
        self.plotted["2d"] = self.ax["2d"].imshow(
            self.imagetoplot,
            cmap="gray",
            extent=self.extent,
            interpolation="nearest",
            aspect=aspect,
            zorder=0,
            origin="lower",
            norm=norm,
        )

        # set the x and y limits
        self.ax["2d"].set_xlim(self.extent[0:2])
        self.ax["2d"].set_ylim(self.extent[2:4])

        # add crosshair, to both 2D and 1D slices
        crosskw = dict(alpha=0.5, color=crosshaircolor, linewidth=1)
        self.plotted["crossy"] = self.ax["2d"].axvline(self.crosshair["x"], **crosskw)
        self.plotted["crossyextend"] = self.ax["slicex"].axvline(
            self.crosshair["x"], linestyle="--", **crosskw
        )
        self.plotted["crossx"] = self.ax["2d"].axhline(self.crosshair["y"], **crosskw)
        self.plotted["crossxextend"] = self.ax["slicey"].axhline(
            self.crosshair["y"], linestyle="--", **crosskw
        )
        # plot slicey
        slicekw = dict(color=datacolor, linewidth=1)

        badalpha = 0.15
        h, v = self.slicey
        ok = self.ok_slicey
        self.plotted["slicey"] = self.ax["slicey"].plot(h[ok], v[ok], **slicekw)[0]
        self.plotted["slicey_bad"] = self.ax["slicey"].plot(
            h[ok == False], v[ok == False], alpha=badalpha, **slicekw
        )[0]

        h, v = self.slicex
        ok = self.ok_slicex
        self.plotted["slicex"] = self.ax["slicex"].plot(h[ok], v[ok], **slicekw)[0]
        self.plotted["slicex_bad"] = self.ax["slicex"].plot(
            h[ok == False], v[ok == False], alpha=badalpha, **slicekw
        )[0]

        # set the limits of the color scale and the plots
        for a in self.ax.values():
            a.set_autoscaley_on(False)
        self.set_limits(vmin, vmax)

    def set_limits(self, vmin=None, vmax=None):
        self.plotted["2d"].set_clim(vmin, vmax)
        if self.crosshair["y"] is not None:
            self.ax["slicex"].set_ylim(vmin, vmax)
        if self.crosshair["x"] is not None:
            self.ax["slicey"].set_xlim(vmin, vmax)
        plt.draw()

    def run(
        self,
        message="",  # something to annouce each loop
    ):
        # update the plot
        # plt.draw()

        # keep track of whether we're finished
        self.notconverged = True
        while self.notconverged:
            # say the message at the start of each loop
            self.speak(message)

            # print the self.options
            self.speak("your self.options include:")
            for v in self.options.values():
                self.speak("   " + v["description"])

            # get the keyboard input
            pressed = self.getKeyboard()
            self.speak('"{}" was pressed'.format(pressed.key))
            # process the keyboard input
            try:
                # figure out which option we're on
                thing = self.options[pressed.key.lower()]
                # check that it's a valid position, if need be
                if thing["requiresposition"]:
                    assert pressed.inaxes is not None
                # execute the function associated with this option
                thing["function"](pressed)
            except KeyError:
                self.speak("nothing yet defined for [{}]".format(pressed.key))
            except AssertionError:
                self.speak("that didn't seem to be at a valid position!")

            # update the plot
            plt.draw()

    def quit(self, *args):
        """
        If a [q] is pressed, quit the plot.
        """
        self.notconverged = False
        plt.close(self.figure)

    """
    ### FIX ME -- I think this belongs with mosasaurus, but got copied here. ###
    def setScale(self, pressed=None, default=False):
        '''prompt the user to select range of aperture sizes for extraction'''
        if default:
            self.narrowest = self.obs.narrowest
            self.widest = self.obs.widest
            self.numberofapertures = self.obs.numberofapertures
        else:
            self.speak('Please redefine the aperture sizes.')
            self.speak(' (The old aperture sizes were {}.)'.format(self.extractionwidths))
            self.speak(' ["d" for any of these for defaults]')

            try:
                self.narrowest = np.float(self.input("What is the narrowest aperture?"))
            except ValueError:
                self.narrowest = self.obs.narrowest

            try:
                self.widest = np.float(self.input("What is the widest aperture?"))
            except ValueError:
                self.widest = self.obs.widest

            try:
                self.numberofapertures = int(self.input('How many apertures do you want?'))
            except ValueError:
                self.numberofapertures = self.obs.numberofapertures

        self.extractionwidths = np.linspace(self.narrowest, self.widest, self.numberofapertures)
        self.speak(' The current aperture sizes are {}.'.format(self.extractionwidths))

        # update the plotting, to reflect the new extraction apertures
        try:
            self.updateMasks()
        except AttributeError:
            pass
    """

    def moveCrosshair(self, pressed=None, x=None, y=None):
        """
        Move the crosshair to a new spot,
        either based on a mouse position and a keypress,
        or
        Parameters
        ----------
        pressed : KeyEvent
            the KeyEvent from a button being pressed
        x : float, int, etc...
            the x position to move the vertical crosshair to
        y : float, int, etc...
            the y position to move the horizontal crosshair to
        """

        # pull the values from the mouse click
        if pressed is None:
            self.crosshair["x"] = x
            self.crosshair["y"] = y
        elif pressed.key == "up":
            self.crosshair["y"] += self.dy
        elif pressed.key == "down":
            self.crosshair["y"] -= self.dy
        elif pressed.key == "left":
            self.crosshair["x"] -= self.dx
        elif pressed.key == "right":
            self.crosshair["x"] += self.dx
        else:
            x = pressed.xdata
            y = pressed.ydata

            # update the stored value
            if x is not None:
                self.crosshair["x"] = x
            if y is not None:
                self.crosshair["y"] = y

        # update the position on the 2D plot
        self.plotted["crossy"].set_xdata(self.crosshair["x"])
        self.plotted["crossyextend"].set_xdata(self.crosshair["x"])
        self.plotted["crossx"].set_ydata(self.crosshair["y"])
        self.plotted["crossxextend"].set_ydata(self.crosshair["y"])

        # update the slicey in the 1D plots
        if self.crosshair["x"] != None:
            h, v = self.slicey
            ok = self.ok_slicey
            self.plotted["slicey"].set_data(h[ok], v[ok])
            self.plotted["slicey_bad"].set_data(h[ok == False], v[ok == False])

            # self.plotted['slicey'].set_alpha(*self.alpha_slicey)
            # self.ax['slicey'].set_xlim(0, np.nanmax(self.slicey[0]))

        if self.crosshair["y"] != None:
            h, v = self.slicex
            ok = self.ok_slicex
            self.plotted["slicex"].set_data([h[ok], v[ok]])
            self.plotted["slicex_bad"].set_data([h[ok == False], v[ok == False]])

            # self.plotted['slicex'].set_alpha(*self.alpha_slicex)

            # self.ax['slicex'].set_ylim(0, np.nanmax(self.slicex[1]))

        vmin, vmax = self.plotted["2d"].get_clim()
        self.set_limits(vmin, vmax)

    def movieSlice(
        self,
        direction="y",  # what axis changes in the movie?
        fps=30,  # how many frames per second
        bitrate=1800 * 20,  # bitrate (this seems to work well),
        remake=False,  # should we remake it?
        stride=500,  # how many steps do we take with the movie?
        filename="movie.mp4",  # where should the movie be saved?
        **kw
    ):  # each frame will skip over this many timepoints):
        """Create movie of the spectral cube."""

        self.speak("making a movie!")

        # set up a movie file, which will be populated with plots
        metadata = dict(artist="mosasaurus")

        writer = ani.FFMpegWriter(fps=fps, metadata=metadata, bitrate=bitrate)

        modifiedfilename = filename.replace(".mp4", "+stride{}.mp4".format(stride))

        if (remake == False) and os.path.exists(modifiedfilename):
            self.speak(
                "A movie already exists at {}; re-run with remake=True to do something.".format(
                    filename
                )
            )
            return

        # plot the first spectrum, to set up the plots
        plt.ioff()

        # make the movie
        with writer.saving(self.figure, modifiedfilename, self.figure.get_dpi()):
            if direction == "x":
                for i, x in enumerate(tqdm(self.xaxis[::stride])):
                    # self.speak('   plotting {0}/{1}'.format(i*stride, len(self.xaxis)))
                    self.moveCrosshair(x=x)
                    writer.grab_frame()

            if direction == "y":
                for i, y in enumerate(tqdm(self.yaxis[::stride])):
                    # self.speak('   plotting {0}/{1}'.format(i*stride, len(self.yaxis)))
                    self.moveCrosshair(y=y)
                    writer.grab_frame()

        # finish and display
        self.speak("saved movie to {0}".format(modifiedfilename))
        os.system("open {0}".format(modifiedfilename))


def test_display_works_ok():
    image = np.random.normal(0, 1, (50, 20))
    l = loupe()
    l.setup(image, figsize=(6, 3))
    l.display()


def test():
    fake = createTestImage()
    l = loupe()
    l.setup(fake)
    l.run()
    return fake, l


def testMovie():
    fake = createTestImage()
    l = loupe()
    l.setup(fake)
    l.set_limits(0, np.max(fake))
    l.movieSlice(stride=1)
    return fake, l
