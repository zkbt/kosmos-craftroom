"""
Generate interactive plots, so you can (for example)
use the location of a mouse-click in code.

This is basically a wrapper for matplotlib's event handling
capabilities, but it's easier for me to remember it in the
way I have it organized here. You create a multipanel plot
using the basic gridspec specification, and then you can connect
to event tracking within any of those panels.
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display


# turn off default key mappings for matplotlib
# otherwise they may overlap with your custom ones
# for k in plt.rcParams.keys():
#    if "keymap" in k:
#        plt.rcParams[k] = ""


class iplot:
    """
    An interactive (multipanel) plot, built on matplotlib GridSpec.
    """

    def __init__(self, n_rows=1, n_cols=1, figsize=(7, 3), **kwargs):
        """
        Initialize an interactive multi-panel plot.

        Provide any keywords allowed for matplotlib.gridspec.GridSpec
        to create a multipanel plot that will be easy to connect
        for interactive behavior.

        Parameters
        ----------
        *args : tuple
                All positional arguments will be passed to GridSpec.
        **kwargs : dict
                All keyword arguments will be passed to GridSpec.
        """
        # create a space for displaying outputs
        self.figure_output = widgets.Output()
        self.text_output = widgets.Output()
        self.layout_output = widgets.AppLayout(
            header=None,
            left_sidebar=None,
            center=self.figure_output,  # self.figure.canvas,
            right_sidebar=self.text_output,
            footer=None,
        )

        # with self.figure_output:
        # create the figure
        # with plt.ioff():
        with self.figure_output:
            self.figure = plt.figure(figsize=figsize, constrained_layout=False)
            # self.figure.canvas.toolbar_position = "bottom"

        # create the gridspec object
        self.gs = gridspec.GridSpec(n_rows, n_cols, figure=self.figure, **kwargs)

        # create an empty dictionary to store axes
        self.axes = {}

    def display(self):
        """ """

        plt.show()
        display(self.layout_output)

    def __repr__(self):
        return self.layout_output
        # return f"Interactive{self.gs}"

    def speak(self, x):
        """
        Print text into the display.
        """
        with self.text_output:
            self.text_output.clear_output()
            print(x)

    def subplot(self, row=0, col=0, rowspan=1, colspan=1, name=None, **kwargs):
        """
        Create a plotting Axes at a particular location.

        Parameters
        ----------
        row : int
            The row in this GridSpec object.
        col : int
            The column in this GridSpec object.
        rowspan : int
            The number of rows to span in this GridSpec object.
        colspan : int
            The number of columns to span in this GridSpec object.
        name : str
            A name to help keep track of this interactive panel.
        """

        # do all plot setup within figure display output
        with self.figure_output:
            # create the Axes with the right position and shape
            ax = plt.subplot(
                self.gs[row : row + rowspan, col : col + colspan], **kwargs
            )

            # TO-DO check this doesn't ruin things!
            self.figure = plt.gcf()

            # make sure that a name is assigned
            if name == None:
                name = f"ax{len(self.axes)}"

            # store Axes in dictionary
            self.axes[name] = ax

        return ax

    def onKeyPress(self, event):
        """when a keyboard button is pressed, record the event"""
        self.keypressed = event
        if self.whenpressed is not None:
            self.whenpressed(self.keypressed)
        self.stop()

    def onKeyRelease(self, event):
        """when a keyboard button is released, stop the loop"""
        self.keyreleased = event

        try:
            assert event.key.lower() != "alt"
            assert event.key.lower() != "control"
            self.stop()
        except AssertionError:
            pass

    def onClick(self, event):
        if event.xdata is None:
            print("Hmmm, that wasn't a very nice click. Could you please try again?")
            return
        self.lastMouseClick = event

        self.mouseClicks.append(event)
        self.remainingClicks -= 1
        if self.remainingClicks <= 0:
            self.stop()

    def getMouseClicks(self, n=1):
        # say what's happening
        self.speak(f"waiting for {n} mouse click(s).")

        # set the countdown for the number of clicks
        self.remainingClicks = n
        self.mouseClicks = []

        # start the event handling loop for mouse button releases
        self.cids = [self.watchfor("button_release_event", self.onClick)]
        self.startloop()

        # return a list of mouse clicks, which the loop should have generated
        return self.mouseClicks

    def getKeyboard(self, whenpressed=None):
        """wait for a keyboard press and release.

        whenpressed can be a function that will be called on
        the key press (before the key release); it must be
        able to take a KeyEvent as an argument"""

        # warn what's happening
        print("waiting for a key to be pressed and released")
        # start the loop
        self.cids = [self.watchfor("key_release_event", self.onKeyRelease)]

        # start the loop (it'll end when key is pressed)
        self.startloop()
        print(
            '"{}" pressed at {}, {}'.format(
                self.keyreleased.key, self.keyreleased.xdata, self.keyreleased.ydata
            )
        )

        # return the key that was pressed
        return self.keyreleased

    def watchfor(self, *args):
        """This is a shortcut for mpl_connect.

        For example,
                        self.watchfor('key_press_event', self.onKeyPress)
        will set up a connection that will cause the function
        self.onKeyPress to be called (and passed a KeyEvent) when
        a key_press_event occurs."""
        return self.figure.canvas.mpl_connect(*args)

    def stopwatching(self, cids):
        """shortcut to stop watching for particular events"""
        for cid in cids:
            self.figure.canvas.mpl_disconnect(cid)

    def startloop(self):
        """start the event loop"""
        self.figure.canvas.start_event_loop(0)

    def stoploop(self):
        """stop the event loop"""
        self.figure.canvas.stop_event_loop()

    def stop(self):
        """stop the loop, and disconnect watchers"""
        self.stoploop()
        self.stopwatching(self.cids)


def test():
    import numpy as np

    i = iplot(1, 1)
    i.subplot(0, 0)
    a = i.axes["ax0"]
    a.plot(np.random.normal(0, 1, 10))
    plt.draw()
    key = i.getKeyboard()
    print(key)
    return key


# print(i.getKeyboard(2))
