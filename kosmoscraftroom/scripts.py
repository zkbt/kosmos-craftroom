import numpy as np
import pyperclip as pc


class ScriptWriter:
    # these are internal lamps we're allowed to use
    # (TO-DO = implement truss lamps + instructions for talking to Observatory Specialist)
    available_internal_lamps = ["neon", "krypton", "argon", "quartz"]
    available_truss_lamps = ["helium", "neon", "argon", "bright quartz", "dim quartz"]

    # suggested reference 2"-slit exposure times from https://www.apo.nmsu.edu/arc35m/Instruments/KOSMOS/userguide.html#4p2
    suggested_internal_exposure_times = {
        "red": dict(neon=1, argon=2, krypton=1, quartz=5),
        "blue": dict(neon=2, argon=45, krypton=30, quartz=25),
        None: dict(neon=1, argon=1, krypton=1, quartz=1),
    }

    suggested_truss_exposure_times = {
        "red": {"helium": 120, "neon": 60, "argon": 90, "bright quartz": 60},
        "blue": {"helium": 120, "neon": 60, "argon": 120, "bright quartz": 240},
        None: dict(helium=1, neon=1, argon=1, quartz=1),
    }
    reference_slit_width = 2.0

    def __init__(
        self,
        slit_options={"7.1-ctr": 1, "1.18-ctr": 2},
        disperser_options={None: 1, "red": 6, "blue": 3},
        filter1_options={None: 1},
        filter2_options={None: 1, "ND5": 6},
        binning=[2, 2],
        prefix="",
    ):
        """
        Initialize the basic script setup.

        Parameters
        ----------
        slit_options : list
            List of slit names for which calibrations are wanted.
        disperser_options : list
            List of disperser names for which calibrations are wanted.
        binning : list
            List of binning in x + y pixel directions ([xbinning, ybinning])
        prefix : str
            String to put in the front of every filename.
        """

        self.prefix = prefix

        self.lines = []

        print("Assuming dispersers we might use are...")
        for k, v in disperser_options.items():
            print(f" position #{v} = {k}")
        self.disperser_options = disperser_options
        print()

        print("Assuming slit we might use are...")
        for k, v in slit_options.items():
            print(f" position #{v} = {k}")
        self.slit_options = slit_options
        print()

        print("Assuming filter1 we might use are...")
        for k, v in filter1_options.items():
            print(f" position #{v} = {k}")
        self.filter1_options = filter1_options
        print()

        print("Assuming filter1 we might use are...")
        for k, v in filter2_options.items():
            print(f" position #{v} = {k}")
        self.filter2_options = filter2_options
        print()

        self.binning = binning
        print(f"Assuming binning is...")
        print(f" {self.binning_string()}")
        print()

        print("Before running this script, check in the KOSMOS window:")
        print(
            """'
        disperser can be anything
        slit can be anything
        filter1 can be anything
        filter2 can be anything
        calibration stage can be anything
        binning can be anything
        neon/krypton/argon/quartz can be anything
        """
        )
        self.comment(f"set binning to {self.binning_string()}")
        self.say(f"kosmos set rowBin={self.binning[0]} colBin={self.binning[1]}")

    def __repr__(self):
        return f"<KOSMOS script for {len(self.disperser_options)} dispersers + {len(self.slit_options)} slits>"

    def say(self, s=""):
        """
        Add a command to the list.

        Parameters
        ----------
        s : str
            The command to add.
        """
        self.lines.append(s)

    def comment(self, s=""):
        """
        Add a comment to the list.

        Parameters
        ----------
        s : str
            The command to add.
        """
        self.lines.append(f"#{s}")

    def binning_string(self):
        return f"{self.binning[0]}x{self.binning[1]}"

    def guess_slit_width(self, s):
        """
        Guess the slit width, in arcseconds, from the slit name.

        Parameters
        ----------
        s : str
            The string name of the slit.
        """
        return float(s.split("-")[0])

    def take_lamp(
        self,
        lamp="quartz",
        slit=None,
        disperser=None,
        filter1=None,
        filter2=None,
        n=3,
        t=None,
        note="",
    ):
        """
        Take calibrations with internal lamps.

        Parameters
        ----------
        lamp : str
            The name of the internal lamp to turn on.
        slit : str, None
            The name of the slit to use.
        disperser: str, None
            The name of the disperser to use.
        filter1 : str, None
            The name of filter1 to use.
        filter2 : str, None
            The name of filter2 to use.
        n : int
            The number of iterations (per disperser, per slit).
        t : float
            The exposure time to use. If not provided, a guess will
            be made based on the suggested times and the slit width.
        """

        # turn off all lamps but the active one

        # pick the slit
        slit_number = self.slit_options[slit]
        if slit == None:
            slit_factor = 1
        else:
            slit_width = self.guess_slit_width(slit)
            slit_factor = self.reference_slit_width / slit_width

        # pick the disperser
        disperser_number = self.disperser_options[disperser]

        # pick the filters
        filter1_number = self.filter1_options[filter1]
        filter2_number = self.filter2_options[filter2]

        # set the exposure time, if not provided
        if t is None:
            binning_factor = 1 / np.prod(self.binning)
            t = (
                self.suggested_internal_exposure_times[disperser][lamp]
                * binning_factor
                * slit_factor
            )

        # don't allow exposure times shorter than 0.5s
        t = np.maximum(t, 0.5)

        # summarize observation
        self.comment(
            f"internal: {lamp=}, {slit=}, {disperser=}, {filter1=}, {filter2=}, {n=}, {t=}"
        )

        # put in the calibration stage + turn on the lamps
        s = "kosmos set calstage=in"
        for l in self.available_internal_lamps:
            onoff = {True: "on", False: "off"}[l == lamp]
            s += f" {l}={onoff}"
        self.say(s)

        # set up the slit and disperser and any filter(s)
        self.say(
            f"kosmos set slit={slit_number} disperser={disperser_number} filter1={filter1_number} filter2={filter2_number}"
        )
        filename = f"{self.prefix}cals/internal-{lamp}-{slit}-{disperser}"

        if filter1 != None:
            filename += f"-{filter1}"

        if filter2 != None:
            filename += f"-{filter2}"

        if note != "":
            filename += f"-{note}"

        # take the exposure
        self.say(
            f'kosmosExpose flat time={t:.2f} n={n} name="{filename}" seq=nextByDir comment=""'
        )

        # turn off the lamps
        self.comment("turning off internal lamps")
        self.say(f"kosmos set neon=off krypton=off argon=off quartz=off")
        self.say()

    def take_truss_lamp(
        self,
        lamp="quartz",
        slit=None,
        disperser=None,
        filter1=None,
        filter2=None,
        n=3,
        t=None,
        note="",
    ):
        """
        Take calibrations with truss lamps.

        Parameters
        ----------
        lamp : str
            The name of the truss lamp to turn on.
        slit : str, None
            The name of the slit to use.
        disperser: str, None
            The name of the disperser to use.
        filter1 : str, None
            The name of filter1 to use.
        filter2 : str, None
            The name of filter2 to use.
        n : int
            The number of iterations (per disperser, per slit).
        t : float
            The exposure time to use. If not provided, a guess will
            be made based on the suggested times and the slit width.
        note : str
            An extra note to add to the filename.
        """

        # turn off all lamps but the active one

        # pick the slit
        slit_number = self.slit_options[slit]
        if slit == None:
            slit_factor = 1
        else:
            slit_width = self.guess_slit_width(slit)
            slit_factor = self.reference_slit_width / slit_width

        # pick the disperser
        disperser_number = self.disperser_options[disperser]

        # pick the filters
        filter1_number = self.filter1_options[filter1]
        filter2_number = self.filter2_options[filter2]

        # set the exposure time, if not provided
        if t is None:
            binning_factor = 1 / np.prod(self.binning)
            t = (
                self.suggested_truss_exposure_times[disperser][lamp]
                * binning_factor
                * slit_factor
            )

        # don't allow exposure times shorter than 0.5s
        t = np.maximum(t, 0.5)

        # summarize observation
        self.comment(
            f"truss: {lamp=}, {slit=}, {disperser=}, {filter1=}, {filter2=}, {n=}, {t=}"
        )

        # take out the calibration stage + turn on the lamp
        truss_lamp_number = self.available_truss_lamps.index(lamp) + 1
        self.say(f"tlamps on {truss_lamp_number}")
        self.say("kosmos set calstage=out")

        # set up the slit and disperser and any filter(s)
        self.say(
            f"kosmos set slit={slit_number} disperser={disperser_number} filter1={filter1_number} filter2={filter2_number}"
        )
        filename = f"{self.prefix}cals/truss-{lamp}-{slit}-{disperser}-{slit}"
        if filter1 != None:
            filename += f"-{filter1}"

        if filter2 != None:
            filename += f"-{filter2}"

        if note != "":
            filename += f"-{note}"

        # take the exposure
        self.say(
            f'kosmosExpose flat time={t:.2f} n={n} name="{filename}" seq=nextByDir comment=""'
        )

        # turn off the lamps
        self.comment("turning off truss lamps")
        for truss_lamp_number in np.arange(len(self.available_truss_lamps)) + 1:
            self.say(f"tlamps off {truss_lamp_number}")
        self.say()

    def take_science(self, n=3, n_chunk=3, note="", red=10, blue=10):
        """
        Take science exposure.

        Parameters
        ----------
        n : int
            The number of chunks
        n_chunk : int
            The number of iterations at each disperser position before moving

        note : str
            An extra note to add to the filename.
        """

        self.lines = []

        self.say()

        # turn off all lamps but the active one
        self.comment(f"taking {n} science exposures")

        for i in range(n):
            for i_disperser, disperser in enumerate(self.disperser_options):
                disperser_number = self.disperser_options[disperser]
                t = locals()[disperser]

                self.say(f"# disperser={disperser},  iteration {i+1}/{n}")
                self.say(f"kosmos set disperser={disperser_number}")
                filename = f"{self.prefix}sci/{disperser}"
                if note != "":
                    filename += f"-{note}"
                self.say(
                    f'kosmosExpose object time={t:.2f} n={n_chunk} name="{filename}" seq=nextByDir comment=""'
                )
            self.say()

        self.copy()
        self.print()

    def take_bias(self, n=10):
        self.comment(f"taking {n} bias calibrations")
        self.say(f"kosmos set calstage=in neon=off krypton=off argon=off quartz=off")
        filename = f"{self.prefix}cals/bias"
        self.say(f'kosmosExpose bias n={n} name="{filename}" seq=nextByDir comment=""')
        self.say()

    def take_dark(self, t=120, n=10):
        self.comment(f"taking {n} dark calibrations")
        self.say(f"kosmos set calstage=in neon=off krypton=off argon=off quartz=off")
        filename = f"{self.prefix}cals/dark"
        self.say(
            f'kosmosExpose dark n={n} time={t:.2f} name="{filename}" seq=nextByDir comment=""'
        )
        self.say()

    def print(self):
        s = "\n".join(self.lines)
        print(s)

    def copy(self):
        s = "\n".join(self.lines)
        pc.copy(s)
        print(
            """
        Your TUI script has been copied to the clipboard.
        Please paste into the TUI `Run_Commands` window.
        """
        )


"""
    def take_sky(self, exposure_time=1, n=1, note='', what='sky'):
        for i in range(n):
            self.say(f'# iteration {i}, sky')

            for i_disperser, disperser in enumerate(self.disperser_options):
                disperser_number = self.disperser_options[disperser]
                self.say(f'kosmos set disperser={disperser_number}')

                t = exposure_time
                self.say(f'''
kosmos set disperser={disperser_number}
kosmosExpose flat time={t:.2f} n=1 name="{self.binning[0]}x{self.binning[1]}/sky/{disperser}-{what}" seq=nextByDir comment=""'''
                )
            self.say()

    def take(self, exposure_time=1, n=1, object='WASP39', note=''):
        for i in range(n):
            self.say(f'# iteration {i}, sky')

            for disperser in self.disperser_options:
                d = self.disperser_numbers[disperser]
                t = exposure_time
                self.say(f'''
kosmos set disperser={d}
kosmosExpose flat time={t:.2f} n=1 name="{self.binning[0]}x{self.binning[1]}/sky/{disperser}-{object}{note}" seq=nextByDir comment=""'''
                )
            self.say()
"""
