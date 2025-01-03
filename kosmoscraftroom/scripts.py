import numpy as np
import pyperclip as pc


class ScriptWriter:
    # these are internal lamps we're allowed to use
    # (TO-DO = implement truss lamps + instructions for talking to Observatory Specialist)
    available_internal_lamps = ["neon", "krypton", "argon", "quartz"]
    available_truss_lamps = ["helium", "neon", "argon", "bright quartz", "dim quartz"]

    # suggested reference 2" exposure times from https://www.apo.nmsu.edu/arc35m/Instruments/KOSMOS/userguide.html#4p2
    suggested_internal_exposure_times = dict(
        red=dict(neon=1, argon=2, krypton=1, quartz=15),
        blue=dict(neon=2, argon=45, krypton=30, quartz=80),
    )
    reference_slit_width = 2.0

    def __init__(
        self,
        slits={"7.1-ctr": 1, "1.18-ctr": 2},
        dispersers={"red": 6, "blue": 3},
        binning=[2, 2],
        prefix="",
    ):
        """
        Initialize the basic script setup.

        Parameters
        ----------
        slits : list
            List of slit names for which calibrations are wanted.
        dispersers : list
            List of disperser names for which calibrations are wanted.
        binning : list
            List of binning in x + y pixel directions ([xbinning, ybinning])
        prefix : str
            String to put in the front of every filename.
        """

        self.prefix = prefix

        self.lines = []

        print("Assuming dispersers to calibrate are...")
        for k, v in dispersers.items():
            print(f" position #{v} = {k}")
        self.dispersers = dispersers
        print()

        print("Assuming slits to calibrate are...")
        for k, v in slits.items():
            print(f" position #{v} = {k}")
        self.slits = slits
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
        filter1 must be empty
        filter2 must be empty
        calibration stage can be anything
        binning can be anything
        neon/krypton/argon/quartz can be anything
        """
        )
        self.comment(f"set binning to {self.binning_string()}")
        self.say(f"kosmos set rowBin={self.binning[0]} colBin={self.binning[1]}")

    def __repr__(self):
        return f"<KOSMOS script for {len(self.dispersers)} dispersers + {len(self.slits)} slits>"

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

    def take_lamps(self, lamp, n=3, note=""):
        """
        Take calibrations with lamps.

        Parameters
        ----------
        lamp : str
            The lamp to turn on.
        n : int
            The number of iterations (per disperser, per slit).
        note : str
            An extra note to add to the filename.
        """

        # turn off all lamps but the active one
        self.comment(f"taking {n} {lamp} calibrations")

        s = "kosmos set calstage=in"
        for l in self.available_internal_lamps:
            onoff = {True: "on", False: "off"}[l == lamp]
            s += f" {l}={onoff}"
        self.say(s)

        binning_factor = 1 / np.prod(self.binning)

        for i_slit, slit_name in enumerate(self.slits):
            slit_number = self.slits[slit_name]
            slit_width = self.guess_slit_width(slit_name)
            slit_factor = self.reference_slit_width / slit_width

            for i_disperser, disperser_name in enumerate(self.dispersers):
                disperser_number = self.dispersers[disperser_name]
                t = (
                    self.suggested_internal_exposure_times[disperser_name][lamp]
                    * binning_factor
                    * slit_factor
                )
                t = np.maximum(t, 1.0)
                self.say(
                    f"# lamp {lamp}, slit={slit_name} ({i_slit+1}/{len(self.slits)}), disperser={disperser_name} ({i_disperser+1}/{len(self.dispersers)}),  {n} iterations"
                )
                self.say(f"kosmos set slit={slit_number} disperser={disperser_number}")
                filename = f"{self.prefix}{self.binning_string()}/cals/{disperser_name}-{slit_name}-{lamp}"
                if note != "":
                    filename += f"-{note}"
                self.say(
                    f'kosmosExpose flat time={t:.2f} n={n} name="{filename}" seq=nextByDir comment=""'
                )

        self.comment("turning off lamps")
        self.say(f"kosmos set calstage=in neon=off krypton=off argon=off quartz=off")
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
            for i_disperser, disperser_name in enumerate(self.dispersers):
                disperser_number = self.dispersers[disperser_name]
                t = locals()[disperser_name]

                self.say(f"# disperser={disperser_name},  iteration {i+1}/{n}")
                self.say(f"kosmos set disperser={disperser_number}")
                filename = f"{self.prefix}{self.binning_string()}/sci/{disperser_name}"
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
        filename = f"{self.prefix}{self.binning_string()}/cals/bias"
        self.say(f'kosmosExpose bias n={n} name="{filename}" seq=nextByDir comment=""')
        self.say()

    def take_dark(self, t=120, n=10):
        self.comment(f"taking {n} dark calibrations")
        self.say(f"kosmos set calstage=in neon=off krypton=off argon=off quartz=off")
        filename = f"{self.prefix}{self.binning_string()}/cals/dark"
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

            for i_disperser, disperser_name in enumerate(self.dispersers):
                disperser_number = self.dispersers[disperser_name]
                self.say(f'kosmos set disperser={disperser_number}')

                t = exposure_time
                self.say(f'''
kosmos set disperser={disperser_number}
kosmosExpose flat time={t:.2f} n=1 name="{self.binning[0]}x{self.binning[1]}/sky/{disperser_name}-{what}" seq=nextByDir comment=""'''
                )
            self.say()

    def take(self, exposure_time=1, n=1, object='WASP39', note=''):
        for i in range(n):
            self.say(f'# iteration {i}, sky')

            for disperser in self.dispersers:
                d = self.disperser_numbers[disperser]
                t = exposure_time
                self.say(f'''
kosmos set disperser={d}
kosmosExpose flat time={t:.2f} n=1 name="{self.binning[0]}x{self.binning[1]}/sky/{disperser}-{object}{note}" seq=nextByDir comment=""'''
                )
            self.say()
"""
