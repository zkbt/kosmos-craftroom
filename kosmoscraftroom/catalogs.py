from astropy.table import Table, vstack, unique
from astropy.coordinates import SkyCoord
from astropy.io.ascii import read
import astropy.units as u

import numpy as np


class TUICatalog:
    """
    Tool to make a TUI catalog, as outlined in this documentation:
    https://www.apo.nmsu.edu/35m_operations/TUI/Telescope/UserCatalogs.html

    """

    def __init__(self, name="?"):
        """
        Initialize a catalog
        """
        self.name = name
        self.table = []

    def __repr__(self):
        return f"<'{self.name}' TUICatalog ({len(self.table)} targets)>"

    def from_table(self, table, remove_duplicates=True, sort=True):
        """
        Initialize from a table.
        """
        self.table = table
        if remove_duplicates:
            self.remove_duplicates()
        if sort:
            self.sort()

    def from_TUI(self, filename, names=["name", "ra", "dec"]):
        t = read(filename, names=names)
        table = Table(
            dict(
                names=t["name"],
                sky_coordinates=SkyCoord(
                    ra=t["ra"],
                    dec=t["dec"],
                    unit=(u.hourangle, u.deg),  # TO-DO make sure this is more general!!
                ),
                category=[self.name for i in range(len(t))],
            )
        )
        self.from_table(table)

    def from_exoatlas(self, pop):
        """
        Initialize from an exoplanet-atlas population.
        """
        self._exoatlas_pop = pop
        table = Table(
            dict(
                names=pop.standard["hostname"],
                sky_coordinates=SkyCoord(pop.standard["ra"], pop.standard["dec"]),
                G=pop.standard["gaiamag"],
                distance=pop.standard["distance"],
                category=[self.name for i in range(len(pop.standard))],
            )
        )
        self.from_table(table)

    def remove_duplicates(self):
        self.table = unique(self.table, keys=["names"])

    def sort(self):
        self.table = self.table[np.argsort(self.table["sky_coordinates"].ra)]

    def make_three_columns(self, row):
        return f"""{row['names'].replace(' ', ''):<20} {row['sky_coordinates'].to_string('hmsdms', sep=':', precision=1)}"""

    def make_tui_columns(self, row):
        keywords = []
        return "; ".join(keywords)

    def make_human_columns(self, row):
        keywords = [
            f"{row['category']}",
            f"d={row['distance']:.1f}pc",
            f"G={row['G']:.2f}",
        ]
        return ", ".join(keywords)

    def to_TUI(self, parallactic=True, output=False, color="Black"):
        """
        Write this catalog out to a TUI-friendly format.

        The catalog will be written out to a file set by the catalogs 'name' attribute,
        in a format that can be loaded into TUI via the Slew > Catalogs button.

        Parameters
        ----------
        parallactic : bool
            Should we rotate the slit to be along the parallactic angle?
            Doing so will minimize the effects of differental refraction
            through Earth's atmosphere on the amount light that enters
            the slit.
        output : bool
            Should we print the catalog to the screen?
        """
        filename = f"{self.name}.tui"
        if parallactic:
            preamble = f"CSys=ICRS; RotType=Horizon; RotAng=90; DispColor={color}"
        else:
            preamble = f"CSys=ICRS; RotType=Object; RotAng=0; DispColor={color}"

        lines = [
            f"""{self.make_three_columns(row)}   {self.make_tui_columns(row)}\n"""
            for row in self.table
        ]

        with open(filename, "w") as f:
            f.write(preamble)
            f.write("\n" * 3)
            f.writelines(lines)

        if output:
            with open(filename, "r") as f:
                print(f.read())

        print(f"{self} has been saved to {filename}")

    def to_human_friendly(self, output=False):
        """
        Write this catalog out to a human-friendly format.

        The catalog will be written with the same initial columns as a TUI catalog,
        but with additional information that might help in observation planning,
        such as the category of observing program, the distance, the magnitude.
        """
        filename = f"{self.name}.txt"
        lines = [
            f"""{self.make_three_columns(row)}   {self.make_human_columns(row)}\n"""
            for row in self.table
        ]

        with open(filename, "w") as f:
            f.writelines(lines)

        if output:
            with open(filename, "r") as f:
                print(f.read())

        print(
            f"Human-friendly catalog ({len(self.table)} targets) has been saved to {filename}"
        )

    def __add__(self, other):
        """
        Merge two catalogs together.
        """
        new = TUICatalog(f"{self.name}+{other.name}")
        new.table = vstack([self.table, other.table])
        new.sort()
        return new


#    def to_human(self)G={row['G']:>5.2f}
