
# LEVEL-1C

from l1c.src.initL1c import initL1c
from common.io.writeToa import writeToa, readToa
from common.io.readGeodetic import readGeodetic, getCorners
import mgrs
import numpy as np
from scipy.interpolate import bisplrep, bisplev
import matplotlib.pyplot as plt
from common.io.l1cProduct import writeL1c
from matplotlib import cm
class l1c(initL1c):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1C Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            toa = readToa(
                self.l1bdir,
                self.globalConfig.l1b_toa + band + '.nc'
            )

            lat, lon = readGeodetic(
                self.gmdir,
                self.globalConfig.gm_geoloc
            )

            self.checkSize(lat, toa)

            lat_l1c, lon_l1c, toa_l1c = self.l1cProjtoa(
                lat,
                lon,
                toa,
                band
            )

            self.plotL1bVsL1cGrid(
                lat,
                lon,
                lat_l1c,
                lon_l1c,
                band
            )

            writeL1c(
                self.outdir,
                self.globalConfig.l1c_toa + band,
                lat_l1c,
                lon_l1c,
                toa_l1c
            )

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1C Module!")


    def l1cProjtoa(self, lat, lon, toa, band):

        tck = bisplrep(
            lat.flatten(),
            lon.flatten(),
            toa.flatten()
        )

        m = mgrs.MGRS()

        mgrs_tiles = set([])

        for ii in range(toa.shape[0]):
            for jj in range(toa.shape[1]):

                auxi = m.toMGRS(
                    lat[ii, jj],
                    lon[ii, jj],
                    MGRSPrecision=self.l1cConfig.mgrs_tile_precision
                )

                mgrs_tiles.add(auxi)

        mgrs_tiles = list(mgrs_tiles)

        toa_l1c = np.zeros(len(mgrs_tiles))
        lat_l1c = np.zeros(len(mgrs_tiles))
        lon_l1c = np.zeros(len(mgrs_tiles))

        for ii in range(len(mgrs_tiles)):

            lat_l1c[ii], lon_l1c[ii] = m.toLatLon(
                mgrs_tiles[ii]
            )

            toa_l1c[ii] = bisplev(
                lat_l1c[ii],
                lon_l1c[ii],
                tck
            )

        return lat_l1c, lon_l1c, toa_l1c


    def checkSize(self, lat, toa):

        if lat.shape != toa.shape:

            raise ValueError(
                "Latitude and TOA dimensions do not match: "
                + str(lat.shape)
                + " vs "
                + str(toa.shape)
            )


    def plotL1bVsL1cGrid(
        self,
        lat,
        lon,
        lat_l1c,
        lon_l1c,
        band
    ):

        plt.figure(
            figsize=(10, 8)
        )

        # L1B grid - red
        plt.scatter(
            lon.flatten(),
            lat.flatten(),
            s=6,
            c="red",
            label="L1B grid"
        )

        # L1C grid - blue
        plt.scatter(
            lon_l1c,
            lat_l1c,
            s=10,
            c="blue",
            label="L1C MGRS grid"
        )

        plt.title(
            "L1B grid vs L1C grid - " + band
        )

        plt.xlabel(
            "Longitude [deg]"
        )

        plt.ylabel(
            "Latitude [deg]"
        )

        plt.grid(True)

        plt.legend()

        plt.tight_layout()

        filename = (
            self.outdir
            + "/l1b_vs_l1c_grid_"
            + band
            + ".png"
        )

        plt.savefig(
            filename,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()