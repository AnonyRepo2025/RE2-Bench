import warnings
import numpy as np
from astropy import units as u
from astropy.coordinates import SpectralCoord, Galactic, ICRS
from astropy.coordinates.spectral_coordinate import update_differentials_to_match, attach_zero_velocities
from astropy.utils.exceptions import AstropyUserWarning
from astropy.constants import c
from .low_level_api import BaseLowLevelWCS
from .high_level_api import HighLevelWCSMixin
from .wrappers import SlicedLowLevelWCS
from astropy.wcs.wcs import NoConvergence
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time.formats import FITS_DEPRECATED_SCALES
from astropy.time import Time, TimeDelta

__all__ = ['custom_ctype_to_ucd_mapping', 'SlicedFITSWCS', 'FITSWCSAPIMixin']
C_SI = c.si.value
VELOCITY_FRAMES = {
    'GEOCENT': 'gcrs',
    'BARYCENT': 'icrs',
    'HELIOCENT': 'hcrs',
    'LSRK': 'lsrk',
    'LSRD': 'lsrd'
}
VELOCITY_FRAMES['GALACTOC'] = Galactic(u=0 * u.km, v=0 * u.km, w=0 * u.km,
                                       U=0 * u.km / u.s, V=-220 * u.km / u.s, W=0 * u.km / u.s,
                                       representation_type='cartesian',
                                       differential_type='cartesian')
VELOCITY_FRAMES['LOCALGRP'] = Galactic(u=0 * u.km, v=0 * u.km, w=0 * u.km,
                                       U=0 * u.km / u.s, V=-300 * u.km / u.s, W=0 * u.km / u.s,
                                       representation_type='cartesian',
                                       differential_type='cartesian')
VELOCITY_FRAMES['CMBDIPOL'] = Galactic(l=263.85 * u.deg, b=48.25 * u.deg, distance=0 * u.km,
                                       radial_velocity=-(3.346e-3 / 2.725 * c).to(u.km/u.s))
CTYPE_TO_UCD1 = {

    # Celestial coordinates
    'RA': 'pos.eq.ra',
    'DEC': 'pos.eq.dec',
    'GLON': 'pos.galactic.lon',
    'GLAT': 'pos.galactic.lat',
    'ELON': 'pos.ecliptic.lon',
    'ELAT': 'pos.ecliptic.lat',
    'TLON': 'pos.bodyrc.lon',
    'TLAT': 'pos.bodyrc.lat',
    'HPLT': 'custom:pos.helioprojective.lat',
    'HPLN': 'custom:pos.helioprojective.lon',
    'HPRZ': 'custom:pos.helioprojective.z',
    'HGLN': 'custom:pos.heliographic.stonyhurst.lon',
    'HGLT': 'custom:pos.heliographic.stonyhurst.lat',
    'CRLN': 'custom:pos.heliographic.carrington.lon',
    'CRLT': 'custom:pos.heliographic.carrington.lat',
    'SOLX': 'custom:pos.heliocentric.x',
    'SOLY': 'custom:pos.heliocentric.y',
    'SOLZ': 'custom:pos.heliocentric.z',

    # Spectral coordinates (WCS paper 3)
    'FREQ': 'em.freq',  # Frequency
    'ENER': 'em.energy',  # Energy
    'WAVN': 'em.wavenumber',  # Wavenumber
    'WAVE': 'em.wl',  # Vacuum wavelength
    'VRAD': 'spect.dopplerVeloc.radio',  # Radio velocity
    'VOPT': 'spect.dopplerVeloc.opt',  # Optical velocity
    'ZOPT': 'src.redshift',  # Redshift
    'AWAV': 'em.wl',  # Air wavelength
    'VELO': 'spect.dopplerVeloc',  # Apparent radial velocity
    'BETA': 'custom:spect.doplerVeloc.beta',  # Beta factor (v/c)
    'STOKES': 'phys.polarization.stokes',  # STOKES parameters

    # Time coordinates (https://www.aanda.org/articles/aa/pdf/2015/02/aa24653-14.pdf)
    'TIME': 'time',
    'TAI': 'time',
    'TT': 'time',
    'TDT': 'time',
    'ET': 'time',
    'IAT': 'time',
    'UT1': 'time',
    'UTC': 'time',
    'GMT': 'time',
    'GPS': 'time',
    'TCG': 'time',
    'TCB': 'time',
    'TDB': 'time',
    'LOCAL': 'time',

    # Distance coordinates
    'DIST': 'pos.distance',
    'DSUN': 'custom:pos.distance.sunToObserver'

    # UT() and TT() are handled separately in world_axis_physical_types

}
CTYPE_TO_UCD1_CUSTOM = []

class FITSWCSAPIMixin(BaseLowLevelWCS, HighLevelWCSMixin):
    def world_to_pixel_values(self, *world_arrays):
        # avoid circular import
        from astropy.wcs.wcs import NoConvergence
        try:
            pixel = self.all_world2pix(*world_arrays, 0)
        except NoConvergence as e:
            warnings.warn(str(e))
            # use best_solution contained in the exception and format the same
            # way as all_world2pix does (using _array_converter)
            pixel = self._array_converter(lambda *args: e.best_solution,
                                         'input', *world_arrays, 0)

        return pixel[0] if self.pixel_n_dim == 1 else tuple(pixel)