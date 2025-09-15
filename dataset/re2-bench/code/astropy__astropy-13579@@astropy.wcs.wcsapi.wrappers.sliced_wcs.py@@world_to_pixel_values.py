import numbers
from collections import defaultdict
import numpy as np
from astropy.utils import isiterable
from astropy.utils.decorators import lazyproperty
from ..low_level_api import BaseLowLevelWCS
from .base import BaseWCSWrapper

__all__ = ['sanitize_slices', 'SlicedLowLevelWCS']

class SlicedLowLevelWCS(BaseWCSWrapper):
    @property
    def pixel_n_dim(self):
        return len(self._pixel_keep)
    def _pixel_to_world_values_all(self, *pixel_arrays):
        pixel_arrays = tuple(map(np.asanyarray, pixel_arrays))
        pixel_arrays_new = []
        ipix_curr = -1
        for ipix in range(self._wcs.pixel_n_dim):
            if isinstance(self._slices_pixel[ipix], numbers.Integral):
                pixel_arrays_new.append(self._slices_pixel[ipix])
            else:
                ipix_curr += 1
                if self._slices_pixel[ipix].start is not None:
                    pixel_arrays_new.append(pixel_arrays[ipix_curr] + self._slices_pixel[ipix].start)
                else:
                    pixel_arrays_new.append(pixel_arrays[ipix_curr])

        pixel_arrays_new = np.broadcast_arrays(*pixel_arrays_new)
        return self._wcs.pixel_to_world_values(*pixel_arrays_new)
    def world_to_pixel_values(self, *world_arrays):
        sliced_out_world_coords = self._pixel_to_world_values_all(*[0]*len(self._pixel_keep))

        world_arrays = tuple(map(np.asanyarray, world_arrays))
        world_arrays_new = []
        iworld_curr = -1
        for iworld in range(self._wcs.world_n_dim):
            if iworld in self._world_keep:
                iworld_curr += 1
                world_arrays_new.append(world_arrays[iworld_curr])
            else:
                world_arrays_new.append(sliced_out_world_coords[iworld])

        world_arrays_new = np.broadcast_arrays(*world_arrays_new)
        pixel_arrays = list(self._wcs.world_to_pixel_values(*world_arrays_new))

        for ipixel in range(self._wcs.pixel_n_dim):
            if isinstance(self._slices_pixel[ipixel], slice) and self._slices_pixel[ipixel].start is not None:
                pixel_arrays[ipixel] -= self._slices_pixel[ipixel].start

        # Detect the case of a length 0 array
        if isinstance(pixel_arrays, np.ndarray) and not pixel_arrays.shape:
            return pixel_arrays
        pixel = tuple(pixel_arrays[ip] for ip in self._pixel_keep)
        if self.pixel_n_dim == 1 and self._wcs.pixel_n_dim > 1:
            pixel = pixel[0]
        return pixel