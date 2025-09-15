def pixel_n_dim(self):
    return self.naxis

def pixel_to_world_values(self, *pixel_arrays):
    world = self.all_pix2world(*pixel_arrays, 0)
    return world[0] if self.world_n_dim == 1 else tuple(world)

def all_pix2world(self, *args, **kwargs):
    return self._array_converter(self._all_pix2world, 'output', *args, **kwargs)

def _array_converter(self, func, sky, *args, ra_dec_order=False):

    def _return_list_of_arrays(axes, origin):
        if any([x.size == 0 for x in axes]):
            return axes
        try:
            axes = np.broadcast_arrays(*axes)
        except ValueError:
            raise ValueError('Coordinate arrays are not broadcastable to each other')
        xy = np.hstack([x.reshape((x.size, 1)) for x in axes])
        if ra_dec_order and sky == 'input':
            xy = self._denormalize_sky(xy)
        output = func(xy, origin)
        if ra_dec_order and sky == 'output':
            output = self._normalize_sky(output)
            return (output[:, 0].reshape(axes[0].shape), output[:, 1].reshape(axes[0].shape))
        return [output[:, i].reshape(axes[0].shape) for i in range(output.shape[1])]

    def _return_single_array(xy, origin):
        if xy.shape[-1] != self.naxis:
            raise ValueError('When providing two arguments, the array must be of shape (N, {})'.format(self.naxis))
        if 0 in xy.shape:
            return xy
        if ra_dec_order and sky == 'input':
            xy = self._denormalize_sky(xy)
        result = func(xy, origin)
        if ra_dec_order and sky == 'output':
            result = self._normalize_sky(result)
        return result
    if len(args) == 2:
        try:
            xy, origin = args
            xy = np.asarray(xy)
            origin = int(origin)
        except Exception:
            raise TypeError('When providing two arguments, they must be (coords[N][{}], origin)'.format(self.naxis))
        if xy.shape == () or len(xy.shape) == 1:
            return _return_list_of_arrays([xy], origin)
        return _return_single_array(xy, origin)
    elif len(args) == self.naxis + 1:
        axes = args[:-1]
        origin = args[-1]
        try:
            axes = [np.asarray(x) for x in axes]
            origin = int(origin)
        except Exception:
            raise TypeError('When providing more than two arguments, they must be ' + 'a 1-D array for each axis, followed by an origin.')
        return _return_list_of_arrays(axes, origin)
    raise TypeError('WCS projection has {0} dimensions, so expected 2 (an Nx{0} array and the origin argument) or {1} arguments (the position in each dimension, and the origin argument). Instead, {2} arguments were given.'.format(self.naxis, self.naxis + 1, len(args)))

def _return_list_of_arrays(axes, origin):
    if any([x.size == 0 for x in axes]):
        return axes
    try:
        axes = np.broadcast_arrays(*axes)
    except ValueError:
        raise ValueError('Coordinate arrays are not broadcastable to each other')
    xy = np.hstack([x.reshape((x.size, 1)) for x in axes])
    if ra_dec_order and sky == 'input':
        xy = self._denormalize_sky(xy)
    output = func(xy, origin)
    if ra_dec_order and sky == 'output':
        output = self._normalize_sky(output)
        return (output[:, 0].reshape(axes[0].shape), output[:, 1].reshape(axes[0].shape))
    return [output[:, i].reshape(axes[0].shape) for i in range(output.shape[1])]

def world_n_dim(self):
    return len(self.wcs.ctype)

def world_to_pixel_values(self, *world_arrays):
    from astropy.wcs.wcs import NoConvergence
    try:
        pixel = self.all_world2pix(*world_arrays, 0)
    except NoConvergence as e:
        warnings.warn(str(e))
        pixel = self._array_converter(lambda *args: e.best_solution, 'input', *world_arrays, 0)
    return pixel[0] if self.pixel_n_dim == 1 else tuple(pixel)

def wrapper(func):
    return cls(func, lazy=lazy)

def all_world2pix(self, *args, tolerance=0.0001, maxiter=20, adaptive=False, detect_divergence=True, quiet=False, **kwargs):
    if self.wcs is None:
        raise ValueError('No basic WCS settings were created.')
    return self._array_converter(lambda *args, **kwargs: self._all_world2pix(*args, tolerance=tolerance, maxiter=maxiter, adaptive=adaptive, detect_divergence=detect_divergence, quiet=quiet), 'input', *args, **kwargs)

def _all_world2pix(self, world, origin, tolerance, maxiter, adaptive, detect_divergence, quiet):
    pix0 = self.wcs_world2pix(world, origin)
    if not self.has_distortion:
        return pix0
    pix = pix0.copy()
    dpix = self.pix2foc(pix, origin) - pix0
    pix -= dpix
    dn = np.sum(dpix * dpix, axis=1)
    dnprev = dn.copy()
    tol2 = tolerance ** 2
    k = 1
    ind = None
    inddiv = None
    old_invalid = np.geterr()['invalid']
    old_over = np.geterr()['over']
    np.seterr(invalid='ignore', over='ignore')
    if not adaptive:
        while np.nanmax(dn) >= tol2 and k < maxiter:
            dpix = self.pix2foc(pix, origin) - pix0
            dn = np.sum(dpix * dpix, axis=1)
            if detect_divergence:
                divergent = dn >= dnprev
                if np.any(divergent):
                    slowconv = dn >= tol2
                    inddiv, = np.where(divergent & slowconv)
                    if inddiv.shape[0] > 0:
                        conv = dn < dnprev
                        iconv = np.where(conv)
                        dpixgood = dpix[iconv]
                        pix[iconv] -= dpixgood
                        dpix[iconv] = dpixgood
                        ind, = np.where(slowconv & conv)
                        pix0 = pix0[ind]
                        dnprev[ind] = dn[ind]
                        k += 1
                        adaptive = True
                        break
                dnprev = dn
            pix -= dpix
            k += 1
    if adaptive:
        if ind is None:
            ind, = np.where(np.isfinite(pix).all(axis=1))
            pix0 = pix0[ind]
        while ind.shape[0] > 0 and k < maxiter:
            dpixnew = self.pix2foc(pix[ind], origin) - pix0
            dnnew = np.sum(np.square(dpixnew), axis=1)
            dnprev[ind] = dn[ind].copy()
            dn[ind] = dnnew
            if detect_divergence:
                conv = dnnew < dnprev[ind]
                iconv = np.where(conv)
                iiconv = ind[iconv]
                dpixgood = dpixnew[iconv]
                pix[iiconv] -= dpixgood
                dpix[iiconv] = dpixgood
                subind, = np.where((dnnew >= tol2) & conv)
            else:
                pix[ind] -= dpixnew
                dpix[ind] = dpixnew
                subind, = np.where(dnnew >= tol2)
            ind = ind[subind]
            pix0 = pix0[subind]
            k += 1
    invalid = ~np.all(np.isfinite(pix), axis=1) & np.all(np.isfinite(world), axis=1)
    inddiv, = np.where((dn >= tol2) & (dn >= dnprev) | invalid)
    if inddiv.shape[0] == 0:
        inddiv = None
    if k >= maxiter:
        ind, = np.where((dn >= tol2) & (dn < dnprev) & ~invalid)
        if ind.shape[0] == 0:
            ind = None
    else:
        ind = None
    np.seterr(invalid=old_invalid, over=old_over)
    if (ind is not None or inddiv is not None) and (not quiet):
        if inddiv is None:
            raise NoConvergence("'WCS.all_world2pix' failed to converge to the requested accuracy after {:d} iterations.".format(k), best_solution=pix, accuracy=np.abs(dpix), niter=k, slow_conv=ind, divergent=None)
        else:
            raise NoConvergence("'WCS.all_world2pix' failed to converge to the requested accuracy.\nAfter {:d} iterations, the solution is diverging at least for one input point.".format(k), best_solution=pix, accuracy=np.abs(dpix), niter=k, slow_conv=ind, divergent=inddiv)
    return pix

def wcs_world2pix(self, *args, **kwargs):
    if self.wcs is None:
        raise ValueError('No basic WCS settings were created.')
    return self._array_converter(lambda xy, o: self.wcs.s2p(xy, o)['pixcrd'], 'input', *args, **kwargs)

def _return_single_array(xy, origin):
    if xy.shape[-1] != self.naxis:
        raise ValueError('When providing two arguments, the array must be of shape (N, {})'.format(self.naxis))
    if 0 in xy.shape:
        return xy
    if ra_dec_order and sky == 'input':
        xy = self._denormalize_sky(xy)
    result = func(xy, origin)
    if ra_dec_order and sky == 'output':
        result = self._normalize_sky(result)
    return result

def has_distortion(self):
    return self.sip is not None or self.cpdis1 is not None or self.cpdis2 is not None or (self.det2im1 is not None and self.det2im2 is not None)



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
        sliced_out_world_coords = self._pixel_to_world_values_all(*[0] * len(self._pixel_keep))
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
        if isinstance(pixel_arrays, np.ndarray) and (not pixel_arrays.shape):
            return pixel_arrays
        pixel = tuple((pixel_arrays[ip] for ip in self._pixel_keep))
        if self.pixel_n_dim == 1 and self._wcs.pixel_n_dim > 1:
            pixel = pixel[0]
        return pixel