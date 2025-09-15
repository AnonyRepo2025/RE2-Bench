import builtins
import numpy as np
from astropy.utils.shapes import NDArrayShapeMethods
from astropy.utils.data_info import ParentDtypeInfo
from .function_helpers import (MASKED_SAFE_FUNCTIONS,
                               APPLY_TO_BOTH_FUNCTIONS,
                               DISPATCHED_FUNCTIONS,
                               UNSUPPORTED_FUNCTIONS)

__all__ = ['Masked', 'MaskedNDArray']
get__doc__ = """Masked version of {0.__name__}.

Except for the ability to pass in a ``mask``, parameters are
as for `{0.__module__}.{0.__name__}`.
""".format

class MaskedNDArray(Masked, ndarray):
    _mask = None
    info = MaskedNDArrayInfo()
    _eq_simple = _comparison_method('__eq__')
    _ne_simple = _comparison_method('__ne__')
    __lt__ = _comparison_method('__lt__')
    __le__ = _comparison_method('__le__')
    __gt__ = _comparison_method('__gt__')
    __ge__ = _comparison_method('__ge__')
    @classmethod
    def from_unmasked(cls, data, mask=None, copy=False):
        # Note: have to override since __new__ would use ndarray.__new__
        # which expects the shape as its first argument, not an array.
        data = np.array(data, subok=True, copy=copy)
        self = data.view(cls)
        self._set_mask(mask, copy=copy)
        return self
    @property
    def unmasked(self):
        return super().view(self._data_cls)
    def __array_finalize__(self, obj):
        # If we're a new object or viewing an ndarray, nothing has to be done.
        if obj is None or obj.__class__ is np.ndarray:
            return

        # Logically, this should come from ndarray and hence be None, but
        # just in case someone creates a new mixin, we check.
        super_array_finalize = super().__array_finalize__
        if super_array_finalize:  # pragma: no cover
            super_array_finalize(obj)

        if self._mask is None:
            # Got here after, e.g., a view of another masked class.
            # Get its mask, or initialize ours.
            self._set_mask(getattr(obj, '_mask', False))

        if 'info' in obj.__dict__:
            self.info = obj.info
    @shape.setter
    def shape(self, shape):
        old_shape = self.shape
        self._mask.shape = shape
        # Reshape array proper in try/except just in case some broadcasting
        # or so causes it to fail.
        try:
            super(MaskedNDArray, type(self)).shape.__set__(self, shape)
        except Exception as exc:
            self._mask.shape = old_shape
            # Given that the mask reshaping succeeded, the only logical
            # reason for an exception is something like a broadcast error in
            # in __array_finalize__, or a different memory ordering between
            # mask and data.  For those, give a more useful error message;
            # otherwise just raise the error.
            if 'could not broadcast' in exc.args[0]:
                raise AttributeError(
                    'Incompatible shape for in-place modification. '
                    'Use `.reshape()` to make a copy with the desired '
                    'shape.') from None
            else:  # pragma: no cover
                raise
    def _combine_masks(self, masks, out=None):
        masks = [m for m in masks if m is not None and m is not False]
        if not masks:
            return False
        if len(masks) == 1:
            if out is None:
                return masks[0].copy()
            else:
                np.copyto(out, masks[0])
                return out

        out = np.logical_or(masks[0], masks[1], out=out)
        for mask in masks[2:]:
            np.logical_or(out, mask, out=out)
        return out
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.pop('out', None)
        out_unmasked = None
        out_mask = None
        if out is not None:
            out_unmasked, out_masks = self._get_data_and_masks(*out)
            for d, m in zip(out_unmasked, out_masks):
                if m is None:
                    # TODO: allow writing to unmasked output if nothing is masked?
                    if d is not None:
                        raise TypeError('cannot write to unmasked output')
                elif out_mask is None:
                    out_mask = m

        unmasked, masks = self._get_data_and_masks(*inputs)

        if ufunc.signature:
            # We're dealing with a gufunc. For now, only deal with
            # np.matmul and gufuncs for which the mask of any output always
            # depends on all core dimension values of all inputs.
            # Also ignore axes keyword for now...
            # TODO: in principle, it should be possible to generate the mask
            # purely based on the signature.
            if 'axes' in kwargs:
                raise NotImplementedError("Masked does not yet support gufunc "
                                          "calls with 'axes'.")
            if ufunc is np.matmul:
                # np.matmul is tricky and its signature cannot be parsed by
                # _parse_gufunc_signature.
                unmasked = np.atleast_1d(*unmasked)
                mask0, mask1 = masks
                masks = []
                is_mat1 = unmasked[1].ndim >= 2
                if mask0 is not None:
                    masks.append(
                        np.logical_or.reduce(mask0, axis=-1, keepdims=is_mat1))

                if mask1 is not None:
                    masks.append(
                        np.logical_or.reduce(mask1, axis=-2, keepdims=True)
                        if is_mat1 else
                        np.logical_or.reduce(mask1))

                mask = self._combine_masks(masks, out=out_mask)

            else:
                # Parse signature with private numpy function. Note it
                # cannot handle spaces in tuples, so remove those.
                in_sig, out_sig = np.lib.function_base._parse_gufunc_signature(
                    ufunc.signature.replace(' ', ''))
                axis = kwargs.get('axis', -1)
                keepdims = kwargs.get('keepdims', False)
                in_masks = []
                for sig, mask in zip(in_sig, masks):
                    if mask is not None:
                        if sig:
                            # Input has core dimensions.  Assume that if any
                            # value in those is masked, the output will be
                            # masked too (TODO: for multiple core dimensions
                            # this may be too strong).
                            mask = np.logical_or.reduce(
                                mask, axis=axis, keepdims=keepdims)
                        in_masks.append(mask)

                mask = self._combine_masks(in_masks)
                result_masks = []
                for os in out_sig:
                    if os:
                        # Output has core dimensions.  Assume all those
                        # get the same mask.
                        result_mask = np.expand_dims(mask, axis)
                    else:
                        result_mask = mask
                    result_masks.append(result_mask)

                mask = result_masks if len(result_masks) > 1 else result_masks[0]

        elif method == '__call__':
            # Regular ufunc call.
            mask = self._combine_masks(masks, out=out_mask)

        elif method == 'outer':
            # Must have two arguments; adjust masks as will be done for data.
            assert len(masks) == 2
            masks = [(m if m is not None else False) for m in masks]
            mask = np.logical_or.outer(masks[0], masks[1], out=out_mask)

        elif method in {'reduce', 'accumulate'}:
            # Reductions like np.add.reduce (sum).
            if masks[0] is not None:
                # By default, we simply propagate masks, since for
                # things like np.sum, it makes no sense to do otherwise.
                # Individual methods need to override as needed.
                # TODO: take care of 'out' too?
                if method == 'reduce':
                    axis = kwargs.get('axis', None)
                    keepdims = kwargs.get('keepdims', False)
                    where = kwargs.get('where', True)
                    mask = np.logical_or.reduce(masks[0], where=where,
                                                axis=axis, keepdims=keepdims,
                                                out=out_mask)
                    if where is not True:
                        # Mask also whole rows that were not selected by where,
                        # so would have been left as unmasked above.
                        mask |= np.logical_and.reduce(masks[0], where=where,
                                                      axis=axis, keepdims=keepdims)

                else:
                    # Accumulate
                    axis = kwargs.get('axis', 0)
                    mask = np.logical_or.accumulate(masks[0], axis=axis,
                                                    out=out_mask)

            elif out is not None:
                mask = False

            else:  # pragma: no cover
                # Can only get here if neither input nor output was masked, but
                # perhaps axis or where was masked (in numpy < 1.21 this is
                # possible).  We don't support this.
                return NotImplemented

        elif method in {'reduceat', 'at'}:  # pragma: no cover
            # TODO: implement things like np.add.accumulate (used for cumsum).
            raise NotImplementedError("masked instances cannot yet deal with "
                                      "'reduceat' or 'at'.")

        if out_unmasked is not None:
            kwargs['out'] = out_unmasked
        result = getattr(ufunc, method)(*unmasked, **kwargs)

        if result is None:  # pragma: no cover
            # This happens for the "at" method.
            return result

        if out is not None and len(out) == 1:
            out = out[0]
        return self._masked_result(result, mask, out)
    def _masked_result(self, result, mask, out):
        if isinstance(result, tuple):
            if out is None:
                out = (None,) * len(result)
            if not isinstance(mask, (list, tuple)):
                mask = (mask,) * len(result)
            return tuple(self._masked_result(result_, mask_, out_)
                         for (result_, mask_, out_) in zip(result, mask, out))

        if out is None:
            # Note that we cannot count on result being the same class as
            # 'self' (e.g., comparison of quantity results in an ndarray, most
            # operations on Longitude and Latitude result in Angle or
            # Quantity), so use Masked to determine the appropriate class.
            return Masked(result, mask)

        # TODO: remove this sanity check once test cases are more complete.
        assert isinstance(out, Masked)
        # If we have an output, the result was written in-place, so we should
        # also write the mask in-place (if not done already in the code).
        if out._mask is not mask:
            out._mask[...] = mask
        return out
    def mean(self, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
        # Implementation based on that in numpy/core/_methods.py
        # Cast bool, unsigned int, and int to float64 by default,
        # and do float16 at higher precision.
        is_float16_result = False
        if dtype is None:
            if issubclass(self.dtype.type, (np.integer, np.bool_)):
                dtype = np.dtype('f8')
            elif issubclass(self.dtype.type, np.float16):
                dtype = np.dtype('f4')
                is_float16_result = out is None

        where = ~self.mask & where

        result = self.sum(axis=axis, dtype=dtype, out=out,
                          keepdims=keepdims, where=where)
        n = np.add.reduce(where, axis=axis, keepdims=keepdims)
        result /= n
        if is_float16_result:
            result = result.astype(self.dtype)
        return result