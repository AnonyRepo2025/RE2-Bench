import copy
import io
import itertools
import os
import re
import textwrap
import warnings
import builtins
import numpy as np
from .. import log
from ..io import fits
from . import _docutil as __
from ..utils.compat import possible_filename
from ..utils.exceptions import AstropyWarning, AstropyUserWarning, AstropyDeprecationWarning
from .wcsapi.fitswcs import FITSWCSAPIMixin
from . import _wcs
from copy import deepcopy
from ..visualization.wcsaxes import WCSAxes

__all__ = ['FITSFixedWarning', 'WCS', 'find_all_wcs',
           'DistortionLookupTable', 'Sip', 'Tabprm', 'Wcsprm',
           'WCSBase', 'validate', 'WcsError', 'SingularMatrixError',
           'InconsistentAxisTypesError', 'InvalidTransformError',
           'InvalidCoordinateError', 'NoSolutionError',
           'InvalidSubimageSpecificationError', 'NoConvergence',
           'NonseparableSubimageCoordinateSystemError',
           'NoWcsKeywordsFoundError', 'InvalidTabularParametersError']
__doctest_skip__ = ['WCS.all_world2pix']
NAXIS_DEPRECATE_MESSAGE = """
Private attributes "_naxis1" and "naxis2" have been deprecated since v3.1.
Instead use the "pixel_shape" property which returns a list of NAXISj keyword values.
"""
WCSHDO_SIP = 0x80000
SIP_KW = re.compile('''^[AB]P?_1?[0-9]_1?[0-9][A-Z]?$''')

class WCS(FITSWCSAPIMixin, WCSBase):
    all_pix2world.__doc__ = """
        Transforms pixel coordinates to world coordinates.

        Performs all of the following in series:

            - Detector to image plane correction (if present in the
              FITS file)

            - `SIP`_ distortion correction (if present in the FITS
              file)

            - `distortion paper`_ table-lookup correction (if present
              in the FITS file)

            - `wcslib`_ "core" WCS transformation

        Parameters
        ----------
        {0}

            For a transformation that is not two-dimensional, the
            two-argument form must be used.

        {1}

        Returns
        -------

        {2}

        Notes
        -----
        The order of the axes for the result is determined by the
        ``CTYPEia`` keywords in the FITS header, therefore it may not
        always be of the form (*ra*, *dec*).  The
        `~astropy.wcs.Wcsprm.lat`, `~astropy.wcs.Wcsprm.lng`,
        `~astropy.wcs.Wcsprm.lattyp` and `~astropy.wcs.Wcsprm.lngtyp`
        members can be used to determine the order of the axes.

        Raises
        ------
        MemoryError
            Memory allocation failed.

        SingularMatrixError
            Linear transformation matrix is singular.

        InconsistentAxisTypesError
            Inconsistent or unrecognized coordinate axis types.

        ValueError
            Invalid parameter value.

        ValueError
            Invalid coordinate transformation parameters.

        ValueError
            x- and y-coordinate arrays are not the same size.

        InvalidTransformError
            Invalid coordinate transformation parameters.

        InvalidTransformError
            Ill-conditioned coordinate transformation parameters.
        """.format(__.TWO_OR_MORE_ARGS('naxis', 8),
                   __.RA_DEC_ORDER(8),
                   __.RETURNS('sky coordinates, in degrees', 8))
    wcs_pix2world.__doc__ = """
        Transforms pixel coordinates to world coordinates by doing
        only the basic `wcslib`_ transformation.

        No `SIP`_ or `distortion paper`_ table lookup correction is
        applied.  To perform distortion correction, see
        `~astropy.wcs.WCS.all_pix2world`,
        `~astropy.wcs.WCS.sip_pix2foc`, `~astropy.wcs.WCS.p4_pix2foc`,
        or `~astropy.wcs.WCS.pix2foc`.

        Parameters
        ----------
        {0}

            For a transformation that is not two-dimensional, the
            two-argument form must be used.

        {1}

        Returns
        -------

        {2}

        Raises
        ------
        MemoryError
            Memory allocation failed.

        SingularMatrixError
            Linear transformation matrix is singular.

        InconsistentAxisTypesError
            Inconsistent or unrecognized coordinate axis types.

        ValueError
            Invalid parameter value.

        ValueError
            Invalid coordinate transformation parameters.

        ValueError
            x- and y-coordinate arrays are not the same size.

        InvalidTransformError
            Invalid coordinate transformation parameters.

        InvalidTransformError
            Ill-conditioned coordinate transformation parameters.

        Notes
        -----
        The order of the axes for the result is determined by the
        ``CTYPEia`` keywords in the FITS header, therefore it may not
        always be of the form (*ra*, *dec*).  The
        `~astropy.wcs.Wcsprm.lat`, `~astropy.wcs.Wcsprm.lng`,
        `~astropy.wcs.Wcsprm.lattyp` and `~astropy.wcs.Wcsprm.lngtyp`
        members can be used to determine the order of the axes.

        """.format(__.TWO_OR_MORE_ARGS('naxis', 8),
                   __.RA_DEC_ORDER(8),
                   __.RETURNS('world coordinates, in degrees', 8))
    all_world2pix.__doc__ = """
        all_world2pix(*arg, accuracy=1.0e-4, maxiter=20,
        adaptive=False, detect_divergence=True, quiet=False)

        Transforms world coordinates to pixel coordinates, using
        numerical iteration to invert the full forward transformation
        `~astropy.wcs.WCS.all_pix2world` with complete
        distortion model.


        Parameters
        ----------
        {0}

            For a transformation that is not two-dimensional, the
            two-argument form must be used.

        {1}

        tolerance : float, optional (Default = 1.0e-4)
            Tolerance of solution. Iteration terminates when the
            iterative solver estimates that the "true solution" is
            within this many pixels current estimate, more
            specifically, when the correction to the solution found
            during the previous iteration is smaller
            (in the sense of the L2 norm) than ``tolerance``.

        maxiter : int, optional (Default = 20)
            Maximum number of iterations allowed to reach a solution.

        quiet : bool, optional (Default = False)
            Do not throw :py:class:`NoConvergence` exceptions when
            the method does not converge to a solution with the
            required accuracy within a specified number of maximum
            iterations set by ``maxiter`` parameter. Instead,
            simply return the found solution.

        Other Parameters
        ----------------
        adaptive : bool, optional (Default = False)
            Specifies whether to adaptively select only points that
            did not converge to a solution within the required
            accuracy for the next iteration. Default is recommended
            for HST as well as most other instruments.

            .. note::
               The :py:meth:`all_world2pix` uses a vectorized
               implementation of the method of consecutive
               approximations (see ``Notes`` section below) in which it
               iterates over *all* input points *regardless* until
               the required accuracy has been reached for *all* input
               points. In some cases it may be possible that
               *almost all* points have reached the required accuracy
               but there are only a few of input data points for
               which additional iterations may be needed (this
               depends mostly on the characteristics of the geometric
               distortions for a given instrument). In this situation
               it may be advantageous to set ``adaptive`` = `True` in
               which case :py:meth:`all_world2pix` will continue
               iterating *only* over the points that have not yet
               converged to the required accuracy. However, for the
               HST's ACS/WFC detector, which has the strongest
               distortions of all HST instruments, testing has
               shown that enabling this option would lead to a about
               50-100% penalty in computational time (depending on
               specifics of the image, geometric distortions, and
               number of input points to be converted). Therefore,
               for HST and possibly instruments, it is recommended
               to set ``adaptive`` = `False`. The only danger in
               getting this setting wrong will be a performance
               penalty.

            .. note::
               When ``detect_divergence`` is `True`,
               :py:meth:`all_world2pix` will automatically switch
               to the adaptive algorithm once divergence has been
               detected.

        detect_divergence : bool, optional (Default = True)
            Specifies whether to perform a more detailed analysis
            of the convergence to a solution. Normally
            :py:meth:`all_world2pix` may not achieve the required
            accuracy if either the ``tolerance`` or ``maxiter`` arguments
            are too low. However, it may happen that for some
            geometric distortions the conditions of convergence for
            the the method of consecutive approximations used by
            :py:meth:`all_world2pix` may not be satisfied, in which
            case consecutive approximations to the solution will
            diverge regardless of the ``tolerance`` or ``maxiter``
            settings.

            When ``detect_divergence`` is `False`, these divergent
            points will be detected as not having achieved the
            required accuracy (without further details). In addition,
            if ``adaptive`` is `False` then the algorithm will not
            know that the solution (for specific points) is diverging
            and will continue iterating and trying to "improve"
            diverging solutions. This may result in ``NaN`` or
            ``Inf`` values in the return results (in addition to a
            performance penalties). Even when ``detect_divergence``
            is `False`, :py:meth:`all_world2pix`, at the end of the
            iterative process, will identify invalid results
            (``NaN`` or ``Inf``) as "diverging" solutions and will
            raise :py:class:`NoConvergence` unless the ``quiet``
            parameter is set to `True`.

            When ``detect_divergence`` is `True`,
            :py:meth:`all_world2pix` will detect points for which
            current correction to the coordinates is larger than
            the correction applied during the previous iteration
            **if** the requested accuracy **has not yet been
            achieved**. In this case, if ``adaptive`` is `True`,
            these points will be excluded from further iterations and
            if ``adaptive`` is `False`, :py:meth:`all_world2pix` will
            automatically switch to the adaptive algorithm. Thus, the
            reported divergent solution will be the latest converging
            solution computed immediately *before* divergence
            has been detected.

            .. note::
               When accuracy has been achieved, small increases in
               current corrections may be possible due to rounding
               errors (when ``adaptive`` is `False`) and such
               increases will be ignored.

            .. note::
               Based on our testing using HST ACS/WFC images, setting
               ``detect_divergence`` to `True` will incur about 5-20%
               performance penalty with the larger penalty
               corresponding to ``adaptive`` set to `True`.
               Because the benefits of enabling this
               feature outweigh the small performance penalty,
               especially when ``adaptive`` = `False`, it is
               recommended to set ``detect_divergence`` to `True`,
               unless extensive testing of the distortion models for
               images from specific instruments show a good stability
               of the numerical method for a wide range of
               coordinates (even outside the image itself).

            .. note::
               Indices of the diverging inverse solutions will be
               reported in the ``divergent`` attribute of the
               raised :py:class:`NoConvergence` exception object.

        Returns
        -------

        {2}

        Notes
        -----
        The order of the axes for the input world array is determined by
        the ``CTYPEia`` keywords in the FITS header, therefore it may
        not always be of the form (*ra*, *dec*).  The
        `~astropy.wcs.Wcsprm.lat`, `~astropy.wcs.Wcsprm.lng`,
        `~astropy.wcs.Wcsprm.lattyp`, and
        `~astropy.wcs.Wcsprm.lngtyp`
        members can be used to determine the order of the axes.

        Using the method of fixed-point iterations approximations we
        iterate starting with the initial approximation, which is
        computed using the non-distortion-aware
        :py:meth:`wcs_world2pix` (or equivalent).

        The :py:meth:`all_world2pix` function uses a vectorized
        implementation of the method of consecutive approximations and
        therefore it is highly efficient (>30x) when *all* data points
        that need to be converted from sky coordinates to image
        coordinates are passed at *once*. Therefore, it is advisable,
        whenever possible, to pass as input a long array of all points
        that need to be converted to :py:meth:`all_world2pix` instead
        of calling :py:meth:`all_world2pix` for each data point. Also
        see the note to the ``adaptive`` parameter.

        Raises
        ------
        NoConvergence
            The method did not converge to a
            solution to the required accuracy within a specified
            number of maximum iterations set by the ``maxiter``
            parameter. To turn off this exception, set ``quiet`` to
            `True`. Indices of the points for which the requested
            accuracy was not achieved (if any) will be listed in the
            ``slow_conv`` attribute of the
            raised :py:class:`NoConvergence` exception object.

            See :py:class:`NoConvergence` documentation for
            more details.

        MemoryError
            Memory allocation failed.

        SingularMatrixError
            Linear transformation matrix is singular.

        InconsistentAxisTypesError
            Inconsistent or unrecognized coordinate axis types.

        ValueError
            Invalid parameter value.

        ValueError
            Invalid coordinate transformation parameters.

        ValueError
            x- and y-coordinate arrays are not the same size.

        InvalidTransformError
            Invalid coordinate transformation parameters.

        InvalidTransformError
            Ill-conditioned coordinate transformation parameters.

        Examples
        --------
        >>> import astropy.io.fits as fits
        >>> import astropy.wcs as wcs
        >>> import numpy as np
        >>> import os

        >>> filename = os.path.join(wcs.__path__[0], 'tests/data/j94f05bgq_flt.fits')
        >>> hdulist = fits.open(filename)
        >>> w = wcs.WCS(hdulist[('sci',1)].header, hdulist)
        >>> hdulist.close()

        >>> ra, dec = w.all_pix2world([1,2,3], [1,1,1], 1)
        >>> print(ra)  # doctest: +FLOAT_CMP
        [ 5.52645627  5.52649663  5.52653698]
        >>> print(dec)  # doctest: +FLOAT_CMP
        [-72.05171757 -72.05171276 -72.05170795]
        >>> radec = w.all_pix2world([[1,1], [2,1], [3,1]], 1)
        >>> print(radec)  # doctest: +FLOAT_CMP
        [[  5.52645627 -72.05171757]
         [  5.52649663 -72.05171276]
         [  5.52653698 -72.05170795]]
        >>> x, y = w.all_world2pix(ra, dec, 1)
        >>> print(x)  # doctest: +FLOAT_CMP
        [ 1.00000238  2.00000237  3.00000236]
        >>> print(y)  # doctest: +FLOAT_CMP
        [ 0.99999996  0.99999997  0.99999997]
        >>> xy = w.all_world2pix(radec, 1)
        >>> print(xy)  # doctest: +FLOAT_CMP
        [[ 1.00000238  0.99999996]
         [ 2.00000237  0.99999997]
         [ 3.00000236  0.99999997]]
        >>> xy = w.all_world2pix(radec, 1, maxiter=3,
        ...                      tolerance=1.0e-10, quiet=False)
        Traceback (most recent call last):
        ...
        NoConvergence: 'WCS.all_world2pix' failed to converge to the
        requested accuracy. After 3 iterations, the solution is
        diverging at least for one input point.

        >>> # Now try to use some diverging data:
        >>> divradec = w.all_pix2world([[1.0, 1.0],
        ...                             [10000.0, 50000.0],
        ...                             [3.0, 1.0]], 1)
        >>> print(divradec)  # doctest: +FLOAT_CMP
        [[  5.52645627 -72.05171757]
         [  7.15976932 -70.8140779 ]
         [  5.52653698 -72.05170795]]

        >>> # First, turn detect_divergence on:
        >>> try:  # doctest: +FLOAT_CMP
        ...   xy = w.all_world2pix(divradec, 1, maxiter=20,
        ...                        tolerance=1.0e-4, adaptive=False,
        ...                        detect_divergence=True,
        ...                        quiet=False)
        ... except wcs.wcs.NoConvergence as e:
        ...   print("Indices of diverging points: {{0}}"
        ...         .format(e.divergent))
        ...   print("Indices of poorly converging points: {{0}}"
        ...         .format(e.slow_conv))
        ...   print("Best solution:\\n{{0}}".format(e.best_solution))
        ...   print("Achieved accuracy:\\n{{0}}".format(e.accuracy))
        Indices of diverging points: [1]
        Indices of poorly converging points: None
        Best solution:
        [[  1.00000238e+00   9.99999965e-01]
         [ -1.99441636e+06   1.44309097e+06]
         [  3.00000236e+00   9.99999966e-01]]
        Achieved accuracy:
        [[  6.13968380e-05   8.59638593e-07]
         [  8.59526812e+11   6.61713548e+11]
         [  6.09398446e-05   8.38759724e-07]]
        >>> raise e
        Traceback (most recent call last):
        ...
        NoConvergence: 'WCS.all_world2pix' failed to converge to the
        requested accuracy.  After 5 iterations, the solution is
        diverging at least for one input point.

        >>> # This time turn detect_divergence off:
        >>> try:  # doctest: +FLOAT_CMP
        ...   xy = w.all_world2pix(divradec, 1, maxiter=20,
        ...                        tolerance=1.0e-4, adaptive=False,
        ...                        detect_divergence=False,
        ...                        quiet=False)
        ... except wcs.wcs.NoConvergence as e:
        ...   print("Indices of diverging points: {{0}}"
        ...         .format(e.divergent))
        ...   print("Indices of poorly converging points: {{0}}"
        ...         .format(e.slow_conv))
        ...   print("Best solution:\\n{{0}}".format(e.best_solution))
        ...   print("Achieved accuracy:\\n{{0}}".format(e.accuracy))
        Indices of diverging points: [1]
        Indices of poorly converging points: None
        Best solution:
        [[ 1.00000009  1.        ]
         [        nan         nan]
         [ 3.00000009  1.        ]]
        Achieved accuracy:
        [[  2.29417358e-06   3.21222995e-08]
         [             nan              nan]
         [  2.27407877e-06   3.13005639e-08]]
        >>> raise e
        Traceback (most recent call last):
        ...
        NoConvergence: 'WCS.all_world2pix' failed to converge to the
        requested accuracy.  After 6 iterations, the solution is
        diverging at least for one input point.

        """.format(__.TWO_OR_MORE_ARGS('naxis', 8),
                   __.RA_DEC_ORDER(8),
                   __.RETURNS('pixel coordinates', 8))
    wcs_world2pix.__doc__ = """
        Transforms world coordinates to pixel coordinates, using only
        the basic `wcslib`_ WCS transformation.  No `SIP`_ or
        `distortion paper`_ table lookup transformation is applied.

        Parameters
        ----------
        {0}

            For a transformation that is not two-dimensional, the
            two-argument form must be used.

        {1}

        Returns
        -------

        {2}

        Notes
        -----
        The order of the axes for the input world array is determined by
        the ``CTYPEia`` keywords in the FITS header, therefore it may
        not always be of the form (*ra*, *dec*).  The
        `~astropy.wcs.Wcsprm.lat`, `~astropy.wcs.Wcsprm.lng`,
        `~astropy.wcs.Wcsprm.lattyp` and `~astropy.wcs.Wcsprm.lngtyp`
        members can be used to determine the order of the axes.

        Raises
        ------
        MemoryError
            Memory allocation failed.

        SingularMatrixError
            Linear transformation matrix is singular.

        InconsistentAxisTypesError
            Inconsistent or unrecognized coordinate axis types.

        ValueError
            Invalid parameter value.

        ValueError
            Invalid coordinate transformation parameters.

        ValueError
            x- and y-coordinate arrays are not the same size.

        InvalidTransformError
            Invalid coordinate transformation parameters.

        InvalidTransformError
            Ill-conditioned coordinate transformation parameters.
        """.format(__.TWO_OR_MORE_ARGS('naxis', 8),
                   __.RA_DEC_ORDER(8),
                   __.RETURNS('pixel coordinates', 8))
    pix2foc.__doc__ = """
        Convert pixel coordinates to focal plane coordinates using the
        `SIP`_ polynomial distortion convention and `distortion
        paper`_ table-lookup correction.

        The output is in absolute pixel coordinates, not relative to
        ``CRPIX``.

        Parameters
        ----------

        {0}

        Returns
        -------

        {1}

        Raises
        ------
        MemoryError
            Memory allocation failed.

        ValueError
            Invalid coordinate transformation parameters.
        """.format(__.TWO_OR_MORE_ARGS('2', 8),
                   __.RETURNS('focal coordinates', 8))
    p4_pix2foc.__doc__ = """
        Convert pixel coordinates to focal plane coordinates using
        `distortion paper`_ table-lookup correction.

        The output is in absolute pixel coordinates, not relative to
        ``CRPIX``.

        Parameters
        ----------

        {0}

        Returns
        -------

        {1}

        Raises
        ------
        MemoryError
            Memory allocation failed.

        ValueError
            Invalid coordinate transformation parameters.
        """.format(__.TWO_OR_MORE_ARGS('2', 8),
                   __.RETURNS('focal coordinates', 8))
    det2im.__doc__ = """
        Convert detector coordinates to image plane coordinates using
        `distortion paper`_ table-lookup correction.

        The output is in absolute pixel coordinates, not relative to
        ``CRPIX``.

        Parameters
        ----------

        {0}

        Returns
        -------

        {1}

        Raises
        ------
        MemoryError
            Memory allocation failed.

        ValueError
            Invalid coordinate transformation parameters.
        """.format(__.TWO_OR_MORE_ARGS('2', 8),
                   __.RETURNS('pixel coordinates', 8))
    sip_pix2foc.__doc__ = """
        Convert pixel coordinates to focal plane coordinates using the
        `SIP`_ polynomial distortion convention.

        The output is in pixel coordinates, relative to ``CRPIX``.

        FITS WCS `distortion paper`_ table lookup correction is not
        applied, even if that information existed in the FITS file
        that initialized this :class:`~astropy.wcs.WCS` object.  To
        correct for that, use `~astropy.wcs.WCS.pix2foc` or
        `~astropy.wcs.WCS.p4_pix2foc`.

        Parameters
        ----------

        {0}

        Returns
        -------

        {1}

        Raises
        ------
        MemoryError
            Memory allocation failed.

        ValueError
            Invalid coordinate transformation parameters.
        """.format(__.TWO_OR_MORE_ARGS('2', 8),
                   __.RETURNS('focal coordinates', 8))
    sip_foc2pix.__doc__ = """
        Convert focal plane coordinates to pixel coordinates using the
        `SIP`_ polynomial distortion convention.

        FITS WCS `distortion paper`_ table lookup distortion
        correction is not applied, even if that information existed in
        the FITS file that initialized this `~astropy.wcs.WCS` object.

        Parameters
        ----------

        {0}

        Returns
        -------

        {1}

        Raises
        ------
        MemoryError
            Memory allocation failed.

        ValueError
            Invalid coordinate transformation parameters.
        """.format(__.TWO_OR_MORE_ARGS('2', 8),
                   __.RETURNS('pixel coordinates', 8))
    def calc_footprint(self, header=None, undistort=True, axes=None, center=True):
        """
        Calculates the footprint of the image on the sky.

        A footprint is defined as the positions of the corners of the
        image on the sky after all available distortions have been
        applied.

        Parameters
        ----------
        header : `~astropy.io.fits.Header` object, optional
            Used to get ``NAXIS1`` and ``NAXIS2``
            header and axes are mutually exclusive, alternative ways
            to provide the same information.

        undistort : bool, optional
            If `True`, take SIP and distortion lookup table into
            account

        axes : length 2 sequence ints, optional
            If provided, use the given sequence as the shape of the
            image.  Otherwise, use the ``NAXIS1`` and ``NAXIS2``
            keywords from the header that was used to create this
            `WCS` object.

        center : bool, optional
            If `True` use the center of the pixel, otherwise use the corner.

        Returns
        -------
        coord : (4, 2) array of (*x*, *y*) coordinates.
            The order is clockwise starting with the bottom left corner.
        """
        if axes is not None:
            naxis1, naxis2 = axes
        else:
            if header is None:
                try:
                    # classes that inherit from WCS and define naxis1/2
                    # do not require a header parameter
                    naxis1, naxis2 = self.pixel_shape
                except (AttributeError, TypeError):
                    warnings.warn("Need a valid header in order to calculate footprint\n", AstropyUserWarning)
                    return None
            else:
                naxis1 = header.get('NAXIS1', None)
                naxis2 = header.get('NAXIS2', None)

        if naxis1 is None or naxis2 is None:
            raise ValueError(
                    "Image size could not be determined.")

        if center:
            corners = np.array([[1, 1],
                                [1, naxis2],
                                [naxis1, naxis2],
                                [naxis1, 1]], dtype=np.float64)
        else:
            corners = np.array([[0.5, 0.5],
                                [0.5, naxis2 + 0.5],
                                [naxis1 + 0.5, naxis2 + 0.5],
                                [naxis1 + 0.5, 0.5]], dtype=np.float64)

        if undistort:
            return self.all_pix2world(corners, 1)
        else:
            return self.wcs_pix2world(corners, 1)