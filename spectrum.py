"""``spectrum`` contains functions and classes for processing spectral data."""

import numpy as np
import utilities as utils
import matplotlib.pyplot as plt
import time

from scipy import integrate

class LogBinError(Exception):
    """Exception when something is not log-binned. """
    pass

class Spectrum:
    """Structure for photon and electron spectra with log-binning in energy. 

    Parameters
    ----------
    eng : ndarray
        Abscissa for the spectrum. 
    dNdE : ndarray
        Spectrum stored as dN/dE. 
    rs : float, optional
        The redshift (1+z) of the spectrum. Set to -1 if not specified.

    Attributes
    ----------
    length : int
        The length of the `eng` and `dNdE`.
    underflow : dict of str: float
        The underflow total number of particles and total energy.
    log_bin_width : ndarray
        The log bin width.  
    bin_boundary : ndarray
        The boundary of each energy bin. Has one more entry than `length`.

    """

    # __array_priority__ must be larger than 0, so that radd can work.
    # Otherwise, ndarray + Spectrum works by iterating over the elements of
    # ndarray first, which isn't what we want.
    __array_priority__ = 1

    def __init__(self, eng, dNdE, rs=-1.):

        if eng.size != dNdE.size:
            raise TypeError("""abscissa and spectrum need to be of the
             same size.""")
        if not all(np.diff(eng) > 0):
            raise TypeError("abscissa must be ordered in increasing energy.")

        self.eng = eng
        self.dNdE = dNdE
        self.rs = rs
        self.length = eng.size
        self.underflow = {'N': 0., 'eng': 0.}

        self.bin_boundary = get_bin_bound(self.eng)
        self.log_bin_width = np.diff(np.log(self.bin_boundary))

    def __add__(self, other):
        """Adds two Spectrum instances together, or an array to the spectrum. The `Spectrum` object is on the left.
        
        Parameters
        ----------
        other : Spectrum, ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the summed spectrum. 

        Notes
        -----
        This special function, together with `Spectrum.__radd__`, allows the use of the symbol + to add `Spectrum` objects together.

        The returned `Spectrum` object `underflow` is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        __radd__

        """

        # Removed ability to add int or float. Not likely to be useful I think?

        if np.issubclass_(type(other), Spectrum):
            # Some typical errors.
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("abscissae are different for the two `Spectrum` objects.")
            if not np.array_equal(self.rs, other.rs):
                raise TypeError("redshifts are different for the two `Spectrum` objects.")

            new_spectrum = Spectrum(self.eng, self.dNdE+other.dNdE, self.rs)
            new_spectrum.underflow['N'] = (self.underflow['N'] 
                                          + other.underflow['N'])
            new_spectrum.underflow['eng'] = (self.underflow['eng']
                                            + other.underflow['eng'])

            return new_spectrum

        elif np.isinstance(other, ndarray):

            return Spectrum(self.eng, self.dNdE + other, self.rs)

        else:

            raise TypeError("cannot add object to Spectrum.")

    def __radd__(self, other):
        """Adds two Spectrum instances together, or an array to the spectrum. The `Spectrum` object is on the right.
        
        Parameters
        ----------
        other : Spectrum, ndarray

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the summed spectrum. 

        Notes
        -----
        This special function, together with `Spectrum.__add__`, allows the use of the symbol + to add `Spectrum` objects together.

        The returned `Spectrum` object `underflow` is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        __add__

        """
        
        # Removed ability to add int or float. Not likely to be useful I think?

        if np.issubclass_(type(other), Spectrum):
            # Some typical errors.
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("abscissae are different for the two `Spectrum` objects.")
            if not np.array_equal(self.rs, other.rs):
                raise TypeError("redshifts are different for the two `Spectrum` objects.")

            new_spectrum = Spectrum(self.eng, self.dNdE+other.dNdE, self.rs)
            new_spectrum.underflow['N'] = (self.underflow['N'] 
                                          + other.underflow['N'])
            new_spectrum.underflow['eng'] = (self.underflow['eng']
                                            + other.underflow['eng'])

            return new_spectrum

        elif np.isinstance(other, ndarray):

            return Spectrum(self.eng, self.dNdE + other, self.rs)

        else:

            raise TypeError("cannot add object to Spectrum.")

    def __sub__(self, other):
        """Subtracts one Spectrum instance from another, or subtracts an array from the spectrum. 
        
        Parameters
        ----------
        other : Spectrum or ndarray

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the subtracted `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectrum` objects.

        The returned `Spectrum` object nderflow is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        __rsub__

        """
        return self + -1*other

    def __rsub__(self, other):
        """Subtracts one Spectrum instance from another, or subtracts the spectrum from an array.
        
        Parameters
        ----------
        other : Spectrum or ndarray

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the subtracted `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__sub__`, allows the use of the symbol - to subtract or subtract from `Spectrum` objects.

        See Also
        --------
        __sub__

        """
        return other + -1*self

    def __neg__(self):
        """Negates the spectrum.

        Returns
        -------
        Spectrum
            New `Spectrum` instance with the spectrum negated. 
        """
        return -1*self

    def __mul__(self,other):
        """Takes the product of the spectrum with an array or number. `Spectrum` object is on the left.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the multiplied `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__rmul__`, allows the use of the symbol * to multiply `Spectrum` objects or an array and Spectrum.

        The returned `Spectrum` object `underflow` is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        __rmul__

        """
        if issubdtype(type(other),float) or issubdtype(type(other),int):
            new_spectrum = Spectrum(self.eng, self.dNdE*other, self.rs)
            new_spectrum.underflow['N'] = self.underflow['N']*other
            new_spectrum.underflow['eng'] = self.underflow['eng']*other
            return new_spectrum

        # Removed ability to multiply two `Spectrum` objects, doesn't seem like there's a physical reason for us to implement this.

        elif np.isinstance(other, ndarray):

            return Spectrum(self.eng, self.dNdE*other, self.rs)

        else:

            raise TypeError("cannot multiply object to Spectrum.")

    def __rmul__(self,other):
        """Takes the product of the spectrum with an array or number. `Spectrum` object is on the right.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the multiplied `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__mul__`, allows the use of the symbol * to multiply `Spectrum` objects or an array and Spectrum.

        The returned `Spectrum` object `underflow` is reset to zero if `other` is not a `Spectrum` object.

        See Also
        --------
        __mul__

        """
        if issubdtype(type(other),float) or issubdtype(type(other),int):
            new_spectrum = Spectrum(self.eng, self.dNdE*other, self.rs)
            new_spectrum.underflow['N'] = self.underflow['N']*other
            new_spectrum.underflow['eng'] = self.underflow['eng']*other
            return new_spectrum

        # Removed ability to multiply two `Spectrum` objects, doesn't seem like there's a physical reason for us to implement this.

        elif np.isinstance(other, ndarray):

            return Spectrum(self.eng, self.dNdE*other, self.rs)

        else:

            raise TypeError("cannot multiply object with Spectrum.")
    
    def __truediv__(self,other):
        """Divides the spectrum by an array or number.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the divided `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__truediv__`, allows the use of the symbol / to multiply `Spectrum` objects or an array and Spectrum.

        The returned `Spectrum` object `underflow` is reset to zero.

        """
        return self*(1/other)

    def __rtruediv__(self,other):
        """Divides a number or array by the spectrum.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New `Spectrum` instance which has the divided `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__truediv__`, allows the use of the symbol / to multiply `Spectrum` objects or an array and Spectrum.

        The returned `Spectrum` object `underflow` is reset to zero.

        """
        invSpec = Spectrum(self.eng, 1/self.dNdE, self.rs)
        return other*invSpec

    def contract(self, mat):
        """Performs a dot product on the spectrum with `mat`.

        Parameters
        ----------
        mat : ndarray
            The array to dot into the spectrum with.

        Returns
        -------
        float
            The resulting dot product.

        """
        return np.dot(mat,self.dNdE)

    def totN(self, bound_type=None, bound_arr=None):
        """Returns the total number of particles in part of the spectrum. 

        The part of the spectrum can be specified in two ways, and is specified by `bound_type`. Multiple totals can be obtained through `bound_arr`. 
        
        Parameters
        ----------
        bound_type : {'bin', 'eng', None}
            The type of bounds to use. Bound values do not have to be within the [0:length] for `'bin'` or within the abscissa for `'eng'`. `None` should only be used when computing the total particle number in the spectrum.

            `'bin'` : bounds are specified as the bin boundary, with 0 being the left most boundary, 1 the right-hand of the first bin and so on. This is equivalent to integrating over a histogram. 
            
            `'eng'` : bounds are specified by energy values.

        bound_arr : ndarray, optional
            An array of boundaries (bin or energy), between which the total number of particles will be computed. If unspecified, the total number of particles in the whole spectrum is computed.

        Returns
        -------
        ndarray or float
            Total number of particles in the spectrum. 
        """
        
        dNdlogE = self.eng*self.dNdE
        length = self.length
        log_bin_width = self.log_bin_width

        if bound_type is not None and bound_arr.size is not None:

            if bound_type == 'bin':

                if not all(np.diff(bound_arr) > 0):
                    raise TypeError("bound_arr must have increasing entries.")

                N_in_bin = np.zeros(bound_arr.size-1)

                if bound_arr[0] > length or bound_arr[-1] < 0:
                    return N_in_bin

                for low,upp,i in zip(bound_arr[:-1], bound_arr[1:], 
                    np.arange(N_in_bin.size)):
                    # Set the lower and upper bounds, including case where low and upp are outside of the bins.
                    low_bound = np.amax([0, low])
                    upp_bound = np.amax([upp, length])

                    if low > length or upp < 0:
                        N_in_bin[i] = 0
                        continue

                    low_ceil  = int(np.ceil(low))
                    low_floor = int(np.floor(low))
                    upp_ceil  = int(np.ceil(upp))
                    upp_floor = int(np.floor(upp))
                    # Sum the bins that are completely between the bounds.
                    N_full_bins = np.dot(dNdlogE[low_ceil:upp_floor],log_bin_width[low_ceil:upp_floor])

                    N_part_bins = 0

                    if low_floor == upp_floor or low_ceil == upp_ceil:
                        # Bin indices are within the same bin. The second requirement covers the case where upp_ceil is length. 
                        N_part_bins += (dNdlogE[low_floor] * (upp - low)
                            * log_bin_width[low_floor])
                    else:
                        # Add up part of the bin for the low partial bin and the high partial bin. 
                        N_part_bins += (dNdlogE[low_floor] * (low_ceil - low)
                            * log_bin_width[low_floor])
                        if upp_floor < length:
                        # If upp_floor is length, then there is no partial bin for the upper index. 
                            N_part_bins += (dNdlogE[upp_floor]
                                * (upp-upp_floor) * log_bin_width[upp_floor])

                    return N_full_bins + N_part_bins

            if bound_type == 'eng':

                eng_bin_ind = np.interp( 
                    np.log(bound_arr), 
                    np.log(self.bin_boundary), np.arange(bin_boundary.size), 
                    left = -1, right = length + 1)

                return self.totN('bin', eng_bin_ind)

        else:
            return np.dot(dNdlogE,log_bin_width) + self.underflow['N']

    def toteng(self, bound_type=None, bound_arr=None):
        """Returns the total energy of particles in part of the spectrum. 

        The part of the spectrum can be specified in two ways, and is specified by `bound_type`. Multiple totals can be obtained through `bound_arr`. 

        Parameters
        ----------
        bound_type : {'bin', 'eng', None}
            The type of bounds to use. Bound values do not have to be within the [0:length] for `'bin'` or within the abscissa for `'eng'`. `None` should only be used to obtain the total energy. 

            With `'bin'`, bounds are specified as the bin boundary, with 0 being the left most boundary, 1 the right-hand of the first bin and so on. This is equivalent to integrating over a histogram. 
            
            With `'eng'`, bounds are specified by energy values. 

        bound_arr : ndarray, optional
            An array of boundaries (bin or energy), between which the total number of particles will be computed. If unspecified, the total number of particles in the whole spectrum is computed.


        Returns
        -------
        ndarray or float
            Total energy in the spectrum. 
        """
        eng = self.eng
        dNdlogE = eng*self.dNdE
        length = self.length
        log_bin_width = self.log_bin_width

        if bound_type is not None and bound_arr.size is not None:

            if bound_type == 'bin':

                if not all(np.diff(bound_arr) > 0):
                    raise TypeError("bound_arr must have increasing entries.")

                eng_in_bin = np.zeros(bound_arr.size-1)

                if bound_arr[0] > length or bound_arr[-1] < 0:
                    return eng_in_bin

                for low,upp,i in zip(bound_arr[:-1], bound_arr[1:], 
                    np.arange(eng_in_bin.size)):
                    # Set the lower and upper bounds, including case where low and upp are outside of the bins.
                    low_bound = np.amax([0, low])
                    upp_bound = np.amax([upp, length])

                    if low > length or upp < 0:
                        eng_in_bin[i] = 0
                        continue

                    low_ceil  = int(np.ceil(low))
                    low_floor = int(np.floor(low))
                    upp_ceil  = int(np.ceil(upp))
                    upp_floor = int(np.floor(upp))
                    # Sum the bins that are completely between the bounds.
                    eng_full_bins = np.dot(eng[low_ceil:upp_floor]
                        * dNdlogE[low_ceil:upp_floor],
                        log_bin_width[low_ceil:upp_floor])

                    eng_part_bins = 0

                    if low_floor == upp_floor or low_ceil == upp_ceil:
                        # Bin indices are within the same bin. The second requirement covers the case where upp_ceil is length. 
                        eng_part_bins += (eng[low_floor] * dNdlogE[low_floor]
                            * (upp - low) * log_bin_width[low_floor])
                    else:
                        # Add up part of the bin for the low partial bin and the high partial bin. 
                        eng_part_bins += (eng[low_floor] * dNdlogE[low_floor]
                            * (low_ceil - low) * log_bin_width[low_floor])
                        if upp_floor < length:
                        # If upp_floor is length, then there is no partial bin for the upper index. 
                            eng_part_bins += (eng[upp_floor]
                                * dNdlogE[upp_floor] * (upp-upp_floor) 
                                * log_bin_width[upp_floor])

                    return eng_full_bins + eng_part_bins

            if bound_type == 'eng':

                eng_bin_ind = np.interp( 
                    np.log(bound_arr), 
                    np.log(self.bin_boundary), np.arange(bin_boundary.size), 
                    left = -1, right = length + 1)

                return self.toteng('bin', eng_bin_ind)

        else:
            return (np.dot(dNdlogE, eng * log_bin_width) 
                + self.underflow['eng'])

    def rebin(self, new_eng):
        """ Re-bins the spectrum according to a new abscissa `new_eng`. 

        Rebinning conserves total number and total energy.
        
        Parameters
        ----------
        new_eng : ndarray
            The new abscissa to bin into. If `new_eng[-1]` exceeds the largest entry in the current abscissa, then the new underflow will be filled. If `new_eng[-1]` is smaller, then it will be filled with all of the overflow.

        Returns
        -------
        Spectrum
            The final `Spectrum` object with the new binning.

        """
        eng = self.eng
        dNdE = self.dNdE
        log_bin_width = self.log_bin_width
        
        N = self.totN()

def get_bin_bound(eng):
    """Returns the bin boundary of an abscissa.
    
    The bin boundaries are computed by taking the midpoint of the **log** of the abscissa. The first and last entries are computed by taking all of the bins to be symmetric with respect to the bin center. 

    Parameters
    ----------
    eng : ndarray
        Abscissa from which the bin boundary is obtained.

    Returns
    -------
    ndarray
        The bin boundaries. 
    """
    log_bin_width_low = np.log(eng[1]) - np.log(eng[0])
    log_bin_width_upp = np.log(eng[-1]) - np.log(eng[-2])

    bin_boundary = np.sqrt(eng[:-1] * eng[1:])

    low_lim = np.exp(np.log(eng[0]) - log_bin_width_low / 2)
    upp_lim = np.exp(np.log(eng[-1]) + log_bin_width_upp / 2)
    bin_boundary = np.insert(bin_boundary, 0, low_lim)
    bin_boundary = np.append(bin_boundary, upp_lim)

    return bin_boundary

def discretize(func, eng):
    """Discretizes a continuous function. 

    The function is integrated between the bin boundaries specified by `eng` to obtain the discretized spectrum, so that the spectrum value times the bin width agrees with the integral.

    Parameters
    ----------
    func : function
        A single variable function that takes in energy as an input, and then returns a dN/dE spectrum value. 
    eng : ndarray
        The new abscissa after discretization. 

    Returns
    -------
    Spectrum
        The discretized spectrum. rs is set to -1, and must be set manually. 
    """
    bin_boundary = get_bin_bound(eng)
    bin_width = np.diff(bin_boundary)
    N = np.zeros(eng.size)
    
    for low, upp, i in zip(bin_boundary[:-1], bin_boundary[1:], 
        np.arange(bin_width.size)):
    # Perform an integral over the spectrum for each bin.
        N[i] = integrate.quad(func, low, upp)[0]

    return Spectrum(eng, N/bin_width)









