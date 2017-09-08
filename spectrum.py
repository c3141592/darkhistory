"""``spectrum`` contains functions and classes for processing spectral data."""

from numpy import *
import utilities as utils
import matplotlib.pyplot as plt
import time


class Spectrum:
    """Structure for photon and electron spectra with log-binning in energy. 

    Parameters
    ----------
    eng, dNdE : array_like
        Abscissa for the spectrum, and spectrum stored as dN/dE. 
    rs : float
        The redshift (1+z) of the spectrum.

    Attributes
    ----------
    length : int
        The length of the `eng` and `dNdE`.
    underflow : dict of str: float
        The underflow total number of particles and total energy.
    log_bin_width : float
        The log bin width. 
    bin_boundary : ndarray
        The boundary of each energy bin. Has one more entry than `length`.

    """

    # __array_priority__ must be larger than 0, so that radd can work.
    # Otherwise, ndarray + Spectrum works by iterating over the elements of
    # ndarray first, which isn't what we want.
    __array_priority__ = 1

    def __init__(self, eng, dNdE, rs):

        if eng.size != dNdE.size:
            raise TypeError("""abscissa and spectrum need to be of the
             same size.""")
        if not all(diff(eng) > 0):
            raise TypeError("abscissa must be ordered in increasing energy.")
        if not utils.is_log_spaced(eng):
            raise TypeError("abscissa must be log spaced.")

        self.eng = eng
        self.dNdE = dNdE
        self.rs = rs
        self.length = eng.size
        self.underflow = {'N': 0., 'eng': 0.}

        log_bin_width = np.log(eng[1]) - np.log(eng[0])
        self.log_bin_width = log_bin_width

        bin_boundary = np.sqrt(eng[:-1] * eng[1:])
        low_lim = np.exp(np.log(eng[0]) - log_bin_width / 2)
        upp_lim = np.exp(np.log(eng[-1]) + log_bin_width / 2)
        bin_boundary = np.insert(bin_boundary, 0, low_lim)
        bin_boundary = np.append(bin_boundary, upp_lim)

        self.bin_boundary = bin_boundary

    def __add__(self, other):
        """Adds two ``Spectrum`` instances together, or an array to the spectrum. The Spectrum object is on the left.
        
        Parameters
        ----------
        other : Spectrum, ndarray, float or int

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the summed spectrum. 

        Notes
        -----
        This special function, together with `Spectrum.__radd__`, allows the use of the symbol + to add ``Spectrum`` objects together.

        The returned ``Spectrum`` object `underflow` is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        __radd__

        """

        # Removed ability to add int or float. Not likely to be useful I think?

        if np.issubclass_(type(other), Spectrum):
            # Some typical errors.
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("abscissae are different for the two Spectrum objects.")
            if not np.array_equal(self.rs, other.rs):
                raise TypeError("redshifts are different for the two Spectrum objects.")

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
        """Adds two ``Spectrum`` instances together, or an array to the spectrum. The Spectrum object is on the right.
        
        Parameters
        ----------
        other : Spectrum, ndarray

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the summed spectrum. 

        Notes
        -----
        This special function, together with `Spectrum.__add__`, allows the use of the symbol + to add ``Spectrum`` objects together.

        The returned Spectrum object `underflow` is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        __add__

        """
        
        # Removed ability to add int or float. Not likely to be useful I think?

        if np.issubclass_(type(other), Spectrum):
            # Some typical errors.
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("abscissae are different for the two Spectrum objects.")
            if not np.array_equal(self.rs, other.rs):
                raise TypeError("redshifts are different for the two Spectrum objects.")

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
        """Subtracts one ``Spectrum`` instance from another, or subtracts an array from the spectrum. 
        
        Parameters
        ----------
        other : Spectrum or ndarray

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the subtracted `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__rsub__`, allows the use of the symbol - to subtract or subtract from ``Spectrum`` objects.

        The returned ``Spectrum`` object nderflow is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        __rsub__

        """
        return self + -1*other

    def __rsub__(self, other):
        """Subtracts one ``Spectrum`` instance from another, or subtracts the spectrum from an array.
        
        Parameters
        ----------
        other : Spectrum or ndarray

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the subtracted `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__sub__`, allows the use of the symbol - to subtract or subtract from ``Spectrum`` objects.

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
            New ``Spectrum`` instance with the spectrum negated. 
        """
        return -1*self

    def __mul__(self,other):
        """Takes the product of the spectrum with an array or number. ``Spectrum`` object is on the left.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the multiplied `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__rmul__`, allows the use of the symbol * to multiply ``Spectrum`` objects or an array and ``Spectrum``.

        The returned ``Spectrum`` object `underflow` is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        __rmul__

        """
        if issubdtype(type(other),float) or issubdtype(type(other),int):
            new_spectrum = Spectrum(self.eng, self.dNdE*other, self.rs)
            new_spectrum.underflow['N'] = self.underflow['N']*other
            new_spectrum.underflow['eng'] = self.underflow['eng']*other
            return new_spectrum

        # Removed ability to multiply two Spectrum objects, doesn't seem like there's a physical reason for us to implement this.

        elif np.isinstance(other, ndarray):

            return Spectrum(self.eng, self.dNdE*other, self.rs)

        else:

            raise TypeError("cannot multiply object to Spectrum.")

    def __rmul__(self,other):
        """Takes the product of the spectrum with an array or number. ``Spectrum`` object is on the right.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the multiplied `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__mul__`, allows the use of the symbol * to multiply ``Spectrum`` objects or an array and ``Spectrum``.

        The returned ``Spectrum`` object `underflow` is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        __mul__

        """
        if issubdtype(type(other),float) or issubdtype(type(other),int):
            new_spectrum = Spectrum(self.eng, self.dNdE*other, self.rs)
            new_spectrum.underflow['N'] = self.underflow['N']*other
            new_spectrum.underflow['eng'] = self.underflow['eng']*other
            return new_spectrum

        # Removed ability to multiply two Spectrum objects, doesn't seem like there's a physical reason for us to implement this.

        elif np.isinstance(other, ndarray):

            return Spectrum(self.eng, self.dNdE*other, self.rs)

        else:

            raise TypeError("cannot multiply object with Spectrum.")
    
    def __truediv__(self,other):
        """Divides the spectrum by an array or number.

        Parameters
        ---------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the divided `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__truediv__`, allows the use of the symbol / to multiply ``Spectrum`` objects or an array and ``Spectrum``.

        The returned ``Spectrum`` object `underflow` is reset to zero.

        """
        return self*(1/other)

    def __rtruediv__(self,other):
        """Divides a number or array by the spectrum.

        Parameters
        ---------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the divided `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__truediv__`, allows the use of the symbol / to multiply ``Spectrum`` objects or an array and ``Spectrum``.

        The returned ``Spectrum`` object `underflow` is reset to zero.

        """
        invSpec = Spectrum(self.eng, 1/self.dNdE, self.rs)
        return other*invSpec

        def contract(self, mat):
            """Performs a dot product on the spectrum with another array.

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

    




