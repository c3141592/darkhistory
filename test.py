"""``spectrum`` contains functions and classes for processing spectral data."""
#hello- new branch change
import numpy as np
import utilities as utils
import matplotlib.pyplot as plt
import time
import warnings

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
        """Adds two Spectrum instances together, or an array to the spectrum. The ``Spectrum`` object is on the left.
        
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
        spectrum.Spectrum.__radd__

        """

        # Removed ability to add int or float. Not likely to be useful I think?

        if np.issubclass_(type(other), Spectrum):
            # Some typical errors.
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("abscissae are different for the two ``Spectrum`` objects.")
            if not np.array_equal(self.rs, other.rs):
                raise TypeError("redshifts are different for the two ``Spectrum`` objects.")

            new_spectrum = Spectrum(self.eng, self.dNdE+other.dNdE, self.rs)
            new_spectrum.underflow['N'] = (self.underflow['N'] 
                                          + other.underflow['N'])
            new_spectrum.underflow['eng'] = (self.underflow['eng']
                                            + other.underflow['eng'])

            return new_spectrum

        elif isinstance(other, np.ndarray):

            return Spectrum(self.eng, self.dNdE + other, self.rs)

        else:

            raise TypeError("cannot add object to Spectrum.")

    def __radd__(self, other):
        """Adds two Spectrum instances together, or an array to the spectrum. The ``Spectrum`` object is on the right.
        
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

        The returned ``Spectrum`` object `underflow` is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        spectrum.Spectrum.__add__

        """
        
        # Removed ability to add int or float. Not likely to be useful I think?

        if np.issubclass_(type(other), Spectrum):
            # Some typical errors.
            if not np.array_equal(self.eng, other.eng):
                raise TypeError("abscissae are different for the two ``Spectrum`` objects.")
            if not np.array_equal(self.rs, other.rs):
                raise TypeError("redshifts are different for the two ``Spectrum`` objects.")

            new_spectrum = Spectrum(self.eng, self.dNdE+other.dNdE, self.rs)
            new_spectrum.underflow['N'] = (self.underflow['N'] 
                                          + other.underflow['N'])
            new_spectrum.underflow['eng'] = (self.underflow['eng']
                                            + other.underflow['eng'])

            return new_spectrum

        elif isinstance(other, np.ndarray):

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

        The returned ``Spectrum`` object underflow is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        spectrum.Spectrum.__rsub__

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
            New ``Spectrum`` instance which has the subtracted `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__sub__`, allows the use of the symbol - to subtract or subtract from ``Spectrum`` objects.

        See Also
        --------
        spectrum.Spectrum.__sub__

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
        """Takes the product of the spectrum with an array or number. `Spectrum` object is on the left.

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the multiplied `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__rmul__`, allows the use of the symbol * to multiply ``Spectrum`` objects or an array and Spectrum.

        The returned ``Spectrum`` object `underflow` is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        spectrum.Spectrum.__rmul__

        """
        if np.issubdtype(type(other),float) or np.issubdtype(type(other),int):
            new_spectrum = Spectrum(self.eng, self.dNdE*other, self.rs)
            new_spectrum.underflow['N'] = self.underflow['N']*other
            new_spectrum.underflow['eng'] = self.underflow['eng']*other
            return new_spectrum

        # Removed ability to multiply two ``Spectrum`` objects, doesn't seem like there's a physical reason for us to implement this.

        elif isinstance(other, np.ndarray):

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
            New ``Spectrum`` instance which has the multiplied `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__mul__`, allows the use of the symbol * to multiply ``Spectrum`` objects or an array and Spectrum.

        The returned ``Spectrum`` object `underflow` is reset to zero if `other` is not a ``Spectrum`` object.

        See Also
        --------
        spectrum.Spectrum.__mul__

        """
        if np.issubdtype(type(other),float) or np.issubdtype(type(other),int):
            new_spectrum = Spectrum(self.eng, self.dNdE*other, self.rs)
            new_spectrum.underflow['N'] = self.underflow['N']*other
            new_spectrum.underflow['eng'] = self.underflow['eng']*other
            return new_spectrum

        # Removed ability to multiply two ``Spectrum`` objects, doesn't seem like there's a physical reason for us to implement this.

        elif isinstance(other, np.ndarray):

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
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectrum
            New ``Spectrum`` instance which has the divided `dNdE`. 

        Notes
        -----
        This special function, together with `Spectrum.__truediv__`, allows the use of the symbol / to multiply ``Spectrum`` objects or an array and Spectrum.

        The returned ``Spectrum`` object `underflow` is reset to zero.

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
            The type of bounds to use. Bound values do not have to be within the [0:length] for `'bin'` or within the abscissa for `'eng'`. `None` should only be used when computing the total particle number in the spectrum. For `'bin'`, bounds are specified as the bin boundary, with 0 being the left most boundary, 1 the right-hand of the first bin and so on. This is equivalent to integrating over a histogram. For `'eng'`, bounds are specified by energy values.

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

                    N_in_bin[i] = N_full_bins + N_part_bins

                return N_in_bin

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
            The type of bounds to use. Bound values do not have to be within the [0:length] for `'bin'` or within the abscissa for `'eng'`. `None` should only be used to obtain the total energy. With `'bin'`, bounds are specified as the bin boundary, with 0 being the left most boundary, 1 the right-hand of the first bin and so on. This is equivalent to integrating over a histogram. With `'eng'`, bounds are specified by energy values. 

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

                    eng_in_bin[i] = eng_full_bins + eng_part_bins

                return eng_in_bin

            if bound_type == 'eng':

                eng_bin_ind = np.interp( 
                    np.log(bound_arr), 
                    np.log(self.bin_boundary), np.arange(bin_boundary.size), 
                    left = -1, right = length + 1)

                return self.toteng('bin', eng_bin_ind)

        else:
            return (np.dot(dNdlogE, eng * log_bin_width) 
                + self.underflow['eng'])

    def rebin(self, out_eng):
        """ Re-bins the ``Spectrum`` object according to a new abscissa.

        Rebinning conserves total number and total energy.
        
        Parameters
        ----------
        out_eng : ndarray
            The new abscissa to bin into. If `self.eng` has values that are smaller than `out_eng[0]`, then the new underflow will be filled. If `self.eng` has values that exceed `out_eng[-1]`, then an error is returned.

        Raises
        ------
        OverflowError
            The maximum energy in `out_eng` cannot be smaller than any bin in `self.eng`. 

        
        Note
        ----
        The total number and total energy is conserved by assigning the number of particles N in a bin of energy eng to two adjacent bins in new_eng, with energies eng_low and eng_upp such that eng_low < eng < eng_upp. Then dN_low_dE_low = (eng_upp - eng)/(eng_upp - eng_low)*(N/(E * dlogE_low)), and dN_upp_dE_upp = (eng - eng_low)/(eng_upp - eng_low)*(N/(E*dlogE_upp)).

        If a bin in `self.eng` is below the lowest bin in `out_eng`, then the total number and energy not assigned to the lowest bin are assigned to the underflow. Particles will only be assigned to the lowest bin if there is some overlap between the bin index with respect to `out_eng` bin centers is larger than -1.0.

        If a bin in `self.eng` is above the highest bin in `out_eng`, then an `OverflowError` is thrown. 

        See Also
        --------
        spectrum.rebin_N_arr

        """
        if not np.all(np.diff(out_eng) > 0):
            raise TypeError("new abscissa must be ordered in increasing energy.")
        if out_eng[-1] < self.eng[-1]:
            raise OverflowError("the new abscissa lies below the old one: this function cannot handle overflow (yet?).")
        # Get the bin indices that the current abscissa (self.eng) corresponds to in the new abscissa (new_eng). Can be any number between 0 and self.length-1. Bin indices are wrt the bin centers.

        # Add an additional bin at the lower end of out_eng so that underflow can be treated easily.

        first_bin_eng = np.exp(np.log(out_eng[0]) - (np.log(out_eng[1]) - np.log(out_eng[0])))
        new_eng = np.insert(out_eng, 0, first_bin_eng)



        # Find the relative bin indices for self.eng wrt new_eng. The first bin in new_eng has bin index -1. 
        bin_ind = np.interp(self.eng, new_eng, 
            np.arange(new_eng.size)-1, left = -2, right = new_eng.size)

        # Locate where bin_ind is below 0, above self.length-1 and in between.
        ind_low = np.where(bin_ind < 0)
        ind_high = np.where(bin_ind == new_eng.size)
        ind_reg = np.where( (bin_ind >= 0) & (bin_ind <= new_eng.size - 1) )

        if ind_high[0].size > 0: 
            raise OverflowError("the new abscissa lies below the old one: this function cannot handle overflow (yet?).")


        # Filters to pick out the correct parts of arrays. Correct indices are set to 1, incorrect indices set to 0. 
        # filter_low = np.where(bin_ind < 0, np.ones(self.length), 
        #     np.zeros(self.length))
        # filter_high = np.where(bin_ind == self.length, np.ones(self.length), 
        #     np.zeros(self.length))
        # filter_reg = np.where( (bin_ind >= 0) | (bin_ind <= self.length-1), 
        #     np.ones(self.length),np.zeros(self.length))

        # Get the total N and toteng in each bin of self.dNdE
        N_arr = self.totN('bin', np.arange(self.length + 1))
        toteng_arr = self.toteng('bin', np.arange(self.length + 1))

        # N_arr_low = N_arr * filter_low
        # N_arr_high = N_arr * filter_high
        # N_arr_reg = N_arr * filter_reg

        N_arr_low = N_arr[ind_low]
        N_arr_high = N_arr[ind_high]
        N_arr_reg = N_arr[ind_reg]

        toteng_arr_low = toteng_arr[ind_low]

        # Bin width of the new array. Use only the log bin width, so that dN/dE = N/(E d log E)
        new_E_dlogE = new_eng * np.diff(np.log(get_bin_bound(new_eng)))

        # Regular bins first, done in a completely vectorized fashion. 

        # reg_bin_low is the array of the lower bins to be allocated the particles in N_arr_reg, similarly reg_bin_upp. This should also take care of the fact that bin_ind is an integer.
        reg_bin_low = np.floor(bin_ind[ind_reg]).astype(int)
        reg_bin_upp = reg_bin_low + 1

        reg_N_low = (reg_bin_upp - bin_ind[ind_reg]) * N_arr_reg
        reg_N_upp = (bin_ind[ind_reg] - reg_bin_low) * N_arr_reg

        reg_dNdE_low = ((reg_bin_upp - bin_ind[ind_reg]) * N_arr_reg
                       /new_E_dlogE[reg_bin_low+1])
        reg_dNdE_upp = ((bin_ind[ind_reg] - reg_bin_low) * N_arr_reg
                       /new_E_dlogE[reg_bin_upp+1])


        # Low bins. 
        low_bin_low = np.floor(bin_ind[ind_low]).astype(int)
                      
        N_above_underflow = np.sum((bin_ind[ind_low] - low_bin_low) 
            * N_arr_low)
        eng_above_underflow = N_above_underflow * new_eng[1]

        N_underflow = np.sum(N_arr_low) - N_above_underflow
        eng_underflow = np.sum(toteng_arr_low) - eng_above_underflow
        low_dNdE = N_above_underflow/new_E_dlogE[1]


        # print("reg_bin_low: ", reg_bin_low)
        # print("reg_N_low: ", reg_N_low)
        # print("reg_dNdE_low: ", reg_dNdE_low)
        # print("reg_bin_upp: ", reg_bin_upp)
        # print("reg_N_upp: ", reg_N_upp)
        # print("reg_dNdE_upp: ", reg_dNdE_upp)
        # print("bin_ind[ind_reg]", bin_ind[ind_reg])
        # print("new_eng: ", new_eng)
        # print("bin_ind[ind_low]", bin_ind[ind_low])
        # print("N_arr_low: ", N_arr_low)
        # print("toteng_arr_low: ", toteng_arr_low)
        # print("high_dNdE_low: ", high_dNdE_low)
        # print("high_dNdE_upp: ", high_dNdE_upp)
        # print("low_dNdE: ", low_dNdE)

        # Add up, obtain the new dNdE. 
        new_dNdE = np.zeros(new_eng.size)
        new_dNdE[1] += low_dNdE
        # reg_dNdE_low = -1 refers to new_eng[0]  
        for i,ind in zip(np.arange(reg_bin_low.size), reg_bin_low):
            new_dNdE[ind+1] += reg_dNdE_low[i]
        for i,ind in zip(np.arange(reg_bin_upp.size), reg_bin_upp):
            new_dNdE[ind+1] += reg_dNdE_upp[i]

        # Implement changes.
        self.eng = new_eng[1:]
        self.dNdE = new_dNdE[1:]
        self.length = self.eng.size 
        self.bin_boundary = get_bin_bound(self.eng)
        self.log_bin_width = np.diff(np.log(self.bin_boundary))

        self.underflow['N'] += N_underflow
        self.underflow['eng'] += eng_underflow 

    def redshift(self, new_rs):
        """Redshifts the ``Spectrum`` object as a photon spectrum. 

        Parameters
        ----------
        new_rs : float
            The new redshift (1+z) to redshift to.

        """

        fac = new_rs/self.rs
        
        eng_orig = self.eng

        self.eng = self.eng*fac
        self.dNdE = self.dNdE/fac
        self.underflow['eng'] *= fac
        self.length = self.eng.size 
        self.bin_boundary = get_bin_bound(self.eng)
        self.log_bin_width = np.diff(np.log(self.bin_boundary))

        self.rebin(eng_orig)
        self.rs = new_rs





class Spectra:
    """Structure for a collection of ``Spectrum`` objects.

    Parameters
    ----------
    rs : ndarray
        The redshifts of the ``Spectrum`` objects.
    eng : ndarray
        Energy abscissa for the ``Spectrum``. 
    spec_arr : list of ``Spectrum``
        List of ``Spectrum`` to be stored together.

    Attributes
    ----------

    """
    # __array_priority__ must be larger than 0, so that radd can work.
    # Otherwise, ndarray + Spectrum works by iterating over the elements of
    # ndarray first, which isn't what we want.
    __array_priority__ = 1

    def __init__(self, eng, spec_arr, rs):

        if len(set(spec.length for spec in spec_arr)) > 1:
            raise TypeError("all spectra must have the same length.")

        if len(spec_arr) != rs.size:
            raise TypeError("number of spectra must be equal to redshift array length.")

        if not np.all(np.diff(eng) > 0):
            raise TypeError("abscissa must be ordered in increasing energy.")

        if not np.all(np.diff(rs) <= 0):
            raise TypeError("redshift must be in increasing order.")

        self.eng = eng 
        self.rs  = rs
        self.spec_arr = spec_arr

        self.bin_boundary = get_bin_bound(self.eng)
        self.log_bin_width = np.diff(np.log(self.bin_boundary))

    @classmethod
    def from_Spectrum_list(cls, spectrum_list):
        if not utils.array_equal([spec.eng for spec in spectrum_list]):
            raise TypeError("energy abscissa for the Spectrum objects are not the same.")
        eng = spectrum_list[0].eng
        dNdE_list = [spec.dNdE for spec in spectrum_list]
        rs = np.array([spec.rs for spec in spectrum_list])
        return cls(eng, dNdE_list, rs)

    def __add__(self, other): 
        """Adds two ``Spectra`` instances together.

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New ``Spectra`` instance which is an element-wise sum of the ``Spectrum`` objects in each ``Spectra``.

        Notes
        -----
        This special function, together with `Spectra.__radd__`, allows the use of the symbol + to add ``Spectra`` objects together. 

        See Also
        --------
        spectrum.Spectra.__radd__
        """
        if np.issubclass_(type(other), Spectra):

            if not util.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the two Spectra.')
            if not util.array_equal(self.rs, other.rs):
                raise TypeError('redshifts are different for the two Spectra.')

            return Spectra(self.eng, [spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)], self.rs)

        else: raise TypeError('adding an object that is not of class Spectra.')


    def __radd__(self, other): 
        """Adds two ``Spectra`` instances together.

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New ``Spectra`` instance which is an element-wise sum of the ``Spectrum`` objects in each ``Spectra``.

        Notes
        -----
        This special function, together with `Spectra.__add__`, allows the use of the symbol + to add `Spectra` objects together. 

        See Also
        --------
        spectrum.Spectra.__add__
        """
        if np.issubclass_(type(other), Spectra):

            if not util.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the two Spectra.')
            if not util.array_equal(self.rs, other.rs):
                raise TypeError('redshifts are different for the two Spectra.')

            return Spectra(self.eng, [spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)], self.rs)

        else: raise TypeError('adding an object that is not of class Spectra.')

    def __sub__(self, other):
        """Subtracts one ``Spectra`` instance from another. 

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New ``Spectra`` instance which has the subtracted list of `dNdE`. 

        Notes
        -----
        This special function, together with `Spectra.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectra` objects. 

        See Also
        --------
        spectrum.Spectra.__rsub__
        """
        return self + -1*other 

    def __rsub__(self, other):
        """Subtracts one ``Spectra`` instance from another. 

        Parameters
        ----------
        other : Spectra

        Returns
        -------
        Spectra
            New ``Spectra`` instance which has the subtracted list of `dNdE`. 

        Notes
        -----
        This special function, together with `Spectra.__rsub__`, allows the use of the symbol - to subtract or subtract from `Spectra` objects. 

        See Also
        --------
        spectrum.Spectra.__sub__
        """    
        return other + -1*self

    def __neg__(self):
        """Negates all of the `dNdE`. 

        Returns
        -------
        Spectra
            New ``Spectra`` instance with the `dNdE` negated.
        """
        return -1*self

    def __mul__(self, other):
        """Takes the product of two ``Spectra`` instances. 

        Parameters
        ----------
        other : Spectra, int or float

        Returns
        -------
        Spectra
            New ``Spectra`` instance which is an element-wise product of the ``Spectrum`` objects in each ``Spectra``. 

        Notes
        -----
        This special function, together with `Spectra.__rmul__`, allows the use of the symbol * to add ``Spectra`` objects together.

        See Also
        --------
        spectrum.Spectra.__rmul__
        """
        if np.issubdtype(type(other), float) or np.issubdtype(type(other), int):
            return Spectra(self.eng, [other*spec for spec in self.spec_arr], self.rs)
        elif np.issubclass_(type(other), Spectra):
            if self.rs is not other.rs or self.eng is not other.eng:
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            return Spectra(self.eng, [spec1*spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)], self.rs)
        else:
            raise TypeError("can only multiply Spectra or scalars.")

    def __rmul__(self, other):
        """Takes the product of two ``Spectra`` instances. 

        Parameters
        ----------
        other : Spectra, int or float

        Returns
        -------
        Spectra
            New ``Spectra`` instance which is an element-wise product of the ``Spectrum`` objects in each ``Spectra``. 

        Notes
        -----
        This special function, together with `Spectra.__mul__`, allows the use of the symbol * to add ``Spectra`` objects together.

        See Also
        --------
        spectrum.Spectra.__mul__
        """
        if np.issubdtype(type(other), float) or np.issubdtype(type(other), int):
            return Spectra(self.eng, [other*spec for spec in self.spec_arr], self.rs)
        elif np.issubclass_(type(other), Spectra):
            if self.rs is not other.rs or self.eng is not other.eng:
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            return Spectra(self.eng, [spec2*spec1 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)], self.rs)
        else:
            raise TypeError("can only multiply Spectra or scalars.")

    def __truediv__(self,other):
        """Divides ``Spectra`` by a number or another ``Spectra``. 

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__rtruediv__`, allows the use of the symbol / to divide ``Spectra`` objects by a number or another ``Spectra``. 

        See Also
        --------
        spectrum.Spectra.__rtruediv__
        """
        if np.issubclass_(type(other), Spectra):
            invSpec = Spectra(other.eng, [1./spec for spec in other.spec_arr], other.rs)
            return self*invSpec
        else:
            return self*(1/other)

    def __rtruediv__(self,other):
        """Divides ``Spectra`` by a number or another ``Spectra``. 

        Parameters
        ----------
        other : ndarray, float or int

        Returns
        -------
        Spectra

        Notes
        -----
        This special function, together with `Spectra.__truediv__`, allows the use of the symbol / to divide ``Spectra`` objects by a number or another ``Spectra``. 

        See Also
        --------
        spectrum.Spectra.__truediv__
        """
        invSpec = Spectra(self.eng, [1./spec for spec in self.spec_arr], self.rs)

        return other*invSpec   

    def sum_weight_eng(self,mat):
        """Sums each ``Spectrum``, each `eng` bin weighted by `mat`. 

        Equivalent to contracting `mat` with each `dNdE` in ``Spectra``, `mat` should have length `self.length`. 

        Parameters
        ----------
        mat : ndarray
            The weight in each energy bin. 

        Returns
        -------
        ndarray
            An array of weighted sums, one for each redshift in `self.rs`, with length `self.rs.size`. 
        """
        if isinstance(mat,np.ndarray):
            return np.array([spec.contract(mat) for spec in self.spec_arr])

        else:
            raise TypeError("mat must be an ndarray.")

    def sum_weight_rs(self,mat):
        """Sums the spectrum in each energy bin, weighted by `mat`. 

        Equivalent to contracting `mat` with `[spec.dNdE[i] for spec in spec_arr]` for all `i`. `mat` should have length `self.rs.size`. 

        Parameters
        ----------
        mat : ndarray
            The weight in each redshift bin. 

        Returns
        -------
        ndarray
            An array of weight sums, one for each energy in `self.eng`, with length `self.length`. 

        """
        if isinstance(mat,np.ndarray):
            dNdE_to_sum = [mat[i]*spec.dNdE for i,spec in zip(np.arange(mat.size), self.spec_arr)]
            return np.sum(dNdE_to_sum, axis=0)

        else:
            raise TypeError("mat must be an ndarray.")

    def append(self, spec):
        """Appends a new ``Spectrum``. 

        Parameters
        ----------
        spec : Spectrum
        """
        if not array_equal(self.eng, spec.eng):
            raise TypeError("new Spectrum does not have the same energy abscissa.")
        if self.rs.size > 0 and self.rs[-1] < spec.rs: 
            raise TypeError("new Spectrum has a larger redshift than the current last entry.")

        self.spec_arr.append(spec)
        self.rs = np.append(self.rs, spec.rs)

    def plot(self, ind, step=1):
        """Plots the contained ``Spectrum`` objects. 

        Parameters
        ----------
        ind : int or tuple of int
            Index of ``Spectrum`` to plot, or a tuple of indices providing a range of ``Spectrum`` to plot. 

        step : int, optional
            The number of steps to take before choosing one ``Spectrum`` to plot. 
        """
        if np.issubdtype(type(ind), int):
            plt.plot(self.eng, self.spec_arr[ind].dNdE)
        elif np.issubdtype(type(ind), tuple):
            spec_to_plot = np.stack([self.spec_arr[i].dNdE for i in np.arange(ind[0], ind[1], step)], 
                axis=-1)
            plt.plot(self.eng, spec_to_plot)
        else:
            raise TypeError("ind should be either an integer or a tuple of integers.")


    

######################## MODULE METHODS ########################

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

def rebin_N_arr(N_arr, in_eng, out_eng):
    """Rebins an array of particle number with fixed energy.
    
    Returns a ``Spectrum`` object. The rebinning conserves both total number and total energy.

    Parameters
    ----------
    N_arr : ndarray
        An array of number of particles in each bin. 
    in_eng : ndarray
        An array of the energy abscissa for each bin. The total energy in each bin `i should be `N_arr[i]*in_eng[i]`.
    out_eng : ndarray
        The new abscissa to bin into. If `in_eng` has values that are smaller than 

    Returns
    -------
    Spectrum
        The output ``Spectrum`` with appropriate dN/dE, with abscissa out_eng.

    Raises
    ------
    OverflowError
        The maximum energy in `out_eng` cannot be smaller than any bin in `self.eng`.

    Note
    ----
    The total number and total energy is conserved by assigning the number of particles N in a bin of energy eng to two adjacent bins in new_eng, with energies eng_low and eng_upp such that eng_low < eng < eng_upp. Then dN_low_dE_low = (eng_upp - eng)/(eng_upp - eng_low)*(N/(E * dlogE_low)), and dN_upp_dE_upp = (eng - eng_low)/(eng_upp - eng_low)*(N/(E*dlogE_upp)).

    If a bin in `in_eng` is below the lowest bin in `out_eng`, then the total number and energy not assigned to the lowest bin are assigned to the underflow. Particles will only be assigned to the lowest bin if there is some overlap between the bin index with respect to `out_eng` bin centers is larger than -1.0.

    If a bin in `in_eng` is above the highest bin in `out_eng`, then an `OverflowError` is thrown. 

    See Also
    --------
    spectrum.Spectrum.rebin
    """
    if N_arr.size is not in_eng.size:
        raise TypeError("The array for number of particles has a different length from the abscissa.")

    if not np.all(np.diff(out_eng) > 0):
        raise TypeError("new abscissa must be ordered in increasing energy.")
    if out_eng[-1] < in_eng[-1]:
        raise OverflowError("the new abscissa lies below the old one: this function cannot handle overflow (yet?).")
    # Get the bin indices that the current abscissa (self.eng) corresponds to in the new abscissa (new_eng). Can be any number between 0 and self.length-1. Bin indices are wrt the bin centers.

    # Add an additional bin at the lower end of out_eng so that underflow can be treated easily.

    first_bin_eng = np.exp(np.log(out_eng[0]) - (np.log(out_eng[1]) - np.log(out_eng[0])))
    new_eng = np.insert(out_eng, 0, first_bin_eng)

    # Find the relative bin indices for self.eng wrt new_eng. The first bin in new_eng has bin index -1. 
    bin_ind = np.interp(in_eng, new_eng, 
        np.arange(new_eng.size)-1, left = -2, right = new_eng.size)

    # Locate where bin_ind is below 0, above self.length-1 and in between.
    ind_low = np.where(bin_ind < 0)
    ind_high = np.where(bin_ind == new_eng.size)
    ind_reg = np.where( (bin_ind >= 0) & (bin_ind <= new_eng.size - 1) )

    if ind_high[0].size > 0: 
        raise OverflowError("the new abscissa lies below the old one: this function cannot handle overflow (yet?).")

    # Get the total N and toteng in each bin
    toteng_arr = N_arr*in_eng

    N_arr_low = N_arr[ind_low]
    N_arr_high = N_arr[ind_high]
    N_arr_reg = N_arr[ind_reg]

    toteng_arr_low = toteng_arr[ind_low]

    # Bin width of the new array. Use only the log bin width, so that dN/dE = N/(E d log E)
    new_E_dlogE = new_eng * np.diff(np.log(get_bin_bound(new_eng)))

    # Regular bins first, done in a completely vectorized fashion. 

    # reg_bin_low is the array of the lower bins to be allocated the particles in N_arr_reg, similarly reg_bin_upp. This should also take care of the fact that bin_ind is an integer.
    reg_bin_low = np.floor(bin_ind[ind_reg]).astype(int)
    reg_bin_upp = reg_bin_low + 1

    reg_N_low = (reg_bin_upp - bin_ind[ind_reg]) * N_arr_reg
    reg_N_upp = (bin_ind[ind_reg] - reg_bin_low) * N_arr_reg

    reg_dNdE_low = ((reg_bin_upp - bin_ind[ind_reg]) * N_arr_reg
                   /new_E_dlogE[reg_bin_low+1])
    reg_dNdE_upp = ((bin_ind[ind_reg] - reg_bin_low) * N_arr_reg
                   /new_E_dlogE[reg_bin_upp+1])

    # Low bins. 
    low_bin_low = np.floor(bin_ind[ind_low]).astype(int)
                  
    N_above_underflow = np.sum((bin_ind[ind_low] - low_bin_low) 
        * N_arr_low)
    eng_above_underflow = N_above_underflow * new_eng[1]

    N_underflow = np.sum(N_arr_low) - N_above_underflow
    eng_underflow = np.sum(toteng_arr_low) - eng_above_underflow
    low_dNdE = N_above_underflow/new_E_dlogE[1]

    new_dNdE = np.zeros(new_eng.size)
    new_dNdE[1] += low_dNdE
    # reg_dNdE_low = -1 refers to new_eng[0]  
    for i,ind in zip(np.arange(reg_bin_low.size), reg_bin_low):
        new_dNdE[ind+1] += reg_dNdE_low[i]
    for i,ind in zip(np.arange(reg_bin_upp.size), reg_bin_upp):
        new_dNdE[ind+1] += reg_dNdE_upp[i]

    # Generate the new Spectrum.

    out_spec = Spectrum(new_eng[1:], new_dNdE[1:])
    out_spec.underflow['N'] += N_underflow
    out_spec.underflow['eng'] += eng_underflow

    return out_spec

def discretize(func_dNdE, eng):
    """Discretizes a continuous function. 

    The function is integrated between the bin boundaries specified by `eng` to obtain the discretized spectrum, so that the final spectrum conserves number and energy between the bin **boundaries**. 

    Parameters
    ----------
    func_dNdE : function
        A single variable function that takes in energy as an input, and then returns a dN/dE spectrum value. 
    eng : ndarray
        Both the bin boundaries to integrate between and the new abscissa after discretization (bin centers). 

    Returns
    -------
    Spectrum
        The discretized spectrum. rs is set to -1, and must be set manually. 

    Notes
    -----

    """
    def func_EdNdE(eng):
        return func_dNdE(eng)*eng

    # Generate a list of particle number N and mean energy eng_mean, so that N*eng_mean = total energy in each bin. eng_mean != eng. 
    N = np.zeros(eng.size)
    eng_mean = np.zeros(eng.size)
    
    for low, upp, i in zip(eng[:-1], eng[1:], 
        np.arange(eng.size-1)):
    # Perform an integral over the spectrum for each bin.
        N[i] = integrate.quad(func_dNdE, low, upp)[0]
    # Get the total energy stored in each bin. 
        if N[i] > 0:
            eng_mean[i] = integrate.quad(func_EdNdE, low, upp)[0]/N[i]
        else:
            eng_mean[i] = 0


    return rebin_N_arr(N, eng_mean, eng)









