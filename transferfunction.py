"""``transferfunction`` contains functions and classes for processing transfer functions."""

import numpy as np
import utilities as utils
import spectrum
from scipy import interpolate

class Transferfunction(spectrum.Spectra):

    def __init__(self, spec_arr):
        spectrum.Spectra.__init__(self, spec_arr)

    def __iter__(self):
        return iter(self.spec_arr)

    def __getitem__(self,key):
        if np.issubdtype(key, int) or isinstance(key, slice):
            return self.spec_arr[key]
        else:
            raise TypeError("index must be int.")

    def __setitem__(self,key,value):
        if isinstance(key, int):
            if not isinstance(value, (list, tuple)):
                if np.issubclass_(type(value), Spectrum):
                    self.spec_arr[key] = value
                else:
                    raise TypeError("can only add Spectrum.")
            else:
                raise TypeError("can only add one spectrum per index.")
        elif isinstance(key, slice):
            if len(self.spec_arr[key]) == len(value):
                for i,spec in zip(key,value): 
                    if np.issubclass_(type(spec), Spectrum):
                        self.spec_arr[i] = spec
                    else: 
                        raise TypeError("can only add Spectrum.")
            else:
                raise TypeError("can only add one spectrum per index.")
        else:
            raise TypeError("index must be int.")


    def __add__(self, other): 
        
        if np.issubclass_(type(other), Transferfunction):

            if not util.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the two Transferfunction.')
            if not util.array_equal(self.rs, other.rs):
                raise TypeError('redshifts are different for the two Transferfunction.')

            return Transferfunction([spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)])

        else: raise TypeError('adding an object that is not of class Transferfunction.')


    def __radd__(self, other): 
        
        if np.issubclass_(type(other), Transferfunction):

            if not util.array_equal(self.eng, other.eng):
                raise TypeError('abscissae are different for the two Transferfunction.')
            if not util.array_equal(self.rs, other.rs):
                raise TypeError('redshifts are different for the two Transferfunction.')

            return Transferfunction([spec1 + spec2 for spec1,spec2 in zip(self.spec_arr, other.spec_arr)])

        else: raise TypeError('adding an object that is not of class Transferfunction.')

    def __sub__(self, other):
        
        return self + -1*other 

    def __rsub__(self, other):
          
        return other + -1*self

    def __neg__(self):
        
        return -1*self

    def __mul__(self, other):
       
        if np.issubdtype(type(other), float) or np.issubdtype(type(other), int):
            return Transferfunction([other*spec for spec in self])
        elif np.issubclass_(type(other), Transferfunction):
            if self.rs != other.rs or self.eng != other.eng:
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            return Transferfunction([spec1*spec2 for spec1,spec2 in zip(self, other)])
        else:
            raise TypeError("can only multiply Transferfunction or scalars.")

    def __rmul__(self, other):
        
        if np.issubdtype(type(other), float) or np.issubdtype(type(other), int):
            return Transferfunction([other*spec for spec in self])
        elif np.issubclass_(type(other), Transferfunction):
            if self.rs != other.rs or self.eng != other.eng:
                raise TypeError("the two spectra do not have the same redshift or abscissae.")
            return Transferfunction([spec2*spec1 for spec1,spec2 in zip(self, other)])
        else:
            raise TypeError("can only multiply Transferfunction or scalars.")

    def __truediv__(self,other):
        
        if np.issubclass_(type(other), Transferfunction):
            invSpec = Transferfunction([1./spec for spec in other])
            return self*invSpec
        else:
            return self*(1/other)

    def __rtruediv__(self,other):
        
        invSpec = Transferfunction([1./spec for spec in self])

        return other*invSpec

    def at_rs(self, out_rs, interp_type='val'):
        """Returns the interpolation spectrum at a given redshift.

        Interpolation is logarithmic.

        Parameters
        ----------
            out_rs : ndarray
                The redshifts (or redshift bin indices) at which to interpolate. 
            interp_type : {'val', 'bin'}
                The type of interpolation. 'bin' uses bin index, while 'val' uses the actual redshift. 

        Returns
        -------
        Spectra
            The interpolated spectra. 
        """

        gridvalues = np.stack([spec.dNdE for spec in self.spec_arr])

        interp = interpolate.interp2d(self.eng, np.log(self.rs), gridvalues)

        if interp_type == 'val':
            return Transferfunction(
                [spectrum.Spectrum(self.eng, interp(self.eng, np.log(rs)), rs) for rs in out_rs]
                )
        elif interp_type == 'bin':
            log_rs_value = np.interp(out_rs, np.arange(self.rs.size), np.log(self.rs))
            return Transferfunction(
                [spectrum.Spectrum(self.eng, interp(self.eng, log_rs_value), rs) for rs in out_rs]
                )
        else:
            raise TypeError("Invalid interp_type specified.")

