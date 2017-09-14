"""``transferfunction`` contains functions and classes for processing transfer functions."""

import numpy as np
import utilities as utils
import spectrum
from scipy import interpolate

class Transferfunction(spectrum.Spectra):

    

    def at_rs(self, out_rs, interp_type='val'):
        """Returns the interpolation spectrum at a given redshift.

        Interpolation is logarithmic.

        Parameters
        ----------
            out_rs : float
                The redshift (or redshift bin index) at which to interpolate. 
            interp_type : {'val', 'bin'}
                The type of interpolation. 'bin' uses bin index, while 'val' uses the actual redshift. 

        Returns
        -------
        Spectrum
            The interpolated spectrum. 
        """

        gridvalues = np.stack([spec.dNdE for spec in self.spec_arr])

        interp = interpolate.interp2d(self.eng, self.rs, gridvalues)

        return interp(self.eng, out_rs)

