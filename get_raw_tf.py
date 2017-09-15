"""Functions for importing transfer function data.

Abscissa values are subject to change."""

import numpy as np
import physics as phys
import utilities as utils
import spectrum
import transferfunction as tf 

from astropy.io import fits 
from tqdm import tqdm 

def get_out_eng_absc(in_eng):
    """ Returns the output energy abscissa for a given input energy. 

    Parameters
    ----------
    in_eng : float
        Input energy (in eV). 

    Returns
    -------
    ndarray
        Output energy abscissa. 
    """
    log_bin_width = np.log((phys.me + in_eng)/1e-4)/500
    bin_boundary = 1e-4 * np.exp(np.arange(501) * log_bin_width)
    bin_boundary_low = bin_boundary[0:500]
    bin_boundary_upp = bin_boundary[1:501]

    return np.sqrt(bin_boundary_low * bin_boundary_upp)

def process_raw_tf(file):


    #Redshift abscissa. In decreasing order.
    rs_step = 50
    rs_upp  = 31. 
    rs_low  = 4. 

    log_rs_absc = (np.log(rs_low) + (np.arange(rs_step) + 1)
                 *(np.log(rs_upp) - np.log(rs_low))/rs_step)
    log_rs_absc = np.flipud(log_rs_absc)

    # Input energy abscissa. 

    in_eng_step = 500
    low_in_eng_absc = 3e3 + 100.
    upp_in_eng_absc = 5e3 * np.exp(39 * np.log(1e13/5e3) / 40)
    in_eng_absc = low_in_eng_absc * np.exp((np.arange(in_eng_step)) * 
                  np.log(upp_in_eng_absc/low_in_eng_absc) / in_eng_step)

    # Output energy abscissa
    out_eng_absc_arr = np.array([get_out_eng_absc(in_eng) 
                                for in_eng in in_eng_absc])

    # Initial injected bin in output energy abscissa
    init_inj_eng_arr = [out_eng_absc[out_eng_absc < in_eng][-1] 
        for in_eng,out_eng_absc in zip(in_eng_absc, out_eng_absc_arr)
    ]

    # Import raw data. 

    tf_raw = np.load(file)
    tf_raw = np.swapaxes(tf_raw, 0, 1)
    tf_raw = np.swapaxes(tf_raw, 1, 2)
    tf_raw = np.swapaxes(tf_raw, 2, 3)
    tf_raw = np.flip(tf_raw, axis=0)

    # tf_raw has indices (redshift, xe, out_eng, in_eng), redshift in decreasing order.

    # Prepare the output.

    norm_fac = (in_eng_absc/init_inj_eng_arr)*2

    # The transfer function is expressed as a dN/dE spectrum as a result of injecting approximately 2 particles in out_eng_absc[-1]. The exact number is computed and the transfer function appropriately normalized to 1 particle injection (at energy out_eng_absc[-1]).

    tf_raw_list = [
        [spectrum.Spectrum(out_eng_absc_arr[i], tf_raw[j,0,:,i]/norm_fac[i], 
            np.exp(log_rs_absc[j])) for j in np.arange(tf_raw.shape[0])]
        for i in tqdm(np.arange(tf_raw.shape[-1]))
    ]

    transfer_func_table = [
        tf.Transferfunction(spec_arr, init_inj_eng, rebin_eng = init_inj_eng_arr) for init_inj_eng, out_eng_absc, spec_arr in zip(
                init_inj_eng_arr, out_eng_absc_arr, tqdm(tf_raw_list))
    ]

    #Rebin to the desired abscissa, which is in_eng_absc.
    # for spec_list,out_eng_absc in zip(tqdm(tf_raw_list),out_eng_absc_arr):
    #     for spec in spec_list:
    #         spec.rebin(in_eng_absc)
    #     # Note that the injection energy is out_eng_absc[-1] due to our conventions in the high energy code.
    #     transfer_func_table.append(tf.Transferfunction(spec_list, out_eng_absc[-1]))

    return transfer_func_table













