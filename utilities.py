import numpy as np

def arrays_equal(nd_array_list):
    same = True
    ind = 0
    while same and ind < len(nd_array_list) - 1:
        same = same & np.array_equal(nd_array_list[ind], 
            nd_array_list[ind+1])
        ind += 1
    return same

def log_spaced(arr):
    return np.ptp(np.diff(np.log(arr)))

def div_ignore_by_zero(a, b, val=0):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        c = np.true_divide(a,b)
        c[~ np.isfinite(c)] = val
    return c