import numpy as np
import math as math
from numba import jit

@jit    
def kstwo(left_seg, left_seg_size, right_seg, right_seg_size):
    j1 = 0
    j2 = 0
    fn1 = 0.0
    fn2 = 0.0
    dks = 0.0
    
    left_temp = np.sort(left_seg)
    right_temp = np.sort(right_seg)

    while (j1 < left_seg_size and j2 < right_seg_size):
        d1 = left_temp[j1]
        d2 = right_temp[j2]
        if d1 <= d2:
            fn1 = (j1 + 1)/left_seg_size
            j1 += 1
        if d1 >= d2:
            fn2 = (j2 + 1)/right_seg_size
            j2 += 1
        dks_temp = math.fabs(fn2 - fn1)
        if(dks_temp > dks):
            dks = dks_temp
       
    effective_size = math.sqrt( left_seg_size*right_seg_size / (left_seg_size + right_seg_size) )
   
    return dks*effective_size

    
def dksmax(series, series_size, tLf, tRf):
    dmax = 0.0      # stores the maximum distance
    idmax = 0.0     # stores the index of segmentation
    lc = 1
    tLf = int(tLf)
    tRf = int(tRf)
    
    for k in range(tLf - lc + 1, tRf):
        left_seg_size = k - tLf + 1
        right_seg_size = tRf - k
    
        left_segment = series[tLf : k+1]      # copy data from series to the left segment: from index tLf to k
        right_segment = series[k+1 : tRf+1]     # copy data from series to the right segment: from index k + 1 to tRf

        d = kstwo(left_segment, left_seg_size, right_segment, right_seg_size)
        
        if d > dmax:
            dmax = d
            idmax = k
        
    return [dmax, idmax]