import numpy as np
import math as math
import pandas as pd
import time
from numba import jit

# Constants
def LEVEL(): 
    return 0.95

def L0():
    return 20

def load_data(file_name): 
    data = np.loadtxt(file_name)
    return data.size, np.insert(data, 0, 0, axis=0)

def export_data(series, pointers, nseg):
    
    file1 = open("file1.csv", "w")
    file2 = open("file2.csv", "w")
    file1.write("segment,start,finish,mean,sigma\n")
    file2.write("size,mean,sigma\n")
    for j in np.arange(0, nseg):
        mm = 0.0
        sigma = 0.0
        
        segment = series[int(pointers[j])+1 : int(pointers[j+1])+1]
        mm = segment.mean()
        sigma = ((segment*segment).sum() - mm*mm*segment.size)/(segment.size - 1.0)
        
        
        file1.write("%d,%d,%d,%lg,%lg\n"%(j+1, pointers[j] + 1, pointers[j + 1], mm, sigma))
        file2.write("%d,%lg,%lg\n"%(segment.size, mm, sigma))
    
    file1.close()
    file2.close()

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

def segment():
    #print("Loading Data Time\n")
    #start = time.time()
    series_size, series = load_data("exemplo_nn.dat")
    #end = time.time()
    #print(end - start)
    #print("\n")
    max_seg_number = math.ceil(series_size / L0())
    print("Series Size = ", series_size, "\nL0 (Minimum Segment Size) = ", L0(), "\nMaximum Number of Segments = ceil(n / L0) = ", max_seg_number)
    
    pointers = np.zeros(max_seg_number)
    uu = np.zeros(max_seg_number)
    pointers_temp = np.zeros(max_seg_number)
    uu_temp = np.zeros(max_seg_number)
    
    pointers[1] = series_size
    nseg = 1
    step = 0
    segmenting = 1
    #start = time.time()
    while segmenting == 1:
        step += 1
        new_segments = 0
        segmenting = 0
        
        print("\n------------------------------------------------------\n")
        print("Step ", step, "\n")
        
        pointers_temp[0 : nseg+1] = pointers[0 : nseg+1]
        uu_temp[0 : nseg+1] = uu[0 : nseg+1]
        for j in np.arange(0, nseg + 1):
            print("> Segment [%d]\t Start at: %d\t Non-segmentable: %d\n" % (j+1, pointers[j], uu[j]))
            
        for j in np.arange(0, nseg):
            if(uu[j] == 0):
                dmax, idmax = dksmax(series, series_size, int(pointers[j]) + 1, int(pointers[j + 1]))
                print(">> Maximum Distance (dmax) = %lg\t Segmentation Index (idmax) = %d\n" % (dmax, idmax))
                
                is_size_min = (idmax - pointers[j]) >= L0() and (pointers[j+1] - idmax) >= L0()
                if not is_size_min:
                    uu_temp[j + new_segments] = 1
                    print("Not segmentable: %d..%d < L0\n" %(pointers[j] + 1, pointers[j+1]))
                else:
                    #dcrit = 1.41 * math.exp(0.15*math.log(math.log(po[j+1] - po[j]) - 1.74)) # 90%
                    dcrit = 1.52 * math.exp(0.14*math.log(math.log(pointers[j+1] - pointers[j]) - 1.80)) # 95%
                    #dcrit = 1.72 * math.exp(0.13*math.log(math.log(po[j+1] - po[j]) - 1.86)) # 99%
                    
                    is_significant = dmax > dcrit
                    if not is_significant:
                        uu_temp[j + new_segments] = 1
                        print("Not significant: dmax(%lg) < dcrit(%lg)\n"%(dmax, dcrit))
                    else:
                        segmenting = 1
                        new_segments += 1
                        
                        if(j + new_segments > max_seg_number):
                            print("ERROR\n")
                        
                        pointers_temp = np.insert(pointers_temp, j + new_segments, idmax)
                        uu_temp = np.insert(uu_temp, j + new_segments, 0)
                        
                        flag1 = pointers_temp[j + new_segments + 1] - pointers_temp[j + new_segments] >= 2 * L0()
                        
                        if not flag1:
                            uu_temp[j + new_segments] = 1
                        else:
                            print("Aceito: Starting Index: %d\t Segmentation point : %d\t Ending index: %d\n"%(pointers[j], idmax, pointers[j+1]))
                            
        nseg += new_segments
        pointers[0:nseg+1] = pointers_temp[0 : nseg+1]
        uu[0 : nseg+1] = uu_temp[0 : nseg+1]
            
        print("\n")
        
    #print("Segmenting time\n")
    #end = time.time()
    #print(end - start)
    #print("\n")
    print("Finished.")  
    #print("Exporting data time\n")
    #start = time.time()
    export_data(series, pointers, nseg)
    #end = time.time()
    #print(end - start)
    #print("\n")

start = time.time()

segment()

print("Total time:")
end = time.time()
print(end - start)                
                