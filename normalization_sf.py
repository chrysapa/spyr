import numpy as np

def divisively_normalize_spatialfreq(data, divisive_exponent=2, saturation_constant=0.1):
    """
    Divisively normalizes data taking into account the previous and following spatial frequency level. 
    Data is the 4D decomposition of an image into spatial frequencies and orientations, such as the result 
    of the steerable pyramid transform.
    """
    numlevels=data.shape[0]
    s = saturation_constant
    r = divisive_exponent
    normalizers=np.sum(data, axis=1)
    
    normalized = np.full(data.shape, 0.0)
    normalized[0]=data[0]**r / ((normalizers[0]+normalizers[1])**r + s**r)
    normalized[numlevels-1]=data[numlevels-1]**r/((normalizers[numlevels-1]+normalizers[numlevels-2])**r+s**r)
    
    inter_levels=range(1,numlevels-1)
    for level in (inter_levels):
            normalizer=normalizers[level] + normalizers[level+1] + normalizers[level-1]
            normalized[level] = (data[level])**r/(normalizer**r+s**r)
    
    return normalized