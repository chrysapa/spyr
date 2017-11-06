import scipy
import os
import math

import numpy as np
from numpy import pi
pi=np.pi
import matplotlib.pyplot as plt
import numpy.linalg
from scipy import ndimage
from scipy import misc
from sklearn.metrics import mean_squared_error
import scipy.stats as st

def log0(x, base=None, zero_val=None):
    x = np.asarray(x) 
    y = np.full(x.shape, 0.0 if zero_val is None else zero_val, dtype=np.float)
    ii = (x > 0)
    y[ii] = np.log(x[ii]) / (1 if base is None else np.log(base))
    return y

def log_raised_cos(r, ctrfreq, bandwidth):
    rarg = (pi / bandwidth) * log0(pi / ctrfreq * r, 2, bandwidth)
    y = np.sqrt(0.5 * (np.cos(rarg) + 1))  
    y[np.where(rarg >= pi)] =0
    y[np.where(rarg <= -pi)] = 0  
    return y

def log_raised_coshi(r, ctrfreq, bandwidth):
    ctrfreq = ctrfreq * math.pow(2, bandwidth)
    rarg = (pi / bandwidth) * log0(pi / ctrfreq * r, 2, -pi)
    y = np.sqrt(0.5*(np.cos(rarg)+1))
    y[np.where(rarg >= 0)] = 1
    y[np.where(rarg <= -pi)] = 0 
    return y

def log_raised_coslo(r, ctrfreq, bandwidth):
    ctrfreq= ctrfreq / math.pow(2, bandwidth)
    rarg = (pi / bandwidth) * log0(pi / ctrfreq * r, 2, 0)
    y = np.sqrt(0.5 * (np.cos(rarg) + 1))
    y[np.where(rarg >= pi)] = 0 
    y[np.where(rarg <= 0)] = 1  
    return y

def freqspace(dim):    
    """Equivalent of Matlab freqspace, frequency spacing for frequency response"""
    f1 = []                   
    if dim % 2 == 0:   
        for i in range(-dim, dim-1, 2):
            ft = float(i) / float(dim)
            f1.append(ft)    
    else:                  
        for i in range(-dim+1, dim, 2):
            ft = float(i) / float(dim)
            f1.append(ft) 
    return f1

def freqspace2(dim):    
    """Equivalent of Matlab freqspace, frequency spacing for frequency response"""
    (minval, maxval) = (-dim, dim-1) if (dim % 2 == 0) else (1-dim, dim)
    return np.asarray(range(minval,maxval,2)) / dim

def make_steer_frs(dims, numlevels, numorientations, bandwidth):
    """Makes the frequency responses of the filters for a multiscale image transform.
    
    Arguments:
    dims -- image shape
    numlevels -- number of levels/scales
    numorientations -- number of orientation subbands at each scale
    bandwidth -- spatial frequency bandwidth in octaves

    Returns array that contains all of the frequency responses. 
    array[0] contains the high frequency response,
    array[1] contains the band frequency responses in the form (numlevel, numorientations, dims),
    and array[2] contains the low frequency response 
    """
    
    result = []
    bands=[]
    p = numorientations-1
    const = math.sqrt(float(math.pow(2,(2*p))*math.pow(math.factorial(p),2)) / float(math.factorial(2*p)*(p+1)))
    f1 = freqspace(dims[0])
    f2 = freqspace(dims[1])
    wx, wy = np.meshgrid(f1, f2)
    size = wx.shape
    r = np.sqrt(wx**2 + wy**2)
    theta = np.arctan2(wy, wx) 
   
    bands = np.full((numlevels, numorientations, dims[0], dims[1]), const*1j)
    for level in range(numlevels):
        for orientation in range(numorientations):
            theta_offset = orientation * np.pi / numorientations
            ctrfreq = pi / math.pow(2, (level+1)*bandwidth)
            band = np.cos(theta - theta_offset)**p * log_raised_cos(r, ctrfreq, bandwidth)
            bands[level,orientation,:,:] *= band
    
    hi = log_raised_coshi(r, pi / math.pow(2, bandwidth), bandwidth)

    lo = log_raised_coslo(r, pi / math.pow(2, bandwidth * numlevels), bandwidth)
    
    result.append(hi)
    result.append(bands)
    result.append(lo)
    return result

def est_maxlevel(dims,bandwidth):
    """Estimate max level for the steerable pyramid"""
    lev = math.floor((math.log(min(dims))/math.log(2)-2)/bandwidth)
    lev=int(lev)
    return lev

def build_steer_bands(im, freq_resps, numlevels, numorientations):
    """ Builds subbands multiscale of a multiscale of a multiscale image transform.
    
    Arguments:
    im -- a grayscale image
    freq_resps -- filter frequency responses returned by make_steer_frs
    numlevels -- number of levels/scales
    numorientations -- number of orientation subbands at each scale

    Returns array that contains all of the subbands. 
    array[0] contains the high band,
    array[1] contains the bands in the form (numlevel, numorientations, dims),
    and array[2] contains the low band
    """
    
    dims = im.shape
    bands = []
    pyr = []
    fourier = np.fft.fftshift(np.fft.fft2(im))
    
    freq_resp_hi = freq_resps[0]    
    hi = np.fft.ifft2(np.fft.fftshift(np.multiply(fourier, freq_resp_hi))).real
    
    freq_resp_lo = freq_resps[2]    
    lo = np.fft.ifft2(np.fft.fftshift(np.multiply(fourier, freq_resp_lo))).real
    
    freq_resp_bands = freq_resps[1]    
    for i in range(numlevels):
        for j in range(numorientations):
            freq_respband = freq_resp_bands[i][j]
            ifourier_band = np.fft.ifft2(np.fft.fftshift(np.multiply(fourier, freq_respband))).real
            bands.append(ifourier_band)
    bands = np.reshape(bands, [numlevels, numorientations, dims[0], dims[1]])
    
    pyr.append(hi)
    pyr.append(bands)
    pyr.append(lo)
    return pyr

def recon_steer_bands(pyr, freq_resps, numlevels, numorientations):
    """Reconstructs an image from the subband transform.
    
    Arguments:
    pyr -- the image transform
    freq_resps -- filter frequency responses returned by make_steer_bands, make_quad_frs_imag or make_quad_frs_real
    numlevels -- number of levels/scales
    numorientations -- number of orientation subbands at each scale
    
    Returns the reconstructed image    
    """
    
    result_bands = np.zeros(pyr[0].shape)

    freq_hi = np.fft.fftshift(np.fft.fft2(pyr[0]))
    result_hi = np.fft.ifft2(np.fft.fftshift(np.multiply(freq_hi, np.conjugate(freq_resps[0])))).real                           
    
    freq_lo = np.fft.fftshift(np.fft.fft2(pyr[2]))
    result_lo = np.fft.ifft2(np.fft.fftshift(np.multiply(freq_lo, np.conjugate(freq_resps[2])))).real
            
    freq_resp_band =  freq_resps[1]
    pyr_band = pyr[1]                        
    for i in range(numlevels):
        for j in range(numorientations):  
            freq_band = np.fft.fftshift(np.fft.fft2(pyr_band[i][j]))
            result_band = np.fft.ifft2(np.fft.fftshift(np.multiply(freq_band, np.conjugate(freq_resp_band[i][j])))).real
            result_bands = result_bands + result_band                         
    result = result_bands + result_hi + result_lo
    return result   

def make_quad_frs_imag(dims,numlevels,numorientations,bandwidth):
    """Makes imaginary frequency responses for the quadrature pairs of "make_steer_frs".
    
    Arguments:
    dims -- image shape
    numlevels -- number of levels/scales
    numorientations -- number of orientation subbands at each scale
    bandwidth -- spatial frequency bandwidth in octaves

    Returns array that contains the imaginary part of the quadrature pair  
    array[0] contains the high frequency response,
    array[1] contains the band frequency responses in the form (numlevel, numorientations, dims),
    and array[2] contains the low frequency response 
    """
    
    freq_resps_imag = make_steer_frs(dims,numlevels,numorientations,bandwidth)
    freq_resps_imag[0] = np.zeros(dims)
    freq_resps_imag[2] = np.zeros(dims)
    return freq_resps_imag

def make_quad_frs_real(dims, numlevels, numorientations, bandwidth):
    """Makes real frequency responses for the quadrature pairs of "make_steer_frs".
    
    Arguments:
    dims -- image shape
    numlevels -- number of levels/scales
    numorientations -- number of orientation subbands at each scale
    bandwidth -- spatial frequency bandwidth in octaves

    Returns array that contains the real part of the quadrature pair  
    array[0] contains the high frequency response,
    array[1] contains the band frequency responses in the form (numlevel, numorientations, dims),
    and array[2] contains the low frequency response 
    """
    freq_resps_real = make_steer_frs(dims,numlevels,numorientations,bandwidth)
    freq_resps_real[1] = abs(freq_resps_real[1]) 
    return freq_resps_real

def build_quad_bands(im, freq_resps_imag, freq_resps_real, numlevels, numorientations):
    """ Builds quadrature pair multiscale subbands.
    
    Arguments:
    im -- grayscale image 
    freq_resps_imag, freq_resps_real -- filter frequency responses returned by make_quad_frs_imag, make_quad_frs_real
    numlevels -- number of levels/scales
    numorientations -- number of orientation subbands at each scale
    
    Returns array that contains the quadrature pair multiscale subbands
    array[0] contains the high band,
    array[1] contains the bands in the form (numlevel, numorientations, dims),
    and array[2] contains the low band
    """
    
    pyr_imag = build_steer_bands(im, freq_resps_imag, numlevels, numorientations)
    pyr_real = build_steer_bands(im, freq_resps_real, numlevels, numorientations)
    pyr = pyr_real + np.multiply(1j, pyr_imag)
    return pyr

def view_abs_spyr_images(to_plot, numlevels, numorientations):
    to_plot0 = abs(to_plot[0])
    to_plot2 = abs(to_plot[2])
    to_plot1 = abs(to_plot[1])

    plt.figure()
    plt.gray()
    plt.imshow(to_plot0)
    for level in range(numlevels):
        for orientation in range(numorientations):
            plt.figure()
            plt.gray()
            plt.imshow(to_plot1[level][orientation])
    plt.figure()
    plt.gray()
    plt.imshow(to_plot2)  
    
def view_real_imag_spyr_images(to_plot, numlevels, numorientations):
    to_plot_band = to_plot[1]
    plt.figure()
    plt.gray()
    plt.imshow(to_plot[0].real)
    for level in range(numlevels):
        for orientation in range(numorientations):
            plt.figure()
            plt.gray()        
            f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
            ax1.imshow(to_plot_band[level][orientation].real)
            ax2.imshow(to_plot_band[level][orientation].imag)
    plt.figure()
    plt.gray()
    plt.imshow(to_plot[2].real) 
    
def view_spyr_images(to_plot, numlevels, numorientations):
    to_plot0=to_plot[0]
    to_plot2=to_plot[2]
    to_plot1=to_plot[1]

    plt.figure()
    plt.gray()
    plt.imshow(to_plot0)
    for level in range(numlevels):
        for orientation in range(numorientations):
            plt.figure()
            plt.gray()
            plt.imshow(to_plot1[level][orientation])
    plt.figure()
    plt.gray()
    plt.imshow(to_plot2) 