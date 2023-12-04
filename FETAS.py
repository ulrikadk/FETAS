# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:38:14 2023

@author: Ulrika

Code created to simulate ALMA images for a given area, point spread function 
and flux interval.

"""
import numpy as np
from astropy.io import fits 
import scipy.signal
import cv2
from tqdm import tqdm
from astropy.modeling.models import Sersic2D

np.random.seed(42)

def gal_number(width, height, s_min, s_max, step=0.001):
        area = width/60*height/60
        alpha = -2.05
        S_star = 10**0.6
        phi = 10**2.85
        diff_n = lambda s: (phi/S_star)*(s/S_star)**alpha*np.exp(-s/S_star)
        s = np.arange(s_min,s_max+step,step)
        nsum = 0   
        n = np.zeros(len(s))
        for i in range(1,len(s)):
            n[i] = step*((diff_n(s[i])+diff_n(s[i-1]))/2)*area
            nsum = nsum+n[i]
        return n, s, nsum



#%%

def sky_image(height, width, filename, rebin_psf=1, gal_min=1, n=0):
    
    """
    Description of the input
    Height: The height of the image in arcmin
    Wdith: The width of the image in arcmin
    Filename: Filename of the PSF that is wanted
    rebin_psf: Rebinning of the PSF. 1 is the original PSF, 2 is half as many bins and so on
    gal_min: Minimum numbers of galaxies wished in each flux bin
    n: Sersic profile parameter. If Sersic profile is not wanted (aka only point sources), this should be zero
    """
       
    # Defining the source
    hdul = fits.open(filename)
    header = hdul[0].header #extracting header
    psf = hdul[0].data[0][0] 
         
    # Normalizing psf values between 0 and 1
    psf_max  = psf.max()
    psf_min = psf.min()
    
    oldrange = (psf_max - psf_min)
    newrange = (1 - 0)
    psf_norm = (((psf - psf_min) * newrange) / oldrange) + 0
    
    # Rebinning the PSF
    psf = cv2.resize(psf_norm,(int(psf.shape[0]/rebin_psf),int(psf.shape[1]/rebin_psf)))
    
    # Converting back to original range
    newrange_rev = (psf_max - psf_min)
    oldrange_rev = (1 - 0)
    psf = (((psf - 0) * newrange_rev) / oldrange_rev) + psf_min
    
        
    #Finding image size in degrees
    CDELT1  =  abs(header["CDELT1"])*rebin_psf  # From psf header
    height_pix = int(height/(CDELT1*60))
    width_pix = int(width/(CDELT1*60))
    
    # Noise level
    noise_std = 0.036296891110335544 # *sqrt(integration)
    
    # Create image for the given size, without any sources
    sky_image = np.zeros( (height_pix, width_pix) )
    
    
    # Defining number of galaxies
    s_min = 0.001 #lowest flux
    s_max = 100   #largest flux
    step = 0.001
    
    try: 
        table = np.loadtxt(f"{height}_{width}_{s_min}_{s_max}.cat")
        print("Catalogue loaded from previous run („• ᴗ •„)")
        print(f"{height}_{width}_{s_min}_{s_max}.cat")
        
        gal_num = table[0]
        flux = table[1]
        
        filename_cat = f"{height}_{width}_{s_min}_{s_max}.cat"
        i_height = filename_cat.index("_")
        height = int(filename_cat[0:i_height])
        
        i_width = filename_cat.index("_",i_height+1)
        width = int(filename_cat[i_height+1:i_width])
        
        i_smin = filename_cat.index("_",i_width+1)
        s_min = float(filename_cat[i_width+1:i_smin])
        
        i_smax = filename_cat.index(".",i_smin+1)
        s_max = float(filename_cat[i_smin+1:i_smax])
        
  
    except:
        print("Generating catalouge, please stand by  (´｡• ᵕ •｡`)")
        gal_num_init, flux_init, total = gal_number(height, width, s_min, s_max, step)
        
        gal_num = []
        flux = []
        temp_flux = []
        temp_arr = []
        
        
        for i in tqdm(range(len(gal_num_init))):
            if flux_init[i] > 0:
                   
                if gal_num_init[i] >= gal_min:
                    gal_num.append(gal_num_init[i])
                    flux.append(flux_init[i])
                          
                else:
                    temp_arr.append(gal_num_init[i])
                    temp_flux.append(flux_init[i])
                
                    if sum(temp_arr) >= gal_min:
                        gal_num.append(sum(temp_arr))
                        temp_flux_max = temp_flux[0]
                        temp_flux_min = temp_flux[-1]
            
                        flux.append(np.random.uniform(temp_flux_min, temp_flux_max))
                        temp_arr = []
                        temp_flux = []
    
        np.savetxt(f"{height}_{width}_{s_min}_{s_max}.cat",np.array([gal_num,flux]))
    
    # Preparing dictionary
    sources = {}
    
    # Insert loop here to insert more sources
    count = 0
    source_count = 1
    
    for i in tqdm(gal_num):
        
        # Loop to define the flux for each galaxy bin
        for j in np.arange(0,i):
            
            if flux[count] > 3*noise_std: #If flux is lower than 3*sigma the galaxy will not be plotted
                
                if n == 0:
                    # Find a random place for the source
                    x_pos = int(np.random.rand(1)*width_pix)
                    y_pos = int(np.random.rand(1)*height_pix)
                
                    sources[f"ID{source_count}"] = f'{x_pos} pix, {y_pos} pix, {flux[count]} mJy'
                    source_count += 1
               
                    # inserting the flux
                    sky_image[x_pos, y_pos] = flux[count]
                
                
                if n > 0: 
                    # Find a random place for the source
                    x_pos = int(np.random.rand(1)*width_pix)
                    y_pos = int(np.random.rand(1)*height_pix)
                    
                    sources[f"ID{source_count}"] = f'{x_pos} pix, {y_pos} pix, {flux[count]} mJy'
                    source_count += 1
                    
                    #Inserting flux with a sersic profile
                    r_eff_arcsec = 10/7.855
                    r_eff = (r_eff_arcsec/3600)/CDELT1
                    #x,y =  np.mgrid[0:int(4*r_eff),0:int(4*r_eff)]
                    x,y =  np.mgrid[0:width_pix,0:height_pix]
                    sersic_model = Sersic2D(amplitude=flux[count]/2/r_eff_arcsec, r_eff=r_eff, n=n, x_0=x_pos, y_0=y_pos, ellip=0, theta=0)
                    
                    sky_image += sersic_model(x,y)
            
        count+=1

    
    # Inserting noise into the image
    Noise = np.random.randn(width_pix,height_pix)*noise_std/(0.62*5)  #Find bedre værdi af 0.62
    Noise[Noise > 5*noise_std] = 5*noise_std
     
    sky_image_points = sky_image
    
    #Adding Noise
    noise_free = sky_image
    sky_image =  sky_image + Noise
    sky_image_noise = sky_image + Noise

    # Convolving the sky image with the psf
    sky_image = scipy.signal.fftconvolve(sky_image,psf,mode="same")
    noise_free = scipy.signal.fftconvolve(noise_free,psf,mode="same")
    
    
    # Saving as a fits file  
    hdu = fits.PrimaryHDU(sky_image)
    hdul = fits.HDUList([hdu])
    hdu.header['CTYPE1'] = 'RA---SIN'
    hdu.header['CTYPE2'] = 'DEC--SIN'
    hdu.header['CUNIT1'] = 'deg'
    hdu.header['CUNIT2'] = 'deg'
    hdu.header['CDELT1'] = -CDELT1 
    hdu.header['CDELT2'] = CDELT1 
    hdu.header['CRVAL1'] = -35
    hdu.header['CRVAL2'] = 35 
    hdu.header['CRPIX2'] = height_pix/2
    hdu.header['CRPIX1'] = width_pix/2
    hdu.header['BSCALE'] = 1.000000000000E+00 
    hdu.header['BUNIT'] = 'Jy/beam' 
    
    
    
    for name in tqdm(sources):
        hdu.header[name] = sources[name]
    hdul.writeto('skyimage_test.fits',overwrite=True)
    
    return sky_image, noise_free, gal_num, flux

#%%

import matplotlib.pyplot as plt

sky_image, noise_free, gal_num, flux = sky_image(2,2,"psf_ost.fits",1,1,0)
plt.scatter(flux, gal_num)
#plt.xlim(0,1)
plt.yscale('log')
plt.xscale('log')


"""
#sky_image, noise_free = sky_image(5,5,"psf_ost.fits",5,1,0)
plt.figure()
plt.imshow(sky_image)
plt.title("ALMA simulated sky image with noise")
plt.colorbar()
plt.xlabel("pix")
plt.ylabel("pix")
"""
plt.figure()
plt.hist(sky_image.flatten(), bins='auto', color="indigo")
plt.xlabel("Flux [mJy/beam]")
plt.ylabel("N pixel")
plt.xlim(-0.5,4)
plt.yscale('log')

"""
plt.figure()
plt.imshow(noise_free)
plt.title("ALMA simulated sky image without noise")
plt.colorbar()
plt.xlabel("pix")
plt.ylabel("pix")
"""


