#!/usr/bin/env python
# coding: utf-8

import numpy as np, curvedsky as cs, plottools as pl, healpy as hp, local
from matplotlib.pyplot import *
import warnings
warnings.filterwarnings("ignore")

    
freq = 100
ascale = 0.5

# ptsr mask
Mptsr = local.mask_ptsr(freq)

# bad pixel

Q1 = hp.read_map('/project/projectdirs/cmb/data/planck2018/pr3/frequencymaps/HFI_SkyMap_'+str(freq)+'_2048_R3.01_halfmission-1.fits',field=1)
Q2 = hp.read_map('/project/projectdirs/cmb/data/planck2018/pr3/frequencymaps/HFI_SkyMap_'+str(freq)+'_2048_R3.01_halfmission-2.fits',field=1)
Q  = hp.read_map('/project/projectdirs/cmb/data/planck2018/pr3/frequencymaps/HFI_SkyMap_'+str(freq)+'_2048_R3.01_full.fits',field=1)

Mpix  = local.bad_pixel_mask(Q)
Mpix1 = local.bad_pixel_mask(Q1)
Mpix2 = local.bad_pixel_mask(Q2)

# CO 
Mco = local.mask_co_line(freq)

# combine
rmask  = Mptsr * Mco * Mpix
rmask1 = Mptsr * Mco * Mpix1
rmask2 = Mptsr * Mco * Mpix2

# apodized
nside  = 2048
amask  = cs.utils.apodize(nside, rmask,  ascale)
amask1 = cs.utils.apodize(nside, rmask1, ascale)
amask2 = cs.utils.apodize(nside, rmask2, ascale)

# save
for dtype in ['full','hm1','hm2']:
    aobj = local.analysis(dtype=dtype,wind='base',freq=freq,ascale=ascale)
    hp.fitsfunc.write_map(aobj.famask,amask,overwrite=True)

