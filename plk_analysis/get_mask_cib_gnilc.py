#!/usr/bin/env python
# coding: utf-8

import numpy as np, curvedsky as cs, healpy as hp, local, warnings, misctools, tools_cib
warnings.filterwarnings("ignore")
    
ascale = 1.0

for freq in ['545','857']:

    for field, wind in zip([0,1,2,3],['G20','G40','G60','G70']):
        
        iobj = tools_cib.init_cib(wind=wind,ascale=ascale)

        if misctools.check_path([iobj.fmask,iobj.famask],overwrite=False,verbose=False): continue
        
        # bad pixel
        Imask = hp.fitsfunc.read_map('/global/homes/t/toshiyan/scratch/PR2/mask/HFI_Mask_GalPlane-apo0_2048_R2.00.fits',field=field)
        nside = hp.get_nside(Imask)

        # apodized
        amask = cs.utils.apodize(nside, Imask, ascale)

        # save
        hp.fitsfunc.write_map(iobj.fmask,Imask,overwrite=True)
        hp.fitsfunc.write_map(iobj.famask,amask,overwrite=True)

