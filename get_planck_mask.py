#!/usr/bin/env python
# coding: utf-8

import numpy as np, curvedsky as cs, healpy as hp, local, warnings, misctools
warnings.filterwarnings("ignore")
    
ascale = 0.5

for freq in ['100','143','217','353']:

    # ptsr mask
    Mptsr = local.mask_ptsr(freq)

    # CO 
    Mco = local.mask_co_line(freq)

    for dtype in ['full','hm1','hm2']:

        aobj = local.init_analysis(dtype=dtype,wind='base',freq=freq,ascale=ascale)

        if misctools.check_path(aobj.famask,overwrite=True,verbose=False): continue
        
        # bad pixel
        Imap  = hp.read_map(aobj.fimap['s'][0],field=0)
        nside = hp.get_nside(Imap)
        Mpix  = local.bad_pixel_mask(Imap)
        rmask = Mptsr * Mco * Mpix

        # apodized
        amask  = cs.utils.apodize(nside, rmask, ascale)

        # save
        hp.fitsfunc.write_map(aobj.famask,amask,overwrite=True)

