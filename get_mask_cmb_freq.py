#!/usr/bin/env python
# coding: utf-8

import numpy as np, healpy as hp, local, misctools

ascale = 1.0

#for freq in ['100','143','217','353','smica']:
#for freq in ['smica']:
for freq in ['nilc']:

    # ptsr
    Mptsr = local.mask_ptsr(freq)
    # CO 
    Mco   = local.mask_co_line(freq)

    for dtype in ['full','hm1','hm2']:

        for wind in ['Lens','LG60','LG40','LG20']:
        
            print(freq,dtype,wind)

            aobj = local.init_analysis(dtype=dtype,wind=wind,freq=freq,ascale=ascale)

            if misctools.check_path(aobj.famask,overwrite=True,verbose=False): continue
        
            local.create_mask(aobj,Mptsr,Mco)

