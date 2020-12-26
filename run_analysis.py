# Reconstruction using quadratic estimator
import numpy as np
import local
import tools_cmb
import tools_qrec


#run_cmb = ['ptsr','alm','aps']
run_cmb = ['alm','aps']
#run_cmb = ['aps']
#run_cmb = []

run = ['norm','qrec','n0','mean','aps']
#run = ['norm','qrec','n0','mean','rdn0','aps']
#run = ['aps']

qtypes = ['ilens']
#qtypes = []

kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

kwargs_cmb  = {\
    'snmin':0, \
    'snmax':100, \
    'freq':'143', \
    'dtype':'full', \
    'wind':'Lmask', \
    #'wind':'base', \
    'ascale':1., \
    'lmax':2048, \
    'fltr':'none', \
    'biref': 0\
}

rlmin, rlmax = 100, 2048

# Main calculation
if run_cmb:
    tools_cmb.interface(run_cmb,kwargs_cmb,kwargs_ov)

if run_qrec:
    tools_qrec.interface(run=run,qtypes=qtypes,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,rlmin=rlmin,rlmax=rlmax)

