# Reconstruction using quadratic estimator
import numpy as np
import local
import tools_cmb
import tools_cib
import tools_qrec


#run_cmb = ['alm','aps']
#run_cmb = ['aps']
#run_cmb = ['alm']
run_cmb = []

run_qrec = ['norm','qrec','mean','aps']
#run_qrec = ['aps']
#run_qrec = []

qtypes = ['ilens']
#qtypes = ['lens']


kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

kwargs_cib  = {\
    'snmin':1, \
    'snmax':200, \
    'dtype':'gnilc', \
    'freq':'545', \
    #'freq':'857', \
    'wind':'G40' \
}

# Main calculation

kwargs_cmb  = {\
    'snmin':kwargs_cib['snmin'], \
    'snmax':kwargs_cib['snmax'], \
    'freq':'nilc', \
    'dtype':'full', \
    'wind':'L'+kwargs_cib['wind'], \
    'ascale':1., \
    'lmax':2048, \
    'fltr':'cinv', \
    'biref': 1.\
}

#rlmin, rlmax = 100, 2048
rlmin, rlmax = 100, 1900

if run_cmb:
        tools_cmb.interface(run_cmb,kwargs_cmb,kwargs_ov)

if run_qrec:
        tools_qrec.interface(run=run_qrec,qtypes=qtypes,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_cib=kwargs_cib,rlmin=rlmin,rlmax=rlmax)

