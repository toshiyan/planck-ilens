# Reconstruction using quadratic estimator
import numpy as np
import local
import tools_cmb
import tools_cib
import tools_qrec

#run_cmb = ['ptsr','alm','aps']
#run_cmb = ['alm','aps']
#run_cmb = ['aps']
run_cmb = []

#run_qrec = ['norm','qrec','mean','aps']
run_qrec = ['mean','aps']
#run_qrec = []

qtypes = ['ilens']
#qtypes = ['lens']
#qtypes = ['ilens','lens']
#qtypes = []

kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

#rlmin, rlmax = 100, 2048
rlmin, rlmax = 100, 1900

# Main calculation

for freq in ['smica','nilc']:    
    for wind in ['G40','G60']:
        for beta, snmin in zip([0.,1.],[0,1]):
            
            kwargs_cmb  = {\
                'snmin':snmin, \
                'snmax':200, \
                'freq':freq, \
                'dtype':'full', \
                'wind':'L'+wind, \
                'ascale':1., \
                'lmax':2048, \
                'fltr':'cinv', \
                'biref':beta \
            }

            kwargs_cib  = {\
                'snmin':kwargs_cmb['snmin'], \
                'snmax':kwargs_cmb['snmax'], \
                'dtype':'gnilc', \
                'freq':'545', \
                #'freq':'857', \
                'wind':wind \
            }

            if run_cmb:
                tools_cmb.interface(run_cmb,kwargs_cmb,kwargs_ov)

            if run_qrec:
                tools_qrec.interface(run=run_qrec,qtypes=qtypes,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_cib=kwargs_cib,rlmin=rlmin,rlmax=rlmax)

