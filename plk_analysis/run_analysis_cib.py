# Reconstruction using quadratic estimator
import numpy as np
import local
import tools_cmb
import tools_cib
import tools_qrec


kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

winds = ['G40','G60']

# Main calculation
for wind in winds:
    
    kwargs_cib  = {\
        'snmin':0, \
        'snmax':200, \
        'dtype':'gnilc', \
        'freq':'545', \
        #'freq':'857', \
        'wind':wind \
    }

    tools_cib.interface(['alm'],kwargs_cib,kwargs_ov)

