# Reconstruction using quadratic estimator
import numpy as np
import local
import tools_cmb
#import tools_qrec
#import tools_y


def analysis_flow(run_cmb,kwargs_ov={},kwargs_cmb={}):

    kwargs_qrec = {\
        'rlmin':100, \
        'rlmax':2048, \
        'nside':1024, \
        'n0min':1, \
        'n0max':int(kwargs_cmb['snmax']/2), \
        'mfmin':1, \
        'mfmax':kwargs_cmb['snmax'], \
        'rdmin':1, \
        'rdmax':kwargs_cmb['snmax'], \
        'qlist':['TT'], \
        'rd4sim':False \
    }

    kwargs_cinv = {\
        'chn':1, \
        'nsides' : [1024], \
        'lmaxs'  : [2048], \
        'eps'    : [1e-4], \
        'itns'   : [1000] \
    }

    # Main calculation
    if run_cmb:
        tools_cmb.interface(run_cmb,kwargs_cmb,kwargs_ov,kwargs_cinv)

    #if run_qrec:
    #    tools_qrec.interface(qrun=qrun,run=run_qrec,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec)

    #if run_y:
        # loop over y map options
    #    for ytype in ytypes:
    #        for masktype in mtypes:
    #            kwargs_y = {\
    #                'ytype':ytype,\
    #                'ascale':yascale,\
    #                'masktype':masktype,\
    #                'tausig':kwargs_cmb['tausig']\
    #            }
    #            tools_y.interface(run_y,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb,kwargs_qrec=kwargs_qrec,kwargs_y=kwargs_y)



#run_cmb = ['ptsr','alm','aps']
run_cmb = ['alm','aps']
#run_cmb = []

kwargs_ov   = {\
    'overwrite':True, \
    'verbose':True \
}

kwargs_cmb  = {\
    'snmin':0, \
    'snmax':100, \
    'dtype':'full', \
    'wind':'base', \
    'ascale':1., \
    'lmax':2048, \
    'fltr':'none', \
    'biref': 0\
}


analysis_flow(run_cmb,kwargs_ov=kwargs_ov,kwargs_cmb=kwargs_cmb)


