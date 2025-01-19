
# general
import numpy as np
import healpy as hp
import pickle
import sys

# cmblensplus/curvedsky/
import curvedsky as cs

# cmblensplus/utils/
import cmb as CMB
import misctools
import analysis as ana


# constants
Tcmb  = 2.726e6       # CMB temperature
ac2rad = np.pi/10800. # arcmin -> rad


def add_noise(lcl,sig,theta,alpha=1.,lBmin=2,split=False,fg=False):
    
    # sig is pol noise level

    ln = len(lcl[0,:])
    L  = np.linspace(0,ln-1,ln)
    
    nl = np.zeros((4,ln))
    nl[0,:] = .5*(sig*ac2rad/Tcmb)**2*np.exp(L*(L+1)*(theta*ac2rad)**2/8./np.log(2.))
    nl[1,:] = 2*nl[0,:]
    nl[2,:] = 2*nl[0,:]
    
    # noise split case
    if split: nl *= 2.
    
    # low-ell limit by e.g. foreground
    nl[2,:lBmin] = 1e30 
    if fg:
        BBfg = 1.4e-5 * 10 ** ( np.log10(2.5e-6/1.4e-5)/np.log10(100./2.)*np.log10((L+1e-30)/2.) )
        nl[1,lBmin:] += BBfg[lBmin:] * (2*np.pi/(L[lBmin:]**2+L[lBmin:]+1e-30)) / CMB.Tcmb**2

    Lcl = lcl.copy()
    Lcl[2,:] *= alpha

    return Lcl + nl


def est_beta(icls,fidcl):
    simn = len(icls[:,0])
    icl = np.array( [ icls[i,:]/fidcl for i in range(simn) ] )
    vcl = np.array( [ np.std(np.delete(icl,i,axis=0),axis=0) for i in range(simn) ] )
    return np.array( [ np.sum(icl[i,:]/vcl[i,:]**2)/np.sum(1./vcl[i,:]**2) for i in range(simn) ] ) 


def rec(Lmax,rlmin,rlmax,lcl,ocl0,ocl1=None,qdo=['BB']):
    Ag = {}
    if 'BB' in qdo:
        Ag['BB'] = cs.norm_imag.qbb('lens',Lmax,rlmin,rlmax,lcl[1],ocl0[2])[0]
    if 'BBa' in qdo:
        Ag['BB'] = cs.norm_imag.qbb_asym('lens',Lmax,rlmin,rlmax,lcl[1],ocl0[2],ocl1[2])[0]
    return Ag


