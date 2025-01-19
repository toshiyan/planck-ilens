#!/usr/bin/env python
# coding: utf-8

import numpy as np, basic, curvedsky as cs, binning as bn, cmb as CMB, tqdm, misctools, healpy as hp, analysis as ana
from matplotlib.pyplot import *
Tcmb  = 2.726e6    # CMB temperature
ac2rad = np.pi/180./60.


def sim(i,lmax,rlmin,rlmax,Ag,Ac,fcl,ocl,nl,ilmax=4096,nside=2048,read_so_data=True):
    
    if read_so_data:
        alms = hp.read_alm('/project/projectdirs/sobs/v4_sims/mbs/cmb/fullskyLensedUnabberatedCMB_alm_set00_'+str(i).zfill(5)+'.fits',hdu=(1,2,3))
        Talm = cs.utils.lm_healpy2healpix( alms[0], 5100 ) [:lmax+1,:lmax+1] / Tcmb
        Ealm = cs.utils.lm_healpy2healpix( alms[1], 5100 ) [:lmax+1,:lmax+1] / Tcmb
        Balm = cs.utils.lm_healpy2healpix( alms[2], 5100 ) [:lmax+1,:lmax+1] / Tcmb
        palm = hp.read_alm('/global/project/projectdirs/sobs/v4_sims/mbs/cmb/input_phi/fullskyPhi_alm_'+str(i).zfill(5)+'.fits')
        palm = cs.utils.lm_healpy2healpix( palm, 5100 ) [:lmax+1,:lmax+1]
    else:
        Ucl = CMB.read_camb_cls('../data_local/cosmo2017_10K_acc3_scalCls.dat',output='array')[:,:ilmax+1]
        print('generate alms')
        Talm, Ealm = cs.utils.gauss2alm(ilmax,Ucl[0,:],Ucl[1,:],Ucl[2,:])
        palm = cs.utils.gauss1alm(ilmax, Ucl[3,:])
        # remap
        print('compute deflection angle')
        grad = cs.delens.phi2grad(nside, ilmax, palm)
        print('remapping')
        Talm, Ealm, Balm = cs.delens.remap_tp(nside, ilmax, grad, np.array((Talm,Ealm,0*Ealm)))[:,:lmax+1]

    # biref (1deg rot)
    Ealm, Balm = ana.ebrotate(1.,Ealm,Balm) 
    # add noise and filtering (temp)
    Talm += cs.utils.gauss1alm(lmax,nl[0,:])
    Ealm += cs.utils.gauss1alm(lmax,nl[1,:])
    Balm += cs.utils.gauss1alm(lmax,nl[2,:])
    # simple diagonal c-inverse
    Fl = np.zeros((3,lmax+1,lmax+1))
    for l in range(rlmin,rlmax):
        Fl[:,l,0:l+1] = 1./ocl[:3,l,None]
    Talm *= Fl[0,:,:]
    Ealm *= Fl[1,:,:]
    Balm *= Fl[2,:,:]
    # compute unnormalized estiamtors
    glm, clm = {}, {}
    glm['TB'], clm['TB'] = cs.rec_ilens.qtb(lmax,rlmin,rlmax,fcl[3,:],Talm,Balm)
    glm['EB'], clm['EB'] = cs.rec_ilens.qeb(lmax,rlmin,rlmax,fcl[1,:]-fcl[2,:],Ealm,Balm)
    glm['BB'], clm['BB'] = cs.rec_ilens.qbb(lmax,rlmin,rlmax,fcl[1,:]-fcl[2,:],Balm,Balm)
    # compute cross spectra
    gl, cl = {}, {}
    #for qest in ['TE','TB','EE','EB','BB']:
    for qest in ['TB','EB','BB']:
        gl[qest] = cs.utils.alm2cl(lmax,Ag[qest][:,None]*glm[qest],palm)
        cl[qest] = cs.utils.alm2cl(lmax,Ag[qest][:,None]*clm[qest],palm)
    #return gl['TE'], gl['TB'], gl['EE'], gl['EB'], gl['BB'], cl['TE'], cl['TB'], cl['EE'], cl['EB'], cl['BB']
    return gl['TB'], gl['EB'], gl['BB'], cl['TB'], cl['EB'], cl['BB']


# define parameters
lmax  = 4000       # maximum multipole of output normalization
rlmin, rlmax = 100, lmax  # reconstruction multipole range
sig   = 5.
theta = 1.
L = np.linspace(0,lmax,lmax+1)
# for sim
simn = 100

# load data:
ucl = CMB.read_camb_cls('../data_local/cosmo2017_10K_acc3_scalCls.dat',output='array')[:,:lmax+1]
lcl = CMB.read_camb_cls('../data_local/cosmo2017_10K_acc3_lensedCls.dat',ftype='lens',output='array')[:,:lmax+1]
nl  = np.zeros((4,lmax+1))
nl[0,:] = .5*(sig*ac2rad/Tcmb)**2*np.exp(L*(L+1)*(theta*ac2rad)**2/8./np.log(2.))
nl[1,:] = 2*nl[0,:]
nl[2,:] = 2*nl[0,:]
ocl = lcl + nl

for weight in ['lcl','ucl']:

    # filter:
    fcl = lcl.copy()
    if weight=='ucl':
        fcl[0,:] = ucl[0,:]
        fcl[1,:] = ucl[1,:]
        fcl[2,:] = ucl[0,:]*0.
        fcl[3,:] = ucl[2,:]

    # compute normalization
    Ag, Ac = {}, {}
    Ag['TB'], Ac['TB'] = cs.norm_imag.qtb('lens',rlmax,rlmin,rlmax,fcl[3,:],ocl[0,:],ocl[2,:])
    Ag['EB'], Ac['EB'] = cs.norm_imag.qeb('lens',rlmax,rlmin,rlmax,fcl[1,:]-fcl[2,:],ocl[1,:],ocl[2,:])
    Ag['BB'], Ac['BB'] = cs.norm_imag.qbb('lens',rlmax,rlmin,rlmax,fcl[1,:]-fcl[2,:],ocl[2,:])

    # compute cross spectrum:
    cl = np.zeros((simn,6,lmax+1))
    for i in tqdm.tqdm(range(simn)):
        fname = '../data_local/sim/cross_spec_'+str(i).zfill(5)+'_'+weight+'.dat'
        if misctools.check_path(fname,verbose=False): continue
        cl[i,:,:] = sim(i,lmax,rlmin,rlmax,Ag,Ac,fcl,ocl,nl)
        np.savetxt(fname,cl[i,:,:].T)

