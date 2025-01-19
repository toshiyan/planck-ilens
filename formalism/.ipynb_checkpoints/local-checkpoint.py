
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


def noise(lcl,sig,theta,alpha=1.,lBmin=2,split=False):
    
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

    Lcl = lcl.copy()
    Lcl[2,:] *= alpha

    return Lcl + nl


def sim_iamp(i,M,palm,lmax,rlmin,rlmax,Ag,ocl,lcl,nl,beta,freq):
    
    fname = '/global/homes/t/toshiyan/scratch/plk_biref/test/glm_'+str(freq)+'_'+str(i).zfill(3)+'.pkl'
    
    if misctools.check_path(fname,verbose=False): 

        glm, Ealm, Balm = pickle.load(open(fname,"rb"))
    
    else:

        alms = hp.read_alm('/project/projectdirs/sobs/v4_sims/mbs/cmb/fullskyLensedUnabberatedCMB_alm_set00_'+str(i).zfill(5)+'.fits',hdu=(1,2,3))
        #Talm = cs.utils.lm_healpy2healpix( alms[0], 5100 ) [:lmax+1,:lmax+1] / CMB.Tcmb
        Ealm = cs.utils.lm_healpy2healpix( alms[1], 5100 ) [:lmax+1,:lmax+1] / CMB.Tcmb
        Balm = cs.utils.lm_healpy2healpix( alms[2], 5100 ) [:lmax+1,:lmax+1] / CMB.Tcmb
        # biref
        #print('birefringence')
        Ealm, Balm = ana.ebrotate(beta,Ealm,Balm)
        # add noise and filtering (temp)
        #print('add noise')
        #Talm += cs.utils.gauss1alm(lmax,nl[0,:])
        Ealm += cs.utils.gauss1alm(lmax,nl[1,:])
        Balm += cs.utils.gauss1alm(lmax,nl[2,:])
        #print('mask')
        #Talm = cs.utils.mulwin(Talm,M)
        Ealm, Balm = cs.utils.mulwin_spin(Ealm,Balm,M)
        # simple diagonal c-inverse
        #print('reconstruction')
        Fl = np.zeros((3,lmax+1,lmax+1))
        for l in range(rlmin,rlmax):
            Fl[:,l,0:l+1] = 1./ocl[:3,l,None]
        #Talm *= Fl[0,:,:]
        fEalm = Ealm*Fl[1,:,:]
        fBalm = Balm*Fl[2,:,:]
        # compute unnormalized estiamtors
        #glm['TE'], clm['TE'] = cs.rec_iamp.qte(lmax,rlmin,rlmax,lcl[3,:],Talm,Ealm)
        #glm['TB'], clm['TB'] = cs.rec_iamp.qtb(lmax,rlmin,rlmax,lcl[3,:],Talm,Balm)
        #glm['EE'], clm['EE'] = cs.rec_iamp.qee(lmax,rlmin,rlmax,lcl[1,:],Ealm,Ealm)
        glm = cs.rec_iamp.qeb(lmax,rlmin,rlmax,ocl[1,:]-ocl[2,:],fEalm,fBalm)
        #glm['BB'], clm['BB'] = cs.rec_iamp.qbb(lmax,rlmin,rlmax,lcl[1,:],Balm,Balm)
        #print('cross spec')
        pickle.dump((glm,Ealm,Balm),open(fname,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    
    cl = cs.utils.alm2cl(lmax,Ag['EB'][:,None]*glm,palm)
    W2 = np.mean(M**2)
    EB = cs.utils.alm2cl(lmax,Ealm,Balm)/W2
    return cl, EB


def est_beta(icls,fidcl):
    simn = len(icls[:,0])
    icl = np.array( [ icls[i,:]/fidcl for i in range(simn) ] )
    vcl = np.array( [ np.std(np.delete(icl,i,axis=0),axis=0) for i in range(simn) ] )
    return np.array( [ np.sum(icl[i,:]/vcl[i,:]**2)/np.sum(1./vcl[i,:]**2) for i in range(simn) ] ) 


def rec(Lmax,rlmin,rlmax,lcl,ocl,qdo=['TB','EB','BB']):
    Ag = {}
    if 'TE' in qdo:
        Ag['TE'] = cs.norm_imag.qte('lens',Lmax,rlmin,rlmax,lcl[3,:],ocl[0,:],ocl[1,:])[0]
    if 'TB' in qdo:
        Ag['TB'] = cs.norm_imag.qtb('lens',Lmax,rlmin,rlmax,lcl[3,:],ocl[0,:],ocl[2,:])[0]
    if 'EE' in qdo:
        Ag['EE'] = cs.norm_imag.qee('lens',Lmax,rlmin,rlmax,lcl[1,:],ocl[1,:])[0]
    if 'EB' in qdo:
        Ag['EB'] = cs.norm_imag.qeb('lens',Lmax,rlmin,rlmax,lcl[1,:],ocl[1,:],ocl[2,:])[0]
    if 'BB' in qdo:
        Ag['BB'] = cs.norm_imag.qbb('lens',Lmax,rlmin,rlmax,lcl[1,:],ocl[2,:])[0]
    return Ag


