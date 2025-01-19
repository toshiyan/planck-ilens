
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


#-----------------------------
# foreground/experiment model
#-----------------------------

def foreground(nu,L,name,beta_d=1.5,beta_s=-3.1):
    L80 = L/80.
    if name in ['LiteBIRD','Planck','PICO','PlanckHFI']:
        dust, sync = 2e2, 25.
    if name in ['B3Keck','BA']:
        dust, sync = 1., 1.
    if name in ['AdvACT','ACTPol']: 
        dust, sync = 10., 10.
    
    dBB = dust*(4.7/CMB.Tcmb**2)*L80**(-0.58) * (CMB.Int_dust(nu,beta=beta_d)/CMB.Int_dust(353.,beta=beta_d))**2 * (2*np.pi/L/(L+1.)) 
    sBB = sync*(1.5/CMB.Tcmb**2)*L80**(-0.8) * (nu/23)**(2*beta_s) * (2*np.pi/L/(L+1.))
    fBB = dBB + sBB
    fEE = 2*dBB + 4*sBB
    return fEE, fBB


def experiments(name):
    
    if name == 'LiteBIRD':
        freqs  = np.array( [ 40. , 50., 60. , 68. , 78. , 89. , 100., 119., 140., 166., 195., 235., 280., 337., 402. ] )
        sigps  = np.array( [ 37.5, 24., 19.9, 16.2, 13.5, 11.7, 9.2 , 7.6 , 5.9 , 6.5 , 5.8 , 7.7 , 13.2, 19.5, 37.5 ] )
        thetas = np.array( [ 69. , 56., 48. , 43. , 39. , 35. , 29. , 25. , 23. , 21. , 20. , 19. , 24. , 20. , 17. ] )

    if name == 'LiteBIRDHFI':
        freqs  = np.array( [ 100., 119., 140., 166., 195., 235., 280., 337., 402. ] )
        sigps  = np.array( [ 9.2 , 7.6 , 5.9 , 6.5 , 5.8 , 7.7 , 13.2, 19.5, 37.5 ] )
        thetas = np.array( [ 29. , 25. , 23. , 21. , 20. , 19. , 24. , 20. , 17. ] )

    if name == 'Planck': #arXiv:1509.06770
        freqs  = np.array( [  30.,  44., 70. , 100., 143., 217., 353. ] )
        sigps  = np.array( [ 300., 300., 192.,  52.,  44.,  64., 276. ] ) * np.sqrt(2.)
        thetas = np.array( [ 33.2,  28., 13. ,  9.7,  7.3,   5.,  4.9 ] )

    if name == 'PlanckHFI': 
        freqs  = np.array( [ 100., 143., 217., 353. ] )
        sigps  = np.array( [  52.,  44.,  64., 276. ] ) * np.sqrt(2.)
        thetas = np.array( [  9.7,  7.3,   5.,  4.9 ] )

    if name == 'PlanckSingle': 
        freqs  = np.array( [ 353. ] )
        sigps  = np.array( [ 276. ] ) * np.sqrt(2.)
        thetas = np.array( [  4.9 ] )

    if name == 'CMBHFI': 
        freqs  = np.array( [ 100., 143., 217., 353. ] )
        sigps  = np.array( [  2., 1.,  2., 4. ] )
        thetas = np.array( [  8., 6.,   4.,  2. ] )

    if name == 'AdvACT': #arXiv:1509.06770
        freqs  = np.array( [ 90., 150., 230. ] )
        sigps  = np.array( [ 11.,  9.8, 35.4 ] ) * np.sqrt(2.)
        thetas = np.array( [ 2.2,  1.3,   .9 ] )
        
    if name == 'B3Keck': #arXiv:1808.00568 + sensitivity projection plot
        freqs  = np.array( [ 95., 150., 220., 270. ] )
        sigps  = np.array( [  2.,   3.,  10.,  60. ] )
        thetas = np.array( [ 24.,  30.,  21.,  17. ] )

    if name == 'BA': #arXiv:1808.00568 + sensitivity projection plot
        freqs  = np.array( [ 30., 40., 95., 150., 220., 270. ] )
        sigps  = np.array( [  7.,  8.,  1.,  1.5,  7.,   10. ] )
        thetas = np.array( [ 76., 57., 24.,  15.,  11.,   9. ] )

    if name == 'PICO':
        freqs  = np.array( [  21.,  25.,  30.,  36.,  43.,  52.,  62.,  75., 90., 108., 129., 155., 186., 223., 268., 321., 385., 462., 555., 666., 799. ] )
        sigps  = np.array( [ 23.9, 18.4, 12.4,  7.9,  7.9,  5.7,  5.4,  4.2, 2.8,  2.3,  2.1,  1.8,  4.0,  4.5,  3.1,  4.2,  4.5,  9.1, 45.8, 177.,1050. ] )
        thetas = np.array( [ 38.4,  32., 28.3, 23.6, 22.2, 18.4, 12.8, 10.7, 9.5,  7.9,  7.4,  6.2,  4.3,  3.6,  3.2,  2.6,  2.5,  2.1,  1.5,  1.3,  1.1 ] )


    return freqs, sigps, thetas


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


#-----------------------------
# Others
#-----------------------------

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


