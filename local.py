
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
import constants

# from pylib
import planck_filename as plf


#//// index ////#
ids = [str(i).zfill(5) for i in range(-1,1000)]
ids[0] = 'real'  # change 1st index


def data_directory():
    
    direct = {}

    root = '/global/cscratch1/sd/toshiyan/plk_biref/'
    direct['root'] = root
    direct['inp']  = root + 'input/'
    direct['win']  = root + 'mask/'
    direct['cmb']  = root + 'cmb/'
    direct['bem']  = root + 'beam/'
    direct['cib']  = root + 'cib/'
    direct['xcor'] = root + 'xcor/'
    direct['ptsr'] = root + 'ptsr/'

    return direct


# * Define CMB file names
class cmb:

    def __init__(self,aobj,mtype=['T','E','B']):

        # set directory
        d = data_directory()
        d_alm = d['cmb'] + 'alm/'
        d_aps = d['cmb'] + 'aps/'
        d_map = d['cmb'] + 'map/'

        # cmb signal/noise alms
        self.alms, self.scl, self.cl, self.xcl, self.ocl = {}, {}, {}, {}, {}
        for s in ['s','n','p','c']:
            self.alms[s] = {}
            if s in ['n','p']: tag = aobj.ntag
            if s in ['s','c']: tag = aobj.stag
            for m in mtype:
                self.alms[s][m] = [d_alm+'/'+s+'_'+m+'_'+tag+'_'+x+'.pkl' for x in aobj.ids]
            # cmb aps
            self.scl[s] = d_aps+'aps_sim_1d_'+s+'_'+aobj.stag+'.dat'
            self.cl[s]  = [d_aps+'/rlz/cl_'+s+'_'+aobj.stag+'_'+x+'.dat' for x in aobj.ids]
            self.xcl[s] = d_aps+'aps_sim_1d_'+s+'_'+aobj.stag+'_cross.dat'
            self.ocl[s] = d_aps+'aps_obs_1d_'+s+'_'+aobj.stag+'.dat'


#* Define parameters
class analysis:

    def __init__(self,snmin=0,snmax=100,dtype='full',freq='143',fltr='none',lmin=1,lmax=2048,olmin=1,olmax=2048,wind='base',ascale=1.,biref=0.):

        #//// load config file ////#
        conf = misctools.load_config('CMB')

        # rlz
        self.snmin = conf.getint('snmin',snmin)
        self.snmax = conf.getint('snmax',snmax)
        self.snum  = self.snmax - self.snmin + 1
        self.rlz   = np.linspace(self.snmin,self.snmax,self.snum,dtype=np.int)
        self.ids   = ids[:snmax+1]

        # multipole of converted CMB alms
        self.lmin   = conf.getint('lmin',lmin)
        self.lmax   = conf.getint('lmax',lmax)

        # multipole of output CMB spectrum
        self.olmin  = conf.getint('olmin',olmin)
        self.olmax  = conf.getint('olmax',olmax)

        # full or half mission
        self.dtype  = conf.get('dtype',dtype)
        if self.dtype not in ['full','hm1','hm2']:
            sys.exit('data type is not supported')
        
        # cmb map
        self.freq   = conf.get('freq',freq)
        self.biref  = conf.getfloat('biref',biref)
        if not isinstance(self.biref,float):  sys.error('biref value should be float')

        self.fltr   = conf.get('fltr',fltr)
        if self.fltr == 'cinv':  ascale = 0.

        # window
        self.wind  = conf.get('wind',wind)
        if self.wind == 'Fullsky':  
            ascale = 0.
            self.fltr == 'none'
        self.ascale = conf.getfloat('ascale',ascale)

        # cmb scaling (correction to different cosmology)
        self.sscale = 1.

        # noise scaling (correction to underestimate of noise in sim)
        self.nscale = 1.
        
        # beam
        self.ibl = get_transfer(self.freq,self.lmax)

        # white nosie level
        self.sigma = get_white_noise_level(self.freq)
        

    def filename(self):

        #//// root directories ////#
        d = data_directory()

        #//// basic tags ////#
        # for alm
        apotag = 'a'+str(self.ascale)+'deg'
        self.stag = '_'.join( [ self.dtype , self.freq, self.wind , apotag , self.fltr ] )
        self.ntag = self.stag
        if self.biref!=0.:  self.stag = self.stag + '_beta'+str(self.biref)+'deg'

        #//// Input public maps ////#
        # PLANCK DR3 and FFP10 sims
        self.fimap = plf.load_iqu_filename(PR=3,dtype=self.dtype,freq=self.freq)
        self.fgmap = plf.load_fg_filename(self.freq)

        # reconstructed klm realizations
        self.flens = plf.load_lens_filename()

        # aps of Planck best fit cosmology
        self.flcl = plf.subd['pr3']['cosmo'] + 'COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt'

        # input klm realizations
        if self.biref!=0.:
            self.fbalm = [ d['inp'] + 'sky_blms/sim_'+x[1:]+'_blm.fits' for x in ids ]
        else:
            self.fbalm = None

        #//// Galactic mask ////#
        self.fGmask  = plf.load_mask_filename_gal()

        #//// base best-fit cls ////#
        # for forecast
        self.simcl = d['inp'] + 'forecast_cosmo2017_10K_acc3_lensedCls.dat'
        self.simul = d['inp'] + 'forecast_cosmo2017_10K_acc3_scalCls.dat'

        #//// Derived data filenames ////#
        # cmb map, alm and aps
        self.fcmb = cmb(self)

        # beam
        #self.fbeam = d['bem'] + self.freq + '.dat'

        # extra gaussian random fields
        # ptsr seed
        #self.fpseed  = [d['ptsr']+x+'.fits' for x in ids]
        #ptag = self.wind + '_a'+str(self.ascale)+'deg'
        #self.fptsrcl = d['ptsr']+'ptsr_'+ptag+'.dat'
        #self.fimap['p'] = [ d['ptsr'] + 'ptsr_'+ptag+'_'+x+'.fits' for x in ids]
        
        # set additional mask filename
        d = data_directory()
        dwin = d['win']
        
        # set mask filename
        if   self.wind == 'Fullsky':
            self.fmask = ''
        #elif self.wind == 'Lmask':
        #    self.fmask = self.flens['mask']
        elif self.wind in ['Lens','LG60','LG40','LG20']:
            self.fmask = dwin + 'PR3_'+'_'.join( [ self.dtype , self.freq, self.wind ] )+'.fits'
        elif self.wind == 'base':
            self.fmask = dwin + 'PR3_'+self.dtype+'_'+self.freq+'.fits'
        else:
            sys.exit('unknown wind type, please check')
        
        # apodized map
        self.famask = self.fmask.replace('.fits','_a'+str(self.ascale)+'deg.fits')
        
        # set original file
        if self.ascale==0.:  self.famask = self.fmask


    def array(self):

        #multipole
        self.l    = np.linspace(0,self.lmax,self.lmax+1)
        self.kL   = self.l*(self.l+1)*.5
        self.lfac = self.l*(self.l+1)/(2.*np.pi)

        #theoretical cl
        self.lcl = np.zeros((5,self.lmax+1)) # TT, EE, BB, TE, PP
        self.lcl[:,2:] = np.loadtxt(self.flcl,unpack=True,usecols=(1,3,4,2,5))[:,:self.lmax-1] 
        self.lcl *= 2.*np.pi / (self.l**2+self.l+1e-30)
        self.lcl[0:4,:] /= constants.Tcmb**2 
        self.lcl[4,:] /= (self.l**2+self.l+1e-30)
        self.cpp = self.lcl[4,:]# * (self.l+1.) / (self.l**3+1e-30) / (2*np.pi)
        self.ckk = self.cpp * (self.l**2+self.l)**2/4.
        self.lcl = np.delete(self.lcl,4,axis=0)


#----------------
# initial setup
#----------------

def init_analysis(**kwargs):  # setup parameters, filenames, and arrays
    p = analysis(**kwargs)
    analysis.filename(p)
    analysis.array(p)
    return p

#----------------
# mask related
#----------------

def set_mask(fmask): # load precomputed apodized mask

    # read window
    w = hp.fitsfunc.read_map(fmask,verbose=False)
    if hp.pixelfunc.get_nside(w) != 2048:
        sys.exit('nside of window is not 2048')

    # binary mask
    M = w*1.
    M[M!=0.] = 1.

    # normalization
    wn = np.zeros(5)
    wn[0] = np.mean(M)
    for n in range(1,5):
        wn[n] = np.average(w**n)

    return w, M, wn


def bad_pixel_mask(obsmap,val=-1e30): # define bad pixel mask
    badpix = obsmap.copy()
    badpix[badpix>val] = 1.
    badpix[badpix!=1] = 0.
    return badpix


def mask_co_line(freq): # define co line mask
    
    d = data_directory()
    
    if freq in ['100','217','353']:
        # ratio in logscale
        McoI = hp.read_map(d['win']+'HFI_BiasMap_'+str(freq)+'-CO-noiseRatio_2048_R3.00_full.fits',field=0)
        McoQ = hp.read_map(d['win']+'HFI_BiasMap_'+str(freq)+'-CO-noiseRatio_2048_R3.00_full.fits',field=1)
        McoU = hp.read_map(d['win']+'HFI_BiasMap_'+str(freq)+'-CO-noiseRatio_2048_R3.00_full.fits',field=2)
        McoI[McoI<-2] = 0.
        McoI[McoI!=0] = 1.
        McoQ[McoQ<-2] = 0.
        McoQ[McoQ!=0] = 1.
        McoU[McoU<-2] = 0.
        McoU[McoU!=0] = 1.
        Mco = (1.-McoQ)*(1.-McoU)*(1.-McoI)
    else:
        Mco = 1.

    return Mco


def mask_ptsr(freq,freqn={'100':0,'143':1,'217':2,'353':3,'545':4,'857':5}): # define ptsr mask
    d = data_directory()
    if freq in ['100','143','217','353','545','857']:
        MptsrI = hp.fitsfunc.read_map(d['win']+'HFI_Mask_PointSrc_2048_R2.00.fits',hdu=1,field=freqn[freq]) #100-353
        if freq in ['545','857']:
            MptsrP = 1.
        else:
            MptsrP = hp.fitsfunc.read_map(d['win']+'HFI_Mask_PointSrc_2048_R2.00.fits',hdu=2,field=freqn[freq]) #100-353
        return MptsrI*MptsrP
    else:
        return 1.


def create_mask(aobj,Mptsr,Mco,verbose=True,Galwind=['Lens','LG60','LG40','LG20']):
    # bad pixel
    Imap  = hp.read_map(aobj.fimap['s'][0])
    nside = hp.get_nside(Imap)
    # ptsr x CO x bad pixel
    rmask = Mptsr * Mco * bad_pixel_mask(Imap)
    # Galactic mask
    if aobj.wind in Galwind:
        Lmask = hp.fitsfunc.read_map(aobj.flens['mask'],verbose=verbose) # lensing binary mask
        rmask *= Lmask
        if aobj.wind in ['LG60','LG40','LG20']:
            if aobj.wind == 'LG20': field=0
            if aobj.wind == 'LG40': field=1
            if aobj.wind == 'LG60': field=2
            Gmask = hp.fitsfunc.read_map(aobj.fGmask,field=field,verbose=verbose)
            rmask *= Gmask
        if aobj.freq == 'nilc':
            NilcT = hp.fitsfunc.read_map(plf.subd['pr3']['mask']+'COM_CMB_IQU-nilc_2048_R3.00_full.fits',field=3)
            NilcP = hp.fitsfunc.read_map(plf.subd['pr3']['mask']+'COM_CMB_IQU-nilc_2048_R3.00_full.fits',field=4)
            rmask *= NilcT*NilcP
    # apodized
    if verbose:  print(np.min(rmask*Imap))
    amask  = cs.utils.apodize(nside, rmask, aobj.ascale)
    # save
    hp.fitsfunc.write_map(aobj.fmask,rmask,overwrite=True)
    hp.fitsfunc.write_map(aobj.famask,amask,overwrite=True)
    if verbose:  print(np.min(amask*Imap))


#-----------------------------
# transfer function
#-----------------------------
    
def get_transfer(freq,lmax):
    
    if freq=='100':     return CMB.beam(9.5,lmax)
    elif freq=='143':   return CMB.beam(7.2,lmax)
    elif freq=='217':   return CMB.beam(5.0,lmax)
    elif freq=='353':   return CMB.beam(5.0,lmax)    
    elif freq=='857':   return CMB.beam(2.0,lmax)
    elif freq=='smica': return CMB.beam(5.0,lmax)
    elif freq=='nilc':  return CMB.beam(5.0,lmax)


#-----------------------------
# white noise level
#-----------------------------
    
def get_white_noise_level(freq):
    
    if freq == 'smica': sigma = 40.
    if freq == 'nilc':  sigma = 40.
    if freq == '100':   sigma = 50.
    if freq == '143':   sigma = 45.
    if freq == '217':   sigma = 65.
    return sigma*np.pi/10800./2.72e6 # muK-ac -> str


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


#-----------------------------
# Plot
#-----------------------------


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

