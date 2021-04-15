# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import pickle
import tqdm
import sys
from scipy.signal import savgol_filter

# from cmblensplus/wrap
import curvedsky as cs

# from cmblensplus/utils
import misctools
import cmb as CMB

# from pylib
import planck_filename as plf

# local
import local


class CIB():

    def __init__(self,snmin=0,snmax=100,dtype='gnilc',freq='545',wind='G60',ascale=1.0,lmax=2048):

        conf = misctools.load_config('CIB')

        self.snmin = conf.getint('snmin',snmin)
        self.snmax = conf.getint('snmax',snmax)
        self.snum  = self.snmax - self.snmin + 1
        self.rlz   = np.linspace(self.snmin,self.snmax,self.snum,dtype=np.int)

        # cib map type
        self.dtype  = conf.get('dtype',dtype)
        self.freq   = conf.get('freq',freq)

        # cib mask
        self.wind   = conf.get('wind',wind)
        self.ascale = ascale
        
        # others
        self.lmax   = conf.getint('lmax',lmax)

        # tag
        self.itag   = self.dtype+'_'+self.freq+'_'+self.wind+'_a'+str(self.ascale)+'deg'

        # beam
        if self.dtype == 'gnilc':
            self.ibl = CMB.beam(5.,self.lmax)

        # check
        if self.dtype == 'kcib':
            if self.freq not in ['T+E','MV']:
                sys.exit('need freq=T+E or MV for phicib')

    def filename(self):

        d = local.data_directory()

        #//// public data ////#
        if self.dtype == 'gnilc':
            self.fcib   = plf.load_cib_filename(self.freq)
            self.fmask  = d['cib'] + 'mask/'+self.wind+'.fits'
            self.famask = d['cib'] + 'mask/'+self.wind+'_a'+str(self.ascale)+'deg.fits'
        elif self.dtype == 'lenz':
            self.fcib   = d['cib'] + self.freq + '/' + self.wind + '/cib_fullmission.hpx.fits'
            self.fmask  = d['cib'] + self.freq + '/' + self.wind + '/mask_bool.hpx.fits'
            self.afmask = d['cib'] + self.freq + '/' + self.wind + '/mask_apod.hpx.fits'
        elif self.dtype == 'kcib':
            if self.freq=='T+E': 
                self.fcib = plf.subd['pr3']['lens'] + 'wcib/dat_klm_545_TTTEETEE.fits'
            if self.freq=='MV': 
                self.fcib = plf.subd['pr3']['lens'] + 'wcib/dat_klm_545_MV.fits'
            self.fmask_org = plf.subd['pr3']['lens'] + 'wcib/mask.fits'
            self.fmask  = plf.subd['pr3']['lens'] + 'wcib/mask_'+self.wind+'.fits'
            self.afmask = plf.subd['pr3']['lens'] + 'wcib/mask_'+self.wind+'_a'+str(self.ascale)+'deg.fits'
            
        if self.ascale == 0.0:  self.famask = self.fmask

        # cib alm
        self.falm  = [d['cib'] + 'alm/'+self.itag+'_'+x+'.pkl' for x in local.ids]

        # real cib cl
        self.fcII  = d['cib'] + 'cl_' + self.itag+'_real.dat'


def init_cib(**kwargs):

    iobj = CIB(**kwargs)
    iobj.filename()
    
    return iobj


def map2alm(iobj,W,TK=2.726,**kwargs_ov):

    nside = hp.pixelfunc.get_nside(W)
    l  = np.linspace(0,iobj.lmax,iobj.lmax+1)
    w2 = np.mean(W**2)

    for i in tqdm.tqdm(iobj.rlz,ncols=100,desc='map2alm ('+iobj.dtype+','+str(iobj.freq)+')'):

        if misctools.check_path(iobj.falm[i],**kwargs_ov): continue

        if i==0: # real data

            if iobj.freq == '545': 
                unit = 0.0059757149*9113.0590
            elif iobj.freq == '857': 
                unit = 9.6589431e-05*22533.716
            else: 
                unit = 1.
            
            Imap = W * hp.fitsfunc.read_map(iobj.fcib,verbose=False)/unit/TK # MJy/sr -> Kcmb
            Ialm = cs.utils.hp_map2alm(nside,iobj.lmax,iobj.lmax,Imap) * iobj.ibl[:,None]
            clII = cs.utils.alm2cl(iobj.lmax,Ialm)/w2
            
            # cross with official kappa
            aobj = local.init_analysis()
            rklm = cs.utils.lm_healpy2healpix( hp.read_alm(aobj.flens['MV'][0]), 4096 )[:iobj.lmax+1,:iobj.lmax+1]
            
            # wfactor
            wlen = hp.read_map(aobj.flens['mask'])
            wcib = hp.read_map(iobj.famask)
            wfac = np.mean(wcib*wlen**2)
            
            # cross spectrum
            clIk = cs.utils.alm2cl(iobj.lmax,rklm,Ialm)/wfac
            
        else: # generate sim

            clII, clIk = np.loadtxt(iobj.fcII,unpack=True,usecols=(1,2))

            #Ialm = cs.utils.gauss1alm(iobj.lmax,clII[:iobj.lmax+1])
            aobj = local.init_analysis()
            iklm = hp.read_alm(aobj.flens['IN'][i])
            iklm = cs.utils.lm_healpy2healpix(iklm,4096)[:iobj.lmax+1,:iobj.lmax+1]
            clIIs = savgol_filter( clII, 91, 0)
            clIks = savgol_filter( clIk, 91, 0)
            __, Ialm = cs.utils.gauss2alm_const(iobj.lmax,aobj.ckk,clIIs,clIks,iklm)
            
            # apply mask
            Imap = W * cs.utils.hp_alm2map(nside,iobj.lmax,iobj.lmax,Ialm)
            Ialm = cs.utils.hp_map2alm(nside,iobj.lmax,iobj.lmax,Imap)

        if i == 0:
            np.savetxt(iobj.fcII,np.array((l,clII,clIk)).T)

        pickle.dump((Ialm),open(iobj.falm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def interface(run=['alm'],kwargs_cib={},kwargs_ov={}):
    
    iobj = init_cib(**kwargs_cib)
    
    W = hp.fitsfunc.read_map(iobj.famask,verbose=False)
    
    if 'alm' in run:
        map2alm(iobj,W,**kwargs_ov)



