# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import pickle
import tqdm
from scipy.signal import savgol_filter

# from cmblensplus/wrap
import curvedsky.utils as csu
import basic

# from cmblensplus/utils
import quad_func
import misctools
import binning as bn

# local
import local
import tools_cib


class cross():
    
    def __init__(self,qobj,iobj,ids):
        
        d = local.data_directory()
        
        self.fcli = {}
        self.fmcl = {}
        
        for q in qobj.qlist:
            self.fcli[q] = [d['root']+qobj.qtype+'/aps/rlz/cl_'+q+'_'+qobj.cmbtag+'_'+iobj.itag+qobj.bhe_tag+qobj.ltag+'_'+x+'.dat' for x in ids]
            self.fmcl[q] = d['root']+qobj.qtype+'/aps/cl_sim_1d_'+q+'_'+qobj.cmbtag+'_'+iobj.itag+qobj.bhe_tag+qobj.ltag+'.dat'


def init_quad(snmax,qtypes,**ikwargs):
    
    d = local.data_directory()
    
    kwargs = {\
        'n0min':1, \
        'n0max':int(snmax/2), \
        'mfmin':1, \
        'mfmax':snmax, \
        'rdmin':1, \
        'rdmax':snmax \
    }

    qlist = {}
    qlist['lens'] = ['TT','TE','EE','TB','EB']
    qlist['ilens'] = ['TB','EB','BB']
    
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)
    return {qt: quad_func.quad(root=d['root'],qtype=qt,qlist=qlist[qt],**kwargs,**ikwargs) for qt in qtypes }



def aps(rlz,qobj,xobj,wfac,flens,fcib,**kwargs_ov):

    Lmax = qobj.olmax
    cl   = np.zeros((len(rlz),7,Lmax+1))
    
    for q in qobj.qlist:

        for i, r in enumerate(tqdm.tqdm(rlz,ncols=100,desc='aps ('+qobj.qtype+','+q+')')):
        
            if misctools.check_path(xobj.fcli[q][r],**kwargs_ov): continue
        
            # load qlm
            alm = quad_func.load_rec_alm(qobj,q,r,mean_sub=True)[0]
            #alm = pickle.load(open(qobj.f[q].alm[r],"rb"))[0]
            #mf  = pickle.load(open(qobj.f[q].mfalm[r],"rb"))[0]
            #alm -= mf

            # auto spectrum
            cl[i,0,:] = csu.alm2cl(Lmax,alm)/qobj.wn[4]

            if r>0:
                # load input klm
                klm = hp.read_alm(flens['IN'][r])
                klm = csu.lm_healpy2healpix(klm,4096)[:Lmax+1,:Lmax+1]
                # cross with input klm
                cl[i,1,:] = csu.alm2cl(Lmax,alm,klm)/qobj.wn[2]
        
            # load reconstructed klm
            klm = hp.read_alm(flens['MV'][r])
            klm = csu.lm_healpy2healpix(klm,4096)[:Lmax+1,:Lmax+1]
            # cross with lens
            cl[i,2,:] = csu.alm2cl(Lmax,alm,klm)/wfac['ik']
            # lens auto
            cl[i,3,:] = csu.alm2cl(Lmax,klm)/wfac['kk']

            # load cib alm
            Ilm = pickle.load(open(fcib[r],"rb"))
            # cross with cib
            cl[i,4,:] = csu.alm2cl(Lmax,alm,Ilm)/wfac['iI']
            # cib
            cl[i,5,:] = csu.alm2cl(Lmax,Ilm)/wfac['II']
        
            # lens x cib
            cl[i,6,:] = csu.alm2cl(Lmax,Ilm,klm)/wfac['kI']

            np.savetxt(xobj.fcli[q][r],np.concatenate((qobj.l[None,:],cl[i,:,:])).T)

        # save to files
        if rlz[-1] >= 2 and not misctools.check_path(xobj.fmcl[q],**kwargs_ov):
            i0 = max(0,1-rlz[0])
            np.savetxt(xobj.fmcl[q],np.concatenate((qobj.l[None,:],np.average(cl[i0:,:,:],axis=0),np.std(cl[i0:,:,:],axis=0))).T)


def quad_comb(scb,vcb,ocb,quad=['TB','EB','BB']):
    norm = 0.
    ocbc = 0.
    scbc = 0.
    for q in quad:
        norm += 1./vcb[q]**2
        ocbc += ocb[q]/vcb[q]**2
        scbc += scb[q]/vcb[q]**2
    norm = 1./norm
    ocbc *= norm
    scbc *= norm
    mcbc = np.mean(scbc,axis=0)
    vcbc = np.std(scbc,axis=0)
    return mcbc, vcbc, scbc, ocbc


def binned_spec(mb,aobj,qobj,xobj,cn,quad=['TB','EB','BB'],doreal=True,tcb=None):
    mcb, vcb, scb, ocb = {}, {}, {}, {}
    for q in quad:
        if cn==2:  kk = aobj.ckk
        #if cn==3:  kk = savgol_filter( (np.loadtxt(xobj.fcli[q][0])).T[4], 21, 3) 
        if cn==3:  kk = np.loadtxt(xobj.fmcl[q]).T[4]
        #if cn==3:  kk = np.loadtxt(xobj.fmcl[q]).T[4]
        if cn==5:  kk = savgol_filter( (np.loadtxt(xobj.fcli[q][0])).T[6], 21, 2)
        al = (np.loadtxt(qobj['ilens'].f[q].al)).T[1]
        vl = np.sqrt(al*kk)/np.sqrt(qobj['ilens'].l+1e-30)
        if doreal:
            mcb[q], vcb[q], scb[q], ocb[q] = bn.binned_spec(mb,xobj.fcli[q],cn=cn,opt=True,vl=vl,doreal=doreal)
        else: 
            mcb[q], vcb[q], scb[q] = bn.binned_spec(mb,xobj.fcli[q],cn=cn,opt=True,vl=vl,doreal=False)
    if doreal:
        if tcb is not None:
            for q in quad: ocb[q] -= tcb[q] 
        mcb['MV'], vcb['MV'], scb['MV'], ocb['MV'] = quad_comb(scb,vcb,ocb,quad)
    return mcb, vcb, scb, ocb



def n0_template(aobj,iobj,mb,quad=['TB','EB','BB']):  # template for non-zero biref
    bobj = local.init_analysis(freq=aobj.freq,dtype=aobj.dtype,wind=aobj.wind,fltr=aobj.fltr,biref=1.)
    robj = init_quad(bobj.snmax,ids=bobj.ids,rlz=bobj.rlz,stag=bobj.stag,qtypes=['ilens'],rlmin=100,rlmax=2048)
    yobj = cross(robj['ilens'],iobj,bobj.ids)
    Mkk  = bn.binning(aobj.ckk,mb)*np.pi/180.
    tcb  = binned_spec(mb,bobj,robj,yobj,3,doreal=False)[0]
    for q in quad: tcb[q] = (tcb[q]-Mkk)*.4
    return tcb



def interface(run=['norm','qrec','n0','mean'],qtypes=['lens','ilens'],kwargs_ov={},kwargs_cmb={},kwargs_cib={},rlmin=100,rlmax=2048,ep=1e-30):
    
    aobj = local.init_analysis(**kwargs_cmb)
    iobj = tools_cib.init_cib(**kwargs_cib)
    
    # wfactor
    Wcmb, __, wn = local.set_mask(aobj.famask)
    Wcib = hp.fitsfunc.read_map(iobj.famask,verbose=False)
    Wlen = hp.fitsfunc.read_map(aobj.flens['mask'],verbose=False)
    wfac = {}
    wfac['II'] = np.mean(Wcib**2)
    wfac['iI'] = np.mean(Wcmb**2*Wcib)
    wfac['kI'] = np.mean(Wlen**2*Wcib)
    wfac['kk'] = np.mean(Wlen**4)
    wfac['ik'] = np.mean(Wcmb**2*Wlen**2)
    
    #//// load obscls ////#
    if aobj.fltr == 'none':
        ocl = np.loadtxt(aobj.fcmb.scl['c'],unpack=True,usecols=(1,2,3,4))
        ifl = ocl.copy()

    if aobj.fltr == 'cinv':
        # model spectra
        ncl = np.zeros((4,aobj.lmax+1))
        ncl[0] = (aobj.ibl*aobj.sigma)**2
        ncl[1] = 2*ncl[0]
        ncl[2] = 2*ncl[0]
        cnl = aobj.lcl[0:4,:] + ncl
        # wiener-filtered spectrum
        wcl = (np.loadtxt(aobj.fcmb.scl['c'],usecols=(1,2,3,4))).T
        # obs and inv cl
        ocl, ifl = quad_func.cinv_empirical_fltr(aobj.lcl,wcl,cnl)
        # wfactor
        wn[:] = wn[0]

    #//// define objects ////#
    qobj = init_quad(aobj.snmax,qtypes,ids=aobj.ids,stag=aobj.stag,rlz=aobj.rlz,wn=wn,lcl=aobj.lcl,ocl=ocl,ifl=ifl,falm=aobj.fcmb.alms['c'],rlmin=rlmin,rlmax=rlmax,**kwargs_ov)

    # reconstruction, cross spectrum
    for qt in qobj.keys():
        print(qt)
        qobj[qt].qrec_flow(run=run)
        if 'aps' in run:
            xobj = cross(qobj[qt],iobj,aobj.ids)
            aps(aobj.rlz,qobj[qt],xobj,wfac,aobj.flens,iobj.falm,**kwargs_ov)

