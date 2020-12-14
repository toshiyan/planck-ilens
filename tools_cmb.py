
import numpy as np
import healpy as hp
import scipy.signal as sp
import pickle
import os
from astropy.io import fits
import tqdm

#from cmblensplus/wrap/
import basic
import curvedsky

#from cmblensplus/utils/
import misctools
import analysis

#local module
import prjlib


def save_beam(fsmap,fbeam,lmax):

    if not os.path.exists(fbeam):
        bl = fits.getdata(fsmap,extname='BEAMTF')
        np.savetxt(fbeam,bl)


def reduc_map(dtype,fmap,TK=2.726,field=0,scale=1.):

    if 'hmhd' in dtype:
        map1 = hp.fitsfunc.read_map(fmap[0],field=field,verbose=False)
        map2 = hp.fitsfunc.read_map(fmap[1],field=field,verbose=False)
        rmap = (map1-map2)*.5
    else:
        rmap = hp.fitsfunc.read_map(fmap,field=field,verbose=False)

    return rmap * scale / TK


def map2alm(lmax,fmap,falm,mask,ibl,dtype,scale=1.,beta=0.,**kwargs):

    if misctools.check_path(falm,**kwargs): return
    
    Tmap = mask * reduc_map(dtype,fmap,scale=scale,field=0)
    Qmap = mask * reduc_map(dtype,fmap,scale=scale,field=1)
    Umap = mask * reduc_map(dtype,fmap,scale=scale,field=2)

    nside = hp.pixelfunc.get_nside(Tmap)
    # convert to alm
    Talm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,Tmap)
    Ealm, Balm = curvedsky.utils.hp_map2alm_spin(nside,lmax,lmax,2,Qmap,Umap)
    # beam deconvolution
    Talm *= ibl[:,None]#/pfunc[:,None]
    Ealm *= ibl[:,None]#/pfunc[:,None]
    Balm *= ibl[:,None]#/pfunc[:,None]
    # isotropic rotation
    Ealm, Balm = analysis.ebrotate(beta,Ealm,Balm)
    
    # save to file
    pickle.dump((Talm,Ealm,Balm),open(falm,"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def map2alm_all(rlz,lmax,fmap,falm,wind,fbeam,dtype,sscale=1.,nscale=1.,stype=['s','n'],beta=0.,**kwargs):

    # beam function
    ibl = 1./np.loadtxt(fbeam)[:lmax+1]

    for i in tqdm.tqdm(rlz,ncols=100,desc='map2alm:'):
        
        if i == 0:  # real data
            map2alm(lmax,fmap['s'][i],falm['s']['T'][i],wind,ibl,dtype,**kwargs)
        else:  # simulation
            if 's' in stype:  map2alm(lmax,fmap['s'][i],falm['s']['T'][i],wind,ibl,dtype,scale=sscale,beta=beta,**kwargs)
            if 'n' in stype:  map2alm(lmax,fmap['n'][i],falm['n']['T'][i],wind,ibl,dtype,scale=nscale,**kwargs)


def alm_comb(rlz,falm,stype=['n','p'],overwrite=False,verbose=True):

    for i in tqdm.tqdm(rlz,ncols=100,desc='alm combine:'):

        if misctools.check_path(falm['c']['T'][i],overwrite=overwrite,verbose=verbose):  continue

        Talm, Ealm, Balm = pickle.load(open(falm['s']['T'][i],"rb"))
        Tnlm, Enlm, Bnlm = 0.*Talm, 0.*Ealm, 0.*Balm
        Tplm, Eplm, Bplm = 0.*Talm, 0.*Ealm, 0.*Balm
        if i > 0:
            if 'n' in stype:  Tnlm, Enlm, Bnlm = pickle.load(open(falm['n']['T'][i],"rb"))
            if 'p' in stype:  Tplm, Eplm, Bplm = pickle.load(open(falm['p']['T'][i],"rb"))

        pickle.dump( ( Talm+Tnlm+Tplm, Ealm+Enlm+Eplm, Balm+Bnlm+Bplm ), open(falm['c']['T'][i],"wb"), protocol=pickle.HIGHEST_PROTOCOL )


def wiener_cinv_core(i,dtype,M,cl,bl,Nij,fmap,falm,sscale,nscale,beta=0.,verbose=True,**kwargs):

    lmax  = len(cl[0,:]) - 1
    
    nside = hp.pixelfunc.get_nside(M)
    npix  = 12*nside**2

    TQU   = np.zeros((3,1,npix))

    if i==0: 
        for field in [0,1,2]:
            TQU[field,0,:] = M * reduc_map(dtype,fmap['s'][i],field=field)
    else:
        # signal
        Ts = M * reduc_map(dtype,fmap['s'][i],scale=sscale,field=0)
        Qs = M * reduc_map(dtype,fmap['s'][i],scale=sscale,field=1)
        Us = M * reduc_map(dtype,fmap['s'][i],scale=sscale,field=2)
        if beta != 0.:
            Qs, Us = analysis.rotate(beta,Qs,Us)
        
        # noise, ptsr
        Qn = M * reduc_map(dtype,fmap['n'][i],scale=nscale,field=1)
        Un = M * reduc_map(dtype,fmap['n'][i],scale=nscale,field=1)
        Qp = M * reduc_map(dtype,fmap['p'][i].replace('a0.0deg','a1.0deg'),TK=1.,field=2) # approximately use 1.0deg apodization case
        Up = M * reduc_map(dtype,fmap['p'][i].replace('a0.0deg','a1.0deg'),TK=1.,field=2) # approximately use 1.0deg apodization case

        TQU[0,0,:] = Ts + Tn + Tp
        TQU[1,0,:] = Qs + Qn + Qp
        TQU[2,0,:] = Us + Un + Up

    # cinv
    alm = curvedsky.cninv.cnfilter_freq(3,1,nside,lmax,cl,bl,Nij,T,filter='W',ro=10,verbose=verbose,**kwargs)

    pickle.dump((alm),open(falm[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def wiener_cinv(rlz,dtype,M,cl,fbeam,fmap,falm,sscale,nscale,beta=0,kwargs_ov={},kwargs_cinv={}):

    lmin = 1
    lmax = len(cl[0,:]) - 1

    bl  = np.reshape(np.loadtxt(fbeam)[:lmax+1],(1,lmax+1))
    Nij = M * (30.*(np.pi/10800.)/2.726e6)**(-2)
    Nij = np.reshape((Nij,Nij/2.,Nij/2.),(3,1,len(M)))

    for i in tqdm.tqdm(rlz,ncols=100,desc='wiener cinv:'):
        
        if beta != 0. and i==0: continue  # avoid real biref case

        if misctools.check_path(falm[i],**kwargs_ov): continue

        wiener_cinv_core(i,dtype,M,cl,bl,Nij,fmap,falm,sscale,nscale,beta=beta,verbose=kwargs_ov['verbose'],**kwargs_cinv)



def alm2aps(rlz,lmax,fcmb,w2,stype=['s','n','p','c'],cli_out=True,**kwargs_ov):  # compute aps
    # output is ell, TT(s), TT(n), TT(p), TT(s+n+p)

    if misctools.check_path(fcmb.scl,**kwargs_ov):  return

    eL = np.linspace(0,lmax,lmax+1)
    cl = np.zeros((len(rlz),4,lmax+1))

    for ii, i in enumerate(tqdm.tqdm(rlz,ncols=100,desc='cmb alm2aps:')):

        if 's' in stype:  salm = pickle.load(open(fcmb.alms['s']['T'][i],"rb"))
        if i>0:
            if 'n' in stype:  nalm = pickle.load(open(fcmb.alms['n']['T'][i],"rb"))
            if 'p' in stype:  palm = pickle.load(open(fcmb.alms['p']['T'][i],"rb"))
            if 'c' in stype:  oalm = pickle.load(open(fcmb.alms['c']['T'][i],"rb"))

        #compute cls
        if 's' in stype:  cl[ii,0,:] = curvedsky.utils.alm2cl(lmax,salm) / w2
        if i>0:
            if 'n' in stype:  cl[ii,1,:] = curvedsky.utils.alm2cl(lmax,nalm) / w2
            if 'p' in stype:  cl[ii,2,:] = curvedsky.utils.alm2cl(lmax,palm) / w2
            if 'c' in stype:  cl[ii,3,:] = curvedsky.utils.alm2cl(lmax,oalm) / w2
                
        if cli_out:  np.savetxt(fcmb.cl[i],np.concatenate((eL[None,:],cl[ii,:,:])).T)

    # save to files
    if rlz[-1]>2:
        if kwargs_ov['verbose']:  print('cmb alm2aps: save sim')
        i0 = max(0,1-rlz[0])
        np.savetxt(fcmb.scl,np.concatenate((eL[None,:],np.mean(cl[i0:,:,:],axis=0),np.std(cl[i0:,:,:],axis=0))).T)

    if rlz[0] == 0:
        if kwargs_ov['verbose']:  print('cmb alm2aps: save real')
        np.savetxt(fcmb.ocl,np.array((eL,cl[0,0,:])).T)



def gen_ptsr(rlz,fcmb,fbeam,fseed,fcl,fmap,w,olmax=2048,ilmin=1000,ilmax=3000,overwrite=False,verbose=True): # generating ptsr contributions

    # difference spectrum with smoothing
    scl = (np.loadtxt(fcmb.scl)).T[1][:ilmax+1]
    ncl = (np.loadtxt(fcmb.scl)).T[2][:ilmax+1]
    rcl = (np.loadtxt(fcmb.ocl)).T[1][:ilmax+1]

    # interpolate
    dCL = rcl - scl - ncl
    dcl = sp.savgol_filter(dCL, 101, 1)
    dcl[dcl<=0] = 1e-30
    dcl[:ilmin]  = 1e-30
    np.savetxt(fcl,np.array((np.linspace(0,ilmax,ilmax+1),dcl,dCL)).T)
    dcl = np.sqrt(dcl)

    # generating seed, only for the first run
    for i in rlz:
        if not os.path.exists(fseed[i]):
            alm = curvedsky.utils.gauss1alm(ilmax,np.ones(ilmax+1))
            pickle.dump((alm),open(fseed[i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)

    # load beam function
    bl = np.loadtxt(fbeam)[:ilmax+1]
    nside = hp.pixelfunc.get_nside(w)
    #pfunc = hp.sphtfunc.pixwin(nside)[:lmax+1]

    # multiply cl, transform to map and save it
    for i in tqdm.tqdm(rlz,ncols=100,desc='gen ptsr:'):

        if misctools.check_path(fcmb.alms['p']['T'][i],overwrite=overwrite,verbose=verbose): continue
        
        if i==0: continue
        
        palm = pickle.load(open(fseed[i],"rb"))[:ilmax+1,:ilmax+1]
        palm *= dcl[:,None]*bl[:,None] #multiply beam-convolved cl
        pmap = curvedsky.utils.hp_alm2map(nside,ilmax,ilmax,palm)
        hp.fitsfunc.write_map(fmap['p'][i],pmap,overwrite=True)
        
        palm = curvedsky.utils.hp_map2alm(nside,olmax,olmax,w*pmap) #multiply window
        palm /= bl[:olmax+1,None]  #beam deconvolution
        pickle.dump((palm),open(fcmb.alms['p']['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)



def interface(run=[],kwargs_cmb={},kwargs_ov={},kwargs_cinv={}):

    # define parameters, filenames and functions
    p = prjlib.init_analysis(**kwargs_cmb)

    # read survey window
    w, M, wn = prjlib.set_mask(p.famask)
    if p.fltr == 'cinv':  wn[:] = wn[0]

    # read beam function
    save_beam(p.fimap['s'][0],p.fbeam,p.lmax)

    # generate ptsr
    if 'ptsr' in run and p.biref==0:
        if p.dtype=='dr2_nilc':  
            ilmin, ilmax = 400, 3000
        else: 
            ilmin, ilmax = 1000, p.lmax
        # compute signal and noise spectra but need to change file names
        q = prjlib.init_analysis(**kwargs_cmb)
        q.fcmb.scl = q.fcmb.scl.replace('.dat','_tmp.dat')
        q.fcmb.ocl = q.fcmb.ocl.replace('.dat','_tmp.dat')
        if not misctools.check_path(q.fcmb.scl,**kwargs_ov):
            # compute signal and noise alms
            map2alm_all(p.rlz,ilmax,p.fimap,p.fcmb.alms,w,p.fbeam,p.dtype,p.sscale,p.nscale,**kwargs_ov)
            alm2aps(p.rlz,ilmax,q.fcmb,wn[2],stype=['s','n'],cli_out=False,**kwargs_ov)
        # generate ptsr alm from obs - (sig+noi) spectrum
        gen_ptsr(p.rlz,q.fcmb,p.fbeam,p.fpseed,p.fptsrcl,p.fimap,w,olmax=p.lmax,ilmin=ilmin,ilmax=ilmax,**kwargs_ov) # generate map and alm from above computed aps


    # use normal transform to alm
    if p.fltr == 'none':
        
        stypes = ['s','n','p','c']
    
        if 'alm' in run:  # combine signal, noise and ptsr
            map2alm_all(p.rlz,p.lmax,p.fimap,p.fcmb.alms,w,p.fbeam,p.dtype,p.sscale,p.nscale,beta=p.biref,**kwargs_ov)
            # combine signal, noise and ptsr alms
            alm_comb(p.rlz,p.fcmb.alms,stype=stypes,**kwargs_ov)

        if 'aps' in run:  # compute cl
            alm2aps(p.rlz,p.lmax,p.fcmb,wn[2],stype=stypes,**kwargs_ov)

    # map -> alm with cinv filtering
    if p.fltr == 'cinv':  

        falm = p.fcmb.alms['c']['T']  #output file of cinv alms
    
        if 'alm' in run:  # cinv filtering here
            wiener_cinv(p.rlz,p.dtype,M,p.lcl[0:1,:p.lmax+1],p.fbeam,p.fimap,falm,p.sscale,p.nscale,beta=p.biref,kwargs_ov=kwargs_ov,kwargs_cinv=kwargs_cinv)

        if 'aps' in run:  # aps of filtered spectrum
            p.fcmb.alms['s']['T'] = falm # since cinv only save 'c', not 's'
            alm2aps(p.rlz,p.lmax,p.fcmb,wn[0],stype=['s','c'],**kwargs_ov)
    

