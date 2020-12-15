
import numpy as np
import healpy as hp
import scipy.signal as sp
import pickle
import os
import tqdm

#from cmblensplus/wrap/
import basic
import curvedsky

#from cmblensplus/utils/
import misctools
import analysis
import cmb as CMB

#local module
import local


def get_transfer(freq,lmax):
    
    if freq=='100':  return CMB.beam(9.5,lmax)
    if freq=='143':  return CMB.beam(7.2,lmax)
    if freq=='217':  return CMB.beam(5.0,lmax)
    if freq=='353':  return CMB.beam(5.0,lmax)
    

def reduc_map(dtype,fmap,TK=2.726,field=0,scale=1.):

    if 'hmhd' in dtype:
        map1 = hp.fitsfunc.read_map(fmap[0],field=field,verbose=False)
        map2 = hp.fitsfunc.read_map(fmap[1],field=field,verbose=False)
        rmap = (map1-map2)*.5
    else:
        rmap = hp.fitsfunc.read_map(fmap,field=field,verbose=False)

    return rmap * scale / TK


def map2alm(lmax,fmap,fTlm,fElm,fBlm,wind,ibl,dtype,scale=1.,beta=0.,**kwargs):

    if misctools.check_path([fTlm,fElm,fBlm],**kwargs): return
    
    Tmap = wind * reduc_map(dtype,fmap,scale=scale,field=0)
    Qmap = wind * reduc_map(dtype,fmap,scale=scale,field=1)
    Umap = wind * reduc_map(dtype,fmap,scale=scale,field=2)

    nside = hp.pixelfunc.get_nside(Tmap)
    # convert to alm
    Talm = curvedsky.utils.hp_map2alm(nside,lmax,lmax,Tmap)
    Ealm, Balm = curvedsky.utils.hp_map2alm_spin(nside,lmax,lmax,2,Qmap,Umap)
    # beam deconvolution
    Talm *= ibl[:,None]
    Ealm *= ibl[:,None]
    Balm *= ibl[:,None]
    # isotropic rotation
    Ealm, Balm = analysis.ebrotate(beta,Ealm,Balm)
    
    # save to file
    pickle.dump((Talm),open(fTlm,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((Ealm),open(fElm,"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((Balm),open(fBlm,"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def map2alm_all(rlz,lmax,fmap,falm,wind,dtype,ibl,sscale=1.,nscale=1.,stype=['s','n'],beta=0.,**kwargs):

    for i in tqdm.tqdm(rlz,ncols=100,desc='map2alm:'):
        if i == 0:  # real data
            map2alm(lmax,fmap['s'][i],falm['s']['T'][i],wind,ibl,dtype,**kwargs)
        else:  # simulation
            if 's' in stype:  map2alm(lmax,fmap['s'][i],falm['s']['T'][i],falm['s']['E'][i],falm['s']['B'][i],wind,ibl,dtype,scale=sscale,beta=beta,**kwargs)
            if 'n' in stype:  map2alm(lmax,fmap['n'][i],falm['n']['T'][i],falm['n']['E'][i],falm['n']['B'][i],wind,ibl,dtype,scale=nscale,**kwargs)


def alm_comb(rlz,falm,stype=['n'],mtype=['T','E','B'],overwrite=False,verbose=True):

    for i in tqdm.tqdm(rlz,ncols=100,desc='alm combine:'):

        for m in mtype:
            
            if misctools.check_path(falm['c'][m][i],overwrite=overwrite,verbose=verbose):  continue

            salm = pickle.load(open(falm['s'][m][i],"rb"))
            nalm, palm = 0.*salm, 0.*salm
            if i > 0:
                if 'n' in stype:  nalm = pickle.load(open(falm['n'][m][i],"rb"))
                if 'p' in stype:  palm = pickle.load(open(falm['p'][m][i],"rb"))

            pickle.dump( ( salm+nalm+palm ), open(falm['c'][m][i],"wb"), protocol=pickle.HIGHEST_PROTOCOL )


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
    Talm, Ealm, Balm = curvedsky.cninv.cnfilter_freq(3,1,nside,lmax,cl,bl,Nij,T,filter='W',ro=10,verbose=verbose,**kwargs)

    pickle.dump((Talm),open(falm['c']['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((Ealm),open(falm['c']['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((Balm),open(falm['c']['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


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


def alm2aps(rlz,lmax,fcmb,w2,stype=['s','n','c'],**kwargs_ov):  # compute aps

    eL = np.linspace(0,lmax,lmax+1)
    
    for s in stype:
        
        cl = CMB.aps(rlz,lmax,falm[s],odd=True,w2=w2,mtype=['T','E','B'],fname=fcmb.cl[s],**kwargs_ov)
    
        if rlz[-1]>2:  # save mean
            if kwargs_ov['verbose']:  print('cmb alm2aps: save sim')
            i0 = max(0,1-rlz[0])
            np.savetxt(fcmb.scl[s],np.concatenate((eL[None,:],np.mean(cl[i0:,:,:],axis=0),np.std(cl[i0:,:,:],axis=0))).T)


def gen_ptsr(rlz,fcmb,ibl,fseed,fcl,fmap,w,olmax=2048,ilmin=1000,ilmax=3000,overwrite=False,verbose=True): # generating ptsr contributions

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
    bl = 1./ibl[:ilmax+1]
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
    aobj = prjlib.init_analysis(**kwargs_cmb)

    # read survey window
    wind, M, wn = prjlib.set_mask(aobj.famask)
    if aobj.fltr == 'cinv':  wn[:] = wn[0]

    # approximate transfer function
    ibl = get_transfer(aobj.freq,aobj.lmax)

    # generate ptsr
    if 'ptsr' in run and aobj.biref==0:
        if aobj.dtype=='dr2_nilc':  
            ilmin, ilmax = 400, 3000
        else: 
            ilmin, ilmax = 1000, aobj.lmax
        # compute signal and noise spectra but need to change file names
        q = prjlib.init_analysis(**kwargs_cmb)
        q.fcmb.scl = q.fcmb.scl.replace('.dat','_tmp.dat')
        q.fcmb.ocl = q.fcmb.ocl.replace('.dat','_tmp.dat')
        if not misctools.check_path(q.fcmb.scl,**kwargs_ov):
            # compute signal and noise alms
            map2alm_all(aobj.rlz,ilmax,aobj.fimap,aobj.fcmb.alms,wind,aobj.dtype,ibl,aobj.sscale,aobj.nscale,**kwargs_ov)
            alm2aps(aobj.rlz,ilmax,q.fcmb,wn[2],stype=['s','n'],cli_out=False,**kwargs_ov)
        # generate ptsr alm from obs - (sig+noi) spectrum
        gen_ptsr(aobj.rlz,q.fcmb,ibl,aobj.fpseed,aobj.fptsrcl,aobj.fimap,wind,olmax=aobj.lmax,ilmin=ilmin,ilmax=ilmax,**kwargs_ov) 

    # use normal transform to alm
    if aobj.fltr == 'none':
        
        if 'alm' in run:  # combine signal, noise and ptsr
            map2alm_all(aobj.rlz,aobj.lmax,aobj.fimap,aobj.fcmb.alms,wind,aobj.dtype,ibl,aobj.sscale,aobj.nscale,beta=aobj.biref,**kwargs_ov)
            alm_comb(aobj.rlz,aobj.fcmb.alms,**kwargs_ov)

        if 'aps' in run:  # compute cl
            alm2aps(aobj.rlz,aobj.lmax,aobj.fcmb,wn[2],**kwargs_ov)

    # map -> alm with cinv filtering
    if aobj.fltr == 'cinv':  

        falm = aobj.fcmb.alms['c']['T']  #output file of cinv alms
    
        if 'alm' in run:  # cinv filtering here
            wiener_cinv(aobj.rlz,aobj.dtype,M,aobj.lcl[0:3,:p.lmax+1],ibl,aobj.fimap,falm,aobj.sscale,aobj.nscale,beta=aobj.biref,kwargs_ov=kwargs_ov,kwargs_cinv=kwargs_cinv)

        if 'aps' in run:  # aps of filtered spectrum
            alm2aps(aobj.rlz,aobj.lmax,aobj.fcmb,wn[0],stype=['c'],**kwargs_ov)
    

