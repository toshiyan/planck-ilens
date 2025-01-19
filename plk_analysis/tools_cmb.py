
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


def reduc_map(fmap,TK=2.726,field=0,scale=1.):

    rmap = hp.fitsfunc.read_map(fmap,field=field,verbose=False)

    return rmap * scale / TK


def map2alm(i,s,aobj,wind,scale=1.,mtype=['T','E','B'],**kwargs):

    if aobj.freq == '857': 
        mtype = ['T']

    for m in mtype:
        if misctools.check_path(aobj.fcmb.alms[s][m][i],**kwargs): return

    # convert to alm
    alm = {}
    if 'T' in mtype:
        Tmap  = wind * reduc_map(aobj.fimap[s][i],scale=scale,field=0)
        nside = hp.pixelfunc.get_nside(Tmap)
        alm['T'] = curvedsky.utils.hp_map2alm(nside,aobj.lmax,aobj.lmax,Tmap)

    if 'E' in mtype or 'B' in mtype:
        Qmap  = wind * reduc_map(aobj.fimap[s][i],scale=scale,field=1)
        Umap  = wind * reduc_map(aobj.fimap[s][i],scale=scale,field=2)
        nside = hp.pixelfunc.get_nside(Qmap)
        alm['E'], alm['B'] = curvedsky.utils.hp_map2alm_spin(nside,aobj.lmax,aobj.lmax,2,Qmap,Umap)

        # isotropic rotation
        if s=='s' and i!=0:
            alm['E'], alm['B'] = analysis.ebrotate(aobj.biref,alm['E'],alm['B'])

    # get empirical beam
    #ibl = {}
    #if aobj.freq in ['smica']:
    #    ibl['T'] = 1./local.get_beam(aobj.fimap['s'][0],aobj.lmax,'INT')
    #    ibl['E'] = 1./local.get_beam(aobj.fimap['s'][0],aobj.lmax,'POL')
    #    ibl['B'] = ibl['E']*1.
    #else:
    #    ibl['T'] = local.get_transfer(aobj.freq,aobj.lmax)
    #    ibl['E'] = ibl['I']*1.
    #    ibl['B'] = ibl['I']*1.

    for m in mtype:
        # beam deconvolution
        alm[m] *= aobj.ibl[:,None]
        # save to file
        pickle.dump((alm[m]),open(aobj.fcmb.alms[s][m][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def map2alm_all(aobj,wind,stype=['s','n'],**kwargs):
    
    for i in tqdm.tqdm(aobj.rlz,ncols=100,desc='map2alm:'):
        if i == 0:  # real data
            map2alm(0,'s',aobj,wind,**kwargs)
        else:  # simulation
            for s in stype:
                map2alm(i,s,aobj,wind,**kwargs)


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


def wiener_cinv_core(i,aobj,M,Nij,verbose=True):

    # beam
    bl  = np.reshape(1./aobj.ibl,(1,aobj.lmax+1))
    
    # map
    nside = hp.pixelfunc.get_nside(M)
    TQU   = np.zeros((3,1,12*nside**2))

    if i==0: 
        for field in [0,1,2]:
            TQU[field,0,:] = M * reduc_map(aobj.fimap['s'][i],field=field)
    else:
        # signal
        Ts = M * reduc_map(aobj.fimap['s'][i],field=0)
        Qs = M * reduc_map(aobj.fimap['s'][i],field=1)
        Us = M * reduc_map(aobj.fimap['s'][i],field=2)
        if aobj.biref != 0.:
            Qs, Us = analysis.qurotate(aobj.biref,Qs,Us)
        # noise, ptsr
        Tn = M * reduc_map(aobj.fimap['n'][i],field=0)
        Qn = M * reduc_map(aobj.fimap['n'][i],field=1)
        Un = M * reduc_map(aobj.fimap['n'][i],field=2)
        #Qp = M * reduc_map(aobj.fmap['p'][i].replace('a0.0deg','a1.0deg'),TK=1.,field=2) # approximately use 1.0deg apodization case
        #Up = M * reduc_map(aobj.fmap['p'][i].replace('a0.0deg','a1.0deg'),TK=1.,field=2) # approximately use 1.0deg apodization case
        # combine
        TQU[0,0,:] = Ts + Tn #+ Tp
        TQU[1,0,:] = Qs + Qn #+ Qp
        TQU[2,0,:] = Us + Un #+ Up

    # kwargs
    kwargs_cinv = {\
        'chn'    :1, \
        'nsides' :[nside], \
        'lmaxs'  :[aobj.lmax], \
        'eps'    :[5e-4], \
        'itns'   :[1000], \
        'verbose': verbose, \
        'ro'     :10, \
        'filter' :'W' \
    }
    
    # cinv
    Talm, Ealm, Balm = curvedsky.cninv.cnfilter_freq(3,1,nside,aobj.lmax,aobj.lcl[0:3,:aobj.lmax+1],bl,Nij,TQU,**kwargs_cinv)

    pickle.dump((Talm),open(aobj.fcmb.alms['c']['T'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((Ealm),open(aobj.fcmb.alms['c']['E'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump((Balm),open(aobj.fcmb.alms['c']['B'][i],"wb"),protocol=pickle.HIGHEST_PROTOCOL)


def wiener_cinv(aobj,M,kwargs_ov={}):

    # noise covariance
    Nij = M / aobj.sigma**2
    Nij = np.reshape( ( Nij, Nij/2., Nij/2.) , (3,1,len(M)) )

    # rlz
    for i in tqdm.tqdm(aobj.rlz,ncols=100,desc='wiener cinv:'):
        
        if aobj.biref != 0. and i==0: continue  # avoid real biref case

        if misctools.check_path([aobj.fcmb.alms['c']['T'][i],aobj.fcmb.alms['c']['E'][i],aobj.fcmb.alms['c']['B'][i]],**kwargs_ov): continue

        wiener_cinv_core(i,aobj,M,Nij,verbose=kwargs_ov['verbose'])


def alm2aps(aobj,w2,stype=['s','n','c'],**kwargs_ov):  # compute aps

    eL = np.linspace(0,aobj.lmax,aobj.lmax+1)
    
    for s in stype:
        
        if s == 'n': skip_rlz = [0]
        else: skip_rlz = []
        
        cl = CMB.aps(aobj.rlz,aobj.lmax,aobj.fcmb.alms[s],odd=True,w2=w2,mtype=['T','E','B'],fname=aobj.fcmb.cl[s],skip_rlz=skip_rlz,**kwargs_ov)
    
        if aobj.rlz[-1]>2:  # save mean
            if kwargs_ov['verbose']:  print('cmb alm2aps: save sim')
            i0 = max(0,1-aobj.rlz[0])
            np.savetxt(aobj.fcmb.scl[s],np.concatenate((eL[None,:],np.mean(cl[i0:,:,:],axis=0),np.std(cl[i0:,:,:],axis=0))).T)


def load_fgmap(fgfiles):
    
    Tfg = {}
    for key in fgfiles.keys():
        print(key)
        Tfg[key] = hp.fitsfunc.read_map(fgfiles[key],field=0)
        Qfg[key] = hp.fitsfunc.read_map(fgfiles[key],field=1)
        Ufg[key] = hp.fitsfunc.read_map(fgfiles[key],field=2)
    return Tfg, Qfg, Ufg
        

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


def interface(run=[],kwargs_cmb={},kwargs_ov={}):

    # define parameters, filenames and functions
    aobj = local.init_analysis(**kwargs_cmb)

    # read survey window
    wind, M, wn = local.set_mask(aobj.famask)
    if aobj.fltr == 'cinv':  wn[:] = wn[0]

    # generate ptsr
    if 'fg' in run and aobj.biref==0:
        # compute signal and noise spectra but need to change file names
        q = local.init_analysis(**kwargs_cmb)
        q.fcmb.scl = q.fcmb.scl.replace('.dat','_tmp.dat')
        q.fcmb.ocl = q.fcmb.ocl.replace('.dat','_tmp.dat')
        if not misctools.check_path(q.fcmb.scl,**kwargs_ov):
            # compute signal and noise alms
            map2alm_all(aobj,wind,**kwargs_ov)
            alm2aps(q,wn[2],stype=['s','n'],cli_out=False,**kwargs_ov)
        # generate ptsr alm from obs - (sig+noi) spectrum
        gen_ptsr(aobj.rlz,q.fcmb,aobj.ibl,aobj.fpseed,aobj.fptsrcl,aobj.fimap,wind,olmax=aobj.lmax,ilmin=ilmin,ilmax=ilmax,**kwargs_ov) 

    # use normal transform to alm
    if aobj.freq == '857': 
        mtype = ['T']
    else:
        mtype = ['T','E','B']
    
    if aobj.fltr == 'none':
        
        if 'alm' in run:  # combine signal, noise and ptsr
            map2alm_all(aobj,wind,**kwargs_ov)
            alm_comb(aobj.rlz,aobj.fcmb.alms,mtype=mtype,**kwargs_ov)

        if 'aps' in run:  # compute cl
            alm2aps(aobj,wn[2],**kwargs_ov)

    # map -> alm with cinv filtering
    if aobj.fltr == 'cinv':  

        if 'alm' in run:  # cinv filtering here
            wiener_cinv(aobj,M,kwargs_ov=kwargs_ov)

        if 'aps' in run:  # aps of filtered spectrum
            alm2aps(aobj,wn[0],stype=['c'],**kwargs_ov)
    

