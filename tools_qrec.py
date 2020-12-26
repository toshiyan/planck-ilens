# Reconstruction using quadratic estimator
import numpy as np
import healpy as hp
import pickle
import tqdm

# from cmblensplus/wrap
import curvedsky
import basic

# from cmblensplus/utils
import quad_func
import misctools

# local
import local


def init_quad(ids,stag,rlz=[],rlmin=100,rlmax=2048,**kwargs):
    
    d = prjlib.data_directory()

    kwargs = {}
    kwargs['ilens'] = {\
        'rlmin':rlmin, \
        'rlmax':rlmax, \
        'n0min':1, \
        'n0max':int(kwargs['snmax']/2), \
        'mfmin':1, \
        'mfmax':kwargs['snmax'], \
        'rdmin':1, \
        'rdmax':kwargs['snmax'], \
        'qlist':['TB','EB'], \
        'overwrite':kwargs['overwrite'], \
        'verbose':kwargs['verbose'] \
    }
    
    kwargs['lens'] = kwargs['ilens']
    kwargs['lens']['qlist'] = ['TT','TE','EE','TB','EB']
    
    # setup parameters for lensing reconstruction (see cmblensplus/utils/quad_func.py)
    return {qtype: quad_func.quad(rlz=rlz,stag=stag,root=d['root'],ids=ids,qtype=qtype,**kwargs[qtype]) for qtype in ['lens','ilens'] }


def aps(rlz,qobj,fklm=None,fcib=None,q='TT',**kwargs_ov):

    cl = np.zeros((len(rlz),5,qobj.olmax+1))
    
    for i in tqdm.tqdm(rlz,ncols=100,desc='aps ('+qobj.qtype+')'):
        
        if misctools.check_path(qobj.f[q].cl[i],**kwargs_ov): continue
        
        # load qlm
        alm = pickle.load(open(qobj.f[q].alm[i],"rb"))[0]
        mf  = pickle.load(open(qobj.f[q].mfb[i],"rb"))[0]
        alm -= mf

        # auto spectrum
        cl[i,0,:] = curvedsky.utils.alm2cl(qobj.olmax,alm)/qobj.wn[4]

        if fklm is not None and i>0:
            # load input klm
            iKlm = hp.fitsfunc.read_alm(fklm[i])
            iklm = curvedsky.utils.lm_healpy2healpix(iKlm,2048)        
            # cross with input
            cl[i,1,:] = curvedsky.utils.alm2cl(qobj.olmax,alm,iklm)/qobj.wn[2]
            # input
            cl[i,2,:] = curvedsky.utils.alm2cl(qobj.olmax,iklm)

        if fcib is not None:
            # load cib alm
            Ilm = pickle.load(open(fcib[i],"rb"))
            # cross with cib
            cl[i,3,:] = curvedsky.utils.alm2cl(qobj.olmax,alm,Ilm)/qobj.wn[2]
            # cib
            cl[i,4,:] = curvedsky.utils.alm2cl(qobj.olmax,Ilm)

        np.savetxt(qobj.f[q].cl[i],np.concatenate((qobj.l[None,:],cl[i,:,:])).T)

    # save to files
    if rlz[-1] >= 2:
        i0 = max(0,1-rlz[0])
        np.savetxt(qobj.f[q].mcls,np.concatenate((qobj.l[None,:],np.average(cl[i0:,:,:],axis=0),np.std(cl[i0:,:,:],axis=0))).T)


def interface(run=['norm','qrec','n0','mean'],qtypes=['lens','ilens'],kwargs_ov={},kwargs_cmb={},rlmin=100,rlmax=2048,ep=1e-30):
    
    p = prjlib.init_analysis(**kwargs_cmb)
    __, __, wn = prjlib.set_mask(p.famask)

    #//// load obscls ////#
    if p.fltr == 'none':
        ocl = np.loadtxt(p.fcmb.scl,unpack=True)[4:5]
        ifl = ocl.copy()

    if p.fltr == 'cinv':
        bl  = np.loadtxt(p.fbeam)[:p.lmax+1]
        cnl = p.lcl[0,:] + (1./bl)**2*(30.*np.pi/10800./2.72e6)**2
        wcl = np.loadtxt(p.fcmb.scl,unpack=True)[4]

        # quality factor defined in Planck 2015 lensing paper
        Ql  = (p.lcl[0,:])**2/(wcl*cnl+ep**2)
        # T' = QT^f = Q/(cl+nl) * (T+n)/sqrt(Q)

        ocl = cnl/(Ql+ep)  # corrected observed cl
        ifl = p.lcl[0,:]/(Ql+ep)    # remove theory signal cl in wiener filter
        ocl = np.reshape(ocl,(1,p.lmax+1))
        ifl = np.reshape(ifl,(1,p.lmax+1))

        wn[:] = wn[0]

    #//// define objects ////#
    qobj = init_quad(p.ids,p.stag,rlz=p.rlz,wn=wn,lcl=p.lcl,ocl=ocl,ifl=ifl,falm=p.fcmb.alms['c'],rlmin=rlmin,rlmax=rlmax,**kwargs_cmb,**kwargs_ov)

    # reconstruction
    for qtype in qtypes:
        qobj[qtype].qrec_flow(run=run)
        if 'aps' in run:
            bobj = aobj
            bobj.freq = '857'
            #aobj = local.init_analysis(freq='857')
            for q in qobj[qtype].qlist:
                aps(p.rlz,qobj[qtype],fklm=p.fiklm,fcib=aobj.fcmb.alms['c']['T'],q=q,**kwargs_ov)

