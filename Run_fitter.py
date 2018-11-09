
# coding: utf-8

# ## Notebook to run the fitter on multiple ramps

# In[ ]:


import numpy as np
from ramp_utils.ramp import RampTimeSeq,RampMeasurement
from ramp_utils.fitter import IterativeFitter
import time, sys, pickle, bz2, os
from multiprocessing import Pool
from scipy.interpolate import interp1d



# ### Set the working directory for saving the results

# In[ ]:


dirsave = '/user/gennaro/Functional_work/Up_the_ramp_myfork/Simulations_results/'
testname = '1'


# In[ ]:


outputs_file = dirsave+'Test_'+testname+'_out.pbz2'
inputs_file  = dirsave+'Test_'+testname+'_in.pbz2'


# #### These are two auxiliary function used to generate CRhits 
# #### A good number for WFC3 can be found in http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2009-40.pdf, with ~ 2.5e-5 hits per second per pixel
# 

# In[ ]:


def generateCR_DNs(size):
    '''
    Just generate random uniform DNs
    '''
    return 1000.*np.random.uniform(size=size)

def generateCR(myramp,CRrate=2.5e-5):
    '''
    Main CR generation function that generates CR hit times and depositied counts
    
    :CRrate:
        rate of CR hits in hits/second
    '''
    
    myCRnumber = np.random.poisson(lam=CRrate*myramp.read_times[-1])

    if myCRnumber > 0:
        mytimes = myramp.read_times[-1]*np.random.uniform(size=myCRnumber)
        mycounts = generateCR_DNs(myCRnumber)
        myCRdict = {'times':mytimes.tolist(),'counts':mycounts.tolist()}
    else:
        myCRdict = None

    return myCRdict


# #### Auxiliary function to generate cumulative background electrons
# Used to test the fitter with non-constant flux

# In[ ]:


def get_vbg_electrons(times,vbg_cr,meas,mean_bg_cr=None):
    '''
    Given a tabulated form for the variable background time dependency,
    generate a number of electrons per each read interval in the ramp
    
    :times:
        times at which the countrate is tabulated
    
    :vbg_cr:
        values of the time variable background at those times   
    
    :meas:
        a RampMeasurement object
    
    :mean_bg_cr:
        the mean value for normalizing the countrate within the interval    
    '''

    
    #Create an interpolator from the tabulated values and interpolate at the ramp read times
    bg_int = interp1d(times,vbg_cr,'quadratic')
    varbg = bg_int(meas.RTS.read_times)

    #Normalize if requested
    if mean_bg_cr is not None:
        dt = meas.RTS.read_times[-1]-meas.RTS.read_times[0]
        t_avg = np.trapz(varbg,meas.RTS.read_times) / dt
        varbg = varbg/t_avg * mean_bg_cr

    #Get the total accumulated electrons
    bg_e=[0]
    bg_e.extend([np.random.poisson(lam=vb*dt) for vb,dt in zip(0.5*(varbg[1:]+varbg[:-1]),meas.RTS.read_times[1:]-meas.RTS.read_times[:-1]) ])
    bg_e = np.asarray(bg_e)
    
    return np.cumsum(bg_e)


# ### Define the read sequences

# In[ ]:


dt,nf,ns,ng = 6.,1,0,17
myramp1 = RampTimeSeq('GENERIC',ng,nframes=nf, nskips=ns, read_times=dt*np.arange(ng*(nf+ns)))

dt,nf,ns,ng = 8.,1,0,13
myramp2 = RampTimeSeq('GENERIC',ng,nframes=nf, nskips=ns, read_times=dt*np.arange(ng*(nf+ns)))

dt,nf,ns,ng = 12.,1,0,9
myramp3 = RampTimeSeq('GENERIC',ng,nframes=nf, nskips=ns, read_times=dt*np.arange(ng*(nf+ns)))

dt,nf,ns,ng = 16.,1,0,7
myramp4 = RampTimeSeq('GENERIC',ng,nframes=nf, nskips=ns, read_times=dt*np.arange(ng*(nf+ns)))

dt,nf,ns,ng = 24.,1,0,5
myramp5 = RampTimeSeq('GENERIC',ng,nframes=nf, nskips=ns, read_times=dt*np.arange(ng*(nf+ns)))


myramp6 = RampTimeSeq('HST/WFC3/IR',15,samp_seq='SPARS100') 

dt,nf,ns,ng = 10., 1, 0, 10
ramp_R14_1 = RampTimeSeq('GENERIC',ng,nframes=nf, nskips=ns, read_times=dt*np.arange(ng*(nf+ns)))

dt,nf,ns,ng = 10., 4, 0, 10
ramp_R14_4 = RampTimeSeq('GENERIC',ng,nframes=nf, nskips=ns, read_times=dt*np.arange(ng*(nf+ns)))

# ### Define the detector charachteristics
# This step is necessary when the ramps are of **GENERIC** type

# In[ ]:


gain=1.
RON=15.
KTC=0.
bias=10000
full_well=100000


# ### Setup the properties of the measurements on which to run the fitter

# In[ ]:


#myfluxes   = [   0.05,     4,      64,    0.5,    0.5,    0.5,    0.5]
#myramps    = [myramp6,myramp6,myramp6,myramp6,myramp6,myramp6,myramp6]
#myCRrates  = [   5e-4,   5e-4,   5e-4,     0.,     0.,     0.,     0.]
#tbg  = np.linspace(0,1500,10)
#cbg  = np.array([1.0,1.2,1.5,1.3,1.7,2.0,2.2,2.4,2.0,1.5])

#mybgs      = [   None,   None,   None,   None, 
#               {'times':tbg,'vbg_er':cbg,'mean_bg_er':1.},
#               {'times':tbg,'vbg_er':np.power(cbg,3),'mean_bg_er':1.},
#               {'times':tbg,'vbg_er':cbg,'mean_bg_er':2.},]

myfluxes  = [        9.,        9.]
myramps   = [ramp_R14_1,ramp_R14_4]
myCRrates = [        0.,        0.]
mybgs     = [      None,      None]

if (len(myfluxes) == len(myramps) == len(myCRrates) == len(mybgs)) == False:
    assert False


# ### Setup the fitter method and options

# In[ ]:


fitpars = {'one_iteration_method':'Nelder-Mead'}


# ### Setup the job

# In[ ]:


ntest      = 10000
printevery = 500
n_jobs     = 30
chunksize  = 15 


# ### Run the fitter on multiple ramps

# #### Function that does the fit of individual measurements

# In[ ]:


def one_fit(l,meas):

    if (l % printevery) == 0:
        print("Starting fit {} of {}".format(l, len(meas_l)))
        sys.stdout.flush()

    fitter = IterativeFitter(meas,fitpars = fitpars)
    error,counter, goodints, crloops_counter  = fitter.perform_fit()
    outerate = fitter.mean_electron_rate
    fitter.goodness_of_fit(mode='poisson-likelihood')
    gof_stat = fitter.gof_stat
    gof_pval = fitter.gof_pval
     
    return goodints,counter,error,crloops_counter,outerate,gof_stat,gof_pval


# #### Function that prepares the random measurement ramp given as set of inputs

# In[ ]:


def one_meas(flux,ramp,CRrate,extra_bg):
    
    CRdict = generateCR(ramp,CRrate=CRrate)
        
    if ramp.detector == 'GENERIC':
        meas = RampMeasurement(ramp,flux,gain=gain,RON=RON,KTC=KTC,bias=bias,full_well=full_well,CRdict=CRdict)
    else:
        meas = RampMeasurement(ramp,flux,CRdict=CRdict)

    if extra_bg is not None:
        ebh = get_vbg_electrons(extra_bg['times'],extra_bg['vbg_er'],meas,mean_bg_cr=extra_bg['mean_bg_er'])   
        meas.add_background(ebh)
    else:
        ebh = None

    return meas,CRdict,ebh
        


# #### Cell that iterates over all the input fluxes/ramp/background combinations to create measurements

# In[ ]:


if os.path.isfile(inputs_file) == True: 
    print('Input file already existing....loading')
    with bz2.BZ2File(inputs_file, 'rb') as f:
        dictoload  = pickle.load(f)
    meas_l     = dictoload['meas_l']
    myfluxes   = dictoload['myfluxes']
    myramps    = dictoload['myramps']
    myCRrates  = dictoload['myCRrates']
    mybgs      = dictoload['mybgs']
    CRdict_l   = dictoload['CRdict_l']
    extra_bg_l = dictoload['extra_bg_l']
    ntest      = len(meas_l)
    del(dictoload)
    
else:
    print('Creating inputs files')
    mypool = Pool(n_jobs)

    ts = time.time()
    inputs = [ [mytuple[0],mytuple[1],mytuple[2],mytuple[3]] for mytuple in zip(myfluxes,myramps,myCRrates,mybgs) for k in range(ntest)]

    meas_l = []
    CRdict_l = []
    extra_bg_l = []
    for inp in inputs:
        m,c,e = one_meas(*inp)
        meas_l.append(m)
        CRdict_l.append(c)
        extra_bg_l.append(e)
    te = time.time()
    dicttosave={'myfluxes':myfluxes,
                'myramps':myramps,
                'myCRrates':myCRrates,
                'mybgs':mybgs,
                'meas_l':meas_l,
                'CRdict_l':CRdict_l,
                'extra_bg_l':extra_bg_l
                }
    
    with bz2.BZ2File(inputs_file, 'w') as f:
        pickle.dump(dicttosave, f)
    mypool.close()
    mypool.join()
    del(dicttosave)
    
    print('Created inputs files, time elapsed: {} seconds'.format(te-ts))



# #### Cell that iterates over single measurements

# In[ ]:


mypool = Pool(n_jobs)

inputs = [ [l,meas] for l,meas in enumerate(meas_l)]
ts = time.time()
goodints_l,counter_l,error_l,crloops_counter_l,outerate_l,gof_stat_l,gof_pval_l = map(list, zip(*mypool.starmap(one_fit,inputs,chunksize)))
te = time.time()

mypool.close()
mypool.join()
print('Elapsed time [minutes]: {}'.format((te-ts)/60.))
print('Time per fit [s]: {}'.format((te-ts)/len(myfluxes)/ntest))


# ### Save the results

# In[ ]:


dicttosave = {'goodints_l':goodints_l,
              'counter_l':counter_l,
              'error_l':error_l,
              'crloops_counter_l':crloops_counter_l,
              'outerate_l':outerate_l,
              'gof_stat_l':gof_stat_l,
              'gof_pval_l':gof_pval_l,
             }

with bz2.BZ2File(outputs_file, 'w') as f:
        pickle.dump(dicttosave, f)

