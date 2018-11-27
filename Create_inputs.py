
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
testname = 'SNR_010'

#Set the measurement properties here

myramps   = [RampTimeSeq('HST/WFC3/IR',13,samp_seq='STEP200')]
myfluxes  = [0.1]
myCRrates = [5e-4]
mybgs     = [None]
mymdict   = None

ntest     = 50#16384


# In[ ]:


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


# #### Function that prepares the random measurement ramp given as set of inputs

# In[ ]:


def one_meas(flux,ramp,CRrate,extra_bg,mdict=None):
    
    CRdict = generateCR(ramp,CRrate=CRrate)
        
    if ramp.detector == 'GENERIC':
        if mdict == None:
            print('Need to specify instrument parameters')
            assert False
        else:
            meas = RampMeasurement(ramp,flux,gain=mdict['gain'],RON=mdict['RON'],KTC=mdict['KTC'],bias=mdict['bias'],full_well=mdict['full_well'],CRdict=CRdict)
    else:
        meas = RampMeasurement(ramp,flux,CRdict=CRdict)

    if extra_bg is not None:
        ebh = get_vbg_electrons(extra_bg['times'],extra_bg['vbg_er'],meas,mean_bg_cr=extra_bg['mean_bg_er'])   
        meas.add_background(ebh)
    else:
        ebh = None

    return meas,CRdict,ebh
        


# #### Cell that iterates over all the input fluxes/ramp/background combinations to create measurements

print('Creating inputs files')

if (len(myfluxes) == len(myramps) == len(myCRrates) == len(mybgs)) == False:
    print('The input parameters have inconsistent lengths')
    assert False


ts = time.time()
inputs = [ [mytuple[0],mytuple[1],mytuple[2],mytuple[3]] for mytuple in zip(myfluxes,myramps,myCRrates,mybgs) for k in range(ntest)]

meas_l = []
CRdict_l = []
extra_bg_l = []
for inp in inputs:
    m,c,e = one_meas(*inp,mdict=mymdict)
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

print('Inputs files created, time elapsed: {} seconds'.format(te-ts))



