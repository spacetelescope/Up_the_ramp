
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
testname = 'R14'

outputs_file = dirsave+'Test_'+testname+'_out.pbz2'
inputs_file  = dirsave+'Test_'+testname+'_in.pbz2'



# ### Setup the fitter method and options

fitpars = {'one_iteration_method':'Nelder-Mead'}


# ### Setup the job

# In[ ]:

printevery = 1
n_jobs     = 10
chunksize  = 1 


# ### Run the fitter on multiple ramps

# #### Function that does the fit of individual measurements

def one_fit(l,meas):

    if (l % printevery) == 0:
        print("Starting fit {} of {}".format(l, len(meas_l)))
        sys.stdout.flush()

    fitter = IterativeFitter(meas,fitpars = fitpars)
    error,counter, goodints, crloops_counter  = fitter.perform_fit()
    outerate = fitter.mean_electron_rate
    fitter.goodness_of_fit(mode='full-likelihood')
    gof_stat = fitter.gof_stat
    gof_pval = fitter.gof_pval

    if (l % printevery) == 0:
        print("Finished fit {} of {}".format(l, len(meas_l)))
        sys.stdout.flush()
     
    return goodints,counter,error,crloops_counter,outerate,gof_stat,gof_pval


# #### Cell that iterates over all the input fluxes/ramp/background combinations to create measurements

if os.path.isfile(inputs_file) == True: 
    print('Loading input file')
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
    print('Inputs file missing')
    assert False



# #### Cell that iterates over single measurements

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

