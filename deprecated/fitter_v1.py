import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, uniform, gamma, power_divergence
from scipy.optimize import minimize
import emcee, corner, time

class IterativeFitter(object):

    '''
    A class for the ramp iterative fitter technique

    :RM:
        a ramp.RampMeasurement object

    :fitpars:
        a dictionary with all the fit parameters

    '''
    def __init__(self,RM,fitpars=None):

        self.RM = RM
        if fitpars is not None:
            self.fitpars = fitpars
        else:
            self.fitpars = {'one_iteration_method':'Powell'}

        '''
        Initialize the "hat" values
        '''
        
        self.x_hat = self.RM.noisy_counts
        self.dt    = np.zeros_like(self.x_hat)

        for i in range(1,self.dt.size):
            self.dt[i] = self.RM.RTS.group_times[i] - self.RM.RTS.group_times[i-1] 

        countrates = (self.x_hat[1:]-self.x_hat[:-1])/self.dt[1:]
        self.mean_countrate = np.median(countrates)   #set initially to median to avoid problems with CR in case of few reads
        self.x_new = np.empty_like(self.x_hat)


    def loglikelihood_all(self,x):
        '''
        Function that returns minus the log-likelihood of the measured counts, given a flux
        and the noise characteristics. Does all reads in one

        :x:
            a numpy array of length equal to the number of groups in the ramp

        '''

        xr = np.round(x).astype(np.int_)
        
        poisson_pmf = np.empty_like(x)
        gaussian_pdf = np.empty_like(x)
        
        for i in range(len(xr)):
            gaussian_pdf[i] = norm.pdf(xr[i],loc=self.RM.noisy_counts[i],scale=self.RM.RON_adu)
            if i == 0:
                poisson_pmf[i] = 1.
            else:
                if xr[i] < xr[i-1]:
                    return np.inf
                else:
                    if ~np.isfinite(self.mean_countrate):
                        print('FINITE',self.mean_countrate)
                        print(self.good_intervals)
                        print(self.RM.noisy_counts)
                        print((self.x_new[1:]-self.x_new[:-1])/self.dt[1:])
                        print((xr[1:]-xr[:-1]))
                        print(self.dt[1:])                        
                        print(np.mean( [ (self.x_new[1:]-self.x_new[:-1])/self.dt[1:]  ]))
                        print(i)

                    poisson_pmf[i]  = poisson.pmf(xr[i]-xr[i-1],mu=self.mean_countrate*self.dt[i])
 
        keep_grps = np.empty_like(xr,dtype=np.bool_)
        for i in range(len(xr)):
            if i == 0:
                intdw = i
                intup = i
            elif i==(len(xr)-1):
                intdw = i-1
                intup = i-1
            else:
                intdw = i-1
                intup = i
                
            if ((self.good_intervals[intdw] == True) | (self.good_intervals[intup] == True)):
                keep_grps[i] = True
            else:
                keep_grps[i] = False
                    

        return -1.* ( np.sum(np.log(gaussian_pdf)[keep_grps]) + np.sum(np.log(poisson_pmf[1:])[self.good_intervals]) )


    def one_iteration(self):

        '''
        This function calls the scipy minimize routine, passing all the needed arguments to it.
        After minimize returns, one_iteration does some book keeping
        '''

        self.x_new = np.round(minimize(self.loglikelihood_all,self.x_hat,method=self.fitpars['one_iteration_method'])['x'])
        countrates = (self.x_new[1:]-self.x_new[:-1])/self.dt[1:]
        self.mean_countrate = np.mean(countrates[self.good_intervals])
        if self.mean_countrate <= 0.:
            self.mean_countrate = 1e-5
        self.x_hat = self.x_new


    def perform_fit(self,thr=None,maxCRiter=10,maxiter=10,CRthr=4.):

        '''
        Wrapper function to perform the up-the-ramp fit, test for cosmic rays, check convergence, issue error status

        :thr:
            Threshold for convergence

        :maxiter:
            Maximum number of iterations

        :CRthr:
            Threshold for CR hits flagging. Represents the number of standard deviations a single delta-group must differ from the expectation, in order to be flagged 
        
        '''

        if thr is None:
            thr = 1./self.RM.FWC_adu/self.RM.RTS.group_times[-1] 

        old_mean_countrate = self.mean_countrate
            
        #Initial flagging of CR hits
        stddev = np.sqrt(self.mean_countrate*self.dt[1:]+2*np.square(self.RM.RON_adu))
        self.good_intervals = np.fabs((self.RM.noisy_counts[1:]-self.RM.noisy_counts[:-1] - self.mean_countrate*self.dt[1:])/stddev) < CRthr

        check_CRs  = 1
        crloops_counter = 0

        while check_CRs:
            crloops_counter = crloops_counter + 1

            if (np.any(self.good_intervals) ==  True):
                check_conv = 1
                counter    = 0
                error      = 0
            else:
                counter    = 0
                error      = 2
                return error, counter, self.good_intervals, crloops_counter

            while check_conv:
                self.one_iteration()
                counter = counter+1
                if np.fabs(self.mean_countrate-old_mean_countrate)/old_mean_countrate < thr:
                    check_conv = 0
                if (counter > maxiter):
                    error = 1
                    return error, counter, self.good_intervals, crloops_counter
                
                old_mean_countrate = self.mean_countrate

            #test here for CR presence
            stddev = np.sqrt(self.mean_countrate*self.dt[1:]+2*np.square(self.RM.RON_adu))
            new_good_intervals = np.fabs((self.RM.noisy_counts[1:]-self.RM.noisy_counts[:-1] - self.mean_countrate*self.dt[1:])/stddev) < CRthr

            if np.array_equal(self.good_intervals,new_good_intervals):
                check_CRs = 0
            else: 
                self.good_intervals = new_good_intervals
                countrates = (self.x_new[1:]-self.x_new[:-1])/self.dt[1:]
                self.mean_countrate = np.mean(countrates[self.good_intervals])
   
                
            if (crloops_counter > maxCRiter):
                error = 3
                return error, counter, self.good_intervals, crloops_counter



        return error, counter, self.good_intervals, crloops_counter




    def goodness_of_fit(self,mode='G-test'):

        '''
        Method to perform a goodness of fit test of the derived count rate.
        It compares the expected counts (from the final rate atimes the interval times) with the "observed counts" the latter are obtained
        not from the actual data, but from the poisson latent variable (i.e. before read noise).
        

        :mode: the type of test perfomed.

               Possible values are 'G-test', 'Pearson-chi-sq'

               G-test: (https://en.wikipedia.org/wiki/G-test)
                   This is the default value.
                   The G-test statistics, based on a likelihood ratio, is a better approximation to the chi-squared distribution
                   than Pearson's chi-square, which fails for small number counts

               Pearson-chi-sq: (https://en.wikipedia.org/wiki/Pearson's_chi-squared_test)
                   Pearsons' chi square is implemented and should give similar results for moderately large observed count rates
        '''


        f_obs = self.RM.gain*(self.x_new[1:]-self.x_new[:-1])[self.good_intervals]
        f_exp = self.RM.gain*(self.mean_countrate * self.dt[1:])[self.good_intervals]
        ddof  = 1
        dof   = np.sum(self.good_intervals) - 1 - ddof

        if mode == 'G-test':
            g,p = power_divergence(f_obs, f_exp=f_exp, ddof=ddof,  lambda_='log-likelihood')

        elif mode == 'Pearson-chi-sq':
            g,p = power_divergence(f_obs, f_exp=f_exp, ddof=ddof,  lambda_='pearson')

        else:
            print('Goodness of fit test type not supported')
        
        self.gof_stat = g
        self.gof_pval = p

    def test_plot(self):
        '''
        Method to plot the fit results
        '''
        f,ax = plt.subplots(1,1,figsize=(10,5))
        ax.scatter(self.RM.RTS.group_times,self.RM.noisy_counts,label='Noisy Counts',s=100,marker='*')
        ax.scatter(self.RM.RTS.group_times,self.x_new,label='Convergence counts',s=25)
        ax.plot(self.RM.RTS.group_times,self.x_new[0]+self.mean_countrate*self.RM.RTS.group_times)

        for j,gi in enumerate(self.good_intervals):
            if ~gi:
                ax.axvline(0.5*(self.RM.RTS.group_times[j]+self.RM.RTS.group_times[j+1]),color='#bbbbbb',linestyle='--')
            
        ax.legend()
        f.tight_layout()


