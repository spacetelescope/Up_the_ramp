import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, uniform, gamma, power_divergence, chi2
from scipy.optimize import minimize
from scipy.optimize.optimize import _minimize_neldermead

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
            self.fitpars = {'one_iteration_method':'Nelder-Mead'}



        '''
        Initialize the dt array (differences between the group times)
        Also inizialize an array that contains the covariance error term for the count differences within an interval. This is the sum
        of two lower-traingular matrices sum (derived from the second term in eq(4) of Roberto (2010) JWST-STScI-002161)
        '''
        
        self.dt    = np.zeros_like(self.RM.noisy_counts,dtype=np.float_)
        self.triangle_sums = np.zeros_like(self.RM.noisy_counts,dtype=np.float_)
        for i in range(1,self.dt.size):
            self.dt[i] = self.RM.RTS.group_times[i] - self.RM.RTS.group_times[i-1] 
            self.triangle_sums[i] = self.RM.RTS.lower_triangle_sum[i] + self.RM.RTS.lower_triangle_sum[i-1]
            
        '''
        Initialize the "hat" values. These are the latent poisson variables "actual number of electrons deposited in each interval"
        The _new values are the auxiliary variables used to iterate over the _hat ones
        '''
        
        self.x_hat = np.floor(self.RM.noisy_counts*self.RM.gain)
        
        for i in range(1,self.x_hat.size):
            if self.x_hat[i] < self.x_hat[i-1]:
                self.x_hat[i] = self.x_hat[i-1]

        self.x_hat = self.x_hat+np.mean(self.RM.noisy_counts*self.RM.gain)-np.mean(self.x_hat)

        electron_rates = (self.x_hat[1:]-self.x_hat[:-1])/self.dt[1:]
        self.mean_electron_rate = np.median(electron_rates)   #set initially to median to avoid problems with CR in case of few reads

        self.x_new = np.copy(self.x_hat)
        


        '''
        Freeze the normal pdfs to compare "true" and measured counts and the poisson pdf to compare the
        mean and "true" electrons
        '''

        self.normal_distr = []
        self.poisson_distr = []
        for i in range(len(self.RM.noisy_counts)):
            self.normal_distr.append(norm(loc=(self.RM.noisy_counts[i]*self.RM.gain),scale=self.RM.RON_e))
            self.poisson_distr.append(poisson(mu=self.mean_electron_rate*self.dt[i]))
            
            
            
        '''
        Initialize a couple of auxiliary variables needed to compute the noise for each count difference
        '''
        self.var_RON_per_diff    = 2.*np.square(self.RM.RON_e)/self.RM.RTS.nframes
        self.var_quant_per_diff  = 2./12.*np.square(self.RM.gain*self.RM.RTS.nframes)


    def loglikelihood_all(self,x):
        '''
        Function that returns minus the log-likelihood of the measured counts, given an electron flux
        and the noise characteristics. Does all reads in one

        :x:
            a numpy array of length equal to the number of groups in the ramp

        '''

        xr = np.round(x).astype(np.int_)
        
        poisson_pmf  = np.empty_like(x)
        gaussian_pdf = np.empty_like(x)
        
        for i in range(len(xr)):
            if i == 0:
                poisson_pmf[i] = 1.
            else:
                if xr[i] < xr[i-1]:
                    return np.inf
                else:
                    poisson_pmf[i]  = self.poisson_distr[i].pmf(xr[i]-xr[i-1])
            gaussian_pdf[i] = self.normal_distr[i].pdf(xr[i])
 
        keep_grps = np.empty_like(xr,dtype=np.bool_)
        for i in range(len(xr)):
            if i == 0:
                intdw = i
                intup = i
            elif i == (len(xr)-1):
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


    def one_iteration(self,conv_attempts = 5):

        '''
        This function calls the scipy minimize routine, passing all the needed arguments to it.
        After minimize returns, one_iteration does some book keeping
        '''

        success,attempts = False, 0
        while success == False:
            attempts = attempts + 1
            
            if self.fitpars['one_iteration_method'] == 'Nelder-Mead':
                '''
                Define the initial simplex
                '''
                sim = np.empty((self.x_hat.size + 1, self.x_hat.size), dtype=self.x_hat.dtype)
                sim[0] = self.x_hat
                for k in range(self.x_hat.size):
                    y = self.x_new + np.random.normal(scale=self.RM.RON_e,size=self.x_hat.size)
                    for i in range(1,self.x_hat.size):
                        if y[i] < y[i-1]:
                            y[i] = y[i-1]
                    sim[k+1] = y+np.mean(self.RM.noisy_counts*self.RM.gain)-np.mean(y)

            
                self.minimize_res = _minimize_neldermead(self.loglikelihood_all,self.x_hat,initial_simplex=sim)
            else:
                self.minimize_res = minimize(self.loglikelihood_all,self.x_hat,method=self.fitpars['one_iteration_method'])
            
            
            success = self.minimize_res['success']
            if success == True:
                self.x_new = np.round(self.minimize_res['x'])
                self.x_hat = np.copy(self.x_new)
            else:
                '''
                If the minimizer did not converge, restart from slightly different initial condistions
                '''
                self.x_hat = self.x_new + np.random.normal(scale=self.RM.RON_e,size=self.x_hat.size)
                for i in range(1,self.x_hat.size):
                    if self.x_hat[i] < self.x_hat[i-1]:
                        self.x_hat[i] = self.x_hat[i-1]
                self.x_hat = self.x_hat+np.mean(self.RM.noisy_counts*self.RM.gain)-np.mean(self.x_hat)


                if attempts >= conv_attempts:
                    success = True
                    self.x_new = np.round(self.minimize_res['x'])
                    self.x_hat = np.copy(self.x_new)
        
        electron_rates = (self.x_new[1:]-self.x_new[:-1])/self.dt[1:]
        
        weights = np.square(self.dt[1:]/self.stddev)
        
        self.mean_electron_rate = np.average(electron_rates[self.good_intervals],weights=weights[self.good_intervals])
        
        for i in range(len(self.RM.noisy_counts)):
            self.poisson_distr[i] = poisson(mu=self.mean_electron_rate*self.dt[i])




    def perform_fit(self,thr=None,maxCRiter=10,maxiter=20,CRthr=4.):

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
            '''
            Two values of the flux that produce, on average, less than one count of difference within that ramp, cannot be distinguished,
            this is the ultimate threshold for convergence (hence the gain term in the squared sum below).
            To this we add the standard deviation of the effective noise: given the same mean flux, one cannot distinguish two measurements
            to better than this noise floor, hence the effRON_e term in the sum
            '''
            
            #thr = np.sqrt(np.sum(np.square(np.array([self.RM.gain,self.RM.effRON_e])))) /self.RM.RTS.group_times[-1] 
            thr = self.RM.gain /self.RM.RTS.group_times[-1]
            


        old_mean_electron_rate = self.mean_electron_rate
            
        # Initial flagging of CR hits. Count-differences that deviate more than a certain threshold from the noise are flagged
        # The noise is computed starting from eq(4) of Robberto (2010), JWST-STScI-002161
        
        self.var_signal_per_diff = (self.RM.RTS.group_times[1:]/self.RM.RTS.nframes
                                    + self.RM.RTS.group_times[:-1]*(1./self.RM.RTS.nframes -2)
                                    + 2./np.square(self.RM.RTS.nframes)*self.triangle_sums[1:]
                                    ) * self.mean_electron_rate
        self.stddev = np.sqrt(self.var_signal_per_diff+self.var_RON_per_diff+self.var_quant_per_diff)
        deltas  = self.RM.gain*(self.RM.noisy_counts[1:]-self.RM.noisy_counts[:-1]) - self.mean_electron_rate*self.dt[1:]
        self.good_intervals = np.fabs( deltas/self.stddev) < CRthr

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
                if np.fabs(self.mean_electron_rate-old_mean_electron_rate) < thr:
                    check_conv = 0
                if (counter > maxiter):
                    error = 1
                    return error, counter, self.good_intervals, crloops_counter
                
                old_mean_electron_rate = self.mean_electron_rate
                

            #test here for CR presence
            self.var_signal_per_diff = (self.RM.RTS.group_times[1:]/self.RM.RTS.nframes
                            + self.RM.RTS.group_times[:-1]*(1./self.RM.RTS.nframes -2)
                            + 2./np.square(self.RM.RTS.nframes)*self.triangle_sums[1:]
                            ) * self.mean_electron_rate
            self.stddev = np.sqrt(self.var_signal_per_diff+self.var_RON_per_diff+self.var_quant_per_diff)
            deltas  = self.RM.gain*(self.RM.noisy_counts[1:]-self.RM.noisy_counts[:-1]) - self.mean_electron_rate*self.dt[1:]
            new_good_intervals = np.fabs(deltas/self.stddev) < CRthr

            if np.array_equal(self.good_intervals,new_good_intervals):
                check_CRs = 0
            else: 
                self.good_intervals = new_good_intervals
                electron_rates = (self.x_new[1:]-self.x_new[:-1])/self.dt[1:]
                self.mean_electron_rate = np.average(electron_rates[self.good_intervals],weights=np.square(1./self.stddev[self.good_intervals]))
   
                
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

               Possible values are 'G-test', 'Pearson-chi-sq', 'Squared-deviations'

               G-test: (https://en.wikipedia.org/wiki/G-test)
                   This is the default value.
                   The G-test statistics, based on a likelihood ratio, is a better approximation to the chi-squared distribution
                   than Pearson's chi-square, which fails for small number counts

               Pearson-chi-sq: (https://en.wikipedia.org/wiki/Pearson's_chi-squared_test)
                   Pearsons' chi square is implemented and should give similar results for moderately large observed count rates

               Squared deviations: (https://en.wikipedia.org/wiki/Reduced_chi-squared_statistic)
                   Use the variance of the counts plus the variance of the readnoise, summed together, as the
                   denominator 
        '''


        f_obs = (self.RM.noisy_counts[1:]-self.RM.noisy_counts[:-1])[self.good_intervals]
        f_exp = (self.mean_electron_rate * self.dt[1:]/self.RM.gain)[self.good_intervals]

        if mode == 'G-test':
            ddof  = 1
            dof   = np.sum(self.good_intervals) - 1 - ddof
            g,p = power_divergence(f_obs, f_exp=f_exp, ddof=ddof,  lambda_='log-likelihood')

        elif mode == 'Pearson-chi-sq':
            ddof  = 1
            dof   = np.sum(self.good_intervals) - 1 - ddof
            g,p = power_divergence(f_obs, f_exp=f_exp, ddof=ddof,  lambda_='pearson')

        elif mode == 'Squared-deviations':
            var_signal_per_diff = (self.RM.RTS.group_times[1:]/self.RM.RTS.nframes
                                   + self.RM.RTS.group_times[:-1]*(1./self.RM.RTS.nframes -2)
                                   + 2./np.square(self.RM.RTS.nframes)*self.triangle_sums[1:]
                                   ) * self.mean_electron_rate
            
            variance = var_signal_per_diff[self.good_intervals]+self.var_RON_per_diff+self.var_quant_per_diff
            variance = variance / np.square(self.RM.gain)
            dof   = np.sum(self.good_intervals) - 1            
            g = np.sum(np.square(f_obs-f_exp)/variance)
            p = chi2.sf(g,dof)      


        else:
            print('Goodness of fit test type not supported')
            assert False
        
        self.gof_stat = g
        self.gof_pval = p

    def test_plot(self):
        '''
        Method to plot the fit results
        '''
        f,ax = plt.subplots(1,1,figsize=(10,5))
        ax.scatter(self.RM.RTS.group_times,self.RM.noisy_counts,label='Noisy Counts',s=100,marker='*')
        ax.scatter(self.RM.RTS.group_times,self.x_new/self.RM.gain,label='Convergence counts',s=25)
        ax.scatter(self.RM.RTS.group_times,self.RM.noisy_counts-self.RM.RON_effective/self.RM.gain,label='Noiseless Counts + \n Bias + KTC + CRs',s=25)
        ax.scatter(self.RM.RTS.group_times,self.RM.noisy_counts-(self.RM.RON_effective-np.mean(self.RM.RON_effective))/self.RM.gain,label='Noiseless Counts + \n Bias + KTC +\n mean RON',s=25)
        
        ax.plot(self.RM.RTS.group_times,(self.x_new[0]+self.mean_electron_rate*self.RM.RTS.group_times)/self.RM.gain)

        for j,gi in enumerate(self.good_intervals):
            if ~gi:
                ax.axvline(0.5*(self.RM.RTS.group_times[j]+self.RM.RTS.group_times[j+1]),color='#bbbbbb',linestyle='--')
            
        ax.legend()
        f.tight_layout()


