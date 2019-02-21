import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, uniform, gamma, power_divergence, chi2
from scipy.optimize import minimize
from scipy.optimize.optimize import _minimize_neldermead
import sys

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
        for i in range(self.RM.RTS.ngroups):
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

        #xr = np.round(x).astype(np.int_) #Mario: rounded version
        
        poisson_pmf  = np.empty_like(x)
        gaussian_pdf = np.empty_like(x)
        
        for i in range(self.RM.RTS.ngroups):
            if i == 0:
                poisson_pmf[i] = 1.
            else:
                if x[i] < x[i-1]:  #Mario: rounded version
                    return np.inf
                else:
                    poisson_pmf[i]  = self.poisson_distr[i].pmf(np.round(x[i]-x[i-1]).astype(np.int_)) #Mario: rounded version
            gaussian_pdf[i] = self.normal_distr[i].pdf(x[i]) #Mario: rounded version
 
        keep_grps = np.empty_like(x,dtype=np.bool_) #Mario: rounded version
        for i in range(self.RM.RTS.ngroups):
            if i == 0:
                intdw = i
                intup = i
            elif i == (self.RM.RTS.ngroups-1):
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
                    sim[k+1] = y + np.mean(self.RM.noisy_counts*self.RM.gain) - np.mean(y)
            
                self.minimize_res = _minimize_neldermead(self.loglikelihood_all,self.x_hat,initial_simplex=sim)
            else:
                self.minimize_res = minimize(self.loglikelihood_all,self.x_hat,method=self.fitpars['one_iteration_method'])
            
            
            success = self.minimize_res['success']
            if success == True:
                self.x_new = np.copy(self.minimize_res['x'])  #Mario: rounded version
                self.x_hat = np.copy(self.x_new)
            else:
                '''
                If the minimizer did not converge, restart from slightly different initial conditions
                '''
                self.x_hat = self.x_new + np.random.normal(scale=self.RM.RON_e,size=self.x_hat.size)
                for i in range(1,self.x_hat.size):
                    if self.x_hat[i] < self.x_hat[i-1]:
                        self.x_hat[i] = self.x_hat[i-1]
                self.x_hat = self.x_hat+np.mean(self.RM.noisy_counts*self.RM.gain)-np.mean(self.x_hat)


                if attempts >= conv_attempts:
                    success = True
                    self.x_new = np.round(self.minimize_res['x']) + np.random.normal(scale=self.RM.RON_e,size=self.x_new.size)
                    for i in range(1,self.x_new.size):
                        if self.x_new[i] < self.x_new[i-1]:
                            self.x_new[i] = self.x_new[i-1]
                    self.x_new = self.x_new + np.mean(self.RM.noisy_counts*self.RM.gain) - np.mean(self.x_new)
                    self.x_hat = np.copy(self.x_new)
        
        electron_rates = (self.x_new[1:]-self.x_new[:-1])/self.dt[1:]
        
#        print(electron_rates)
        covmat_h    = self.covmat[np.ix_(self.good_intervals,self.good_intervals)]
        invcovmat_h = np.linalg.inv(covmat_h)
        self.sigmasq_h   = 1./np.sum(invcovmat_h)
        self.mean_electron_rate = self.sigmasq_h*np.sum(np.matmul(invcovmat_h,electron_rates[self.good_intervals])) 
#        print(self.good_intervals)
#        print(self.sigmasq_h)
#        print(self.mean_electron_rate)
        sys.stdout.flush()

        for i in range(self.RM.RTS.ngroups):
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
            '''
            
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

        self.covmat = np.diag(np.square(self.stddev/self.dt[1:]))
        for k in range(self.RM.RTS.ngroups-2):
            self.covmat[k,k+1] = self.covmat[k+1,k] =  (self.mean_electron_rate * self.RM.RTS.group_times[k+1] * (1.-1./self.RM.RTS.nframes)
                                                        -2*self.mean_electron_rate/np.square(self.RM.RTS.nframes)*self.RM.RTS.lower_triangle_sum[k+1]
                                                        )/self.dt[k+1]/self.dt[k+2]


        check_CRs  = 1
        self.crloops_counter = 0

        while check_CRs:
            self.crloops_counter = self.crloops_counter + 1

            if (np.any(self.good_intervals) ==  True):
                check_conv = 1
                self.counter    = 0
                self.error      = 0
            else:
                self.counter    = 0
                self.error      = 2
                return self.error, self.counter, self.good_intervals, self.crloops_counter

            while check_conv:

                self.one_iteration()
                self.counter = self.counter+1
                if (np.fabs(self.mean_electron_rate-old_mean_electron_rate) < thr) & (self.mean_electron_rate > 0.):
                    check_conv = 0
                if (self.counter > maxiter):
                    self.error = 1
                    return self.error, self.counter, self.good_intervals, self.crloops_counter
                
                old_mean_electron_rate = self.mean_electron_rate
                

            #test here for CR presence
            self.var_signal_per_diff = (self.RM.RTS.group_times[1:]/self.RM.RTS.nframes
                            + self.RM.RTS.group_times[:-1]*(1./self.RM.RTS.nframes -2)
                            + 2./np.square(self.RM.RTS.nframes)*self.triangle_sums[1:]
                            ) * self.mean_electron_rate
            self.stddev = np.sqrt(self.var_signal_per_diff+self.var_RON_per_diff+self.var_quant_per_diff)
            deltas  = self.RM.gain*(self.RM.noisy_counts[1:]-self.RM.noisy_counts[:-1]) - self.mean_electron_rate*self.dt[1:]
            new_good_intervals = np.fabs(deltas/self.stddev) < CRthr
            self.covmat = np.diag(np.square(self.stddev/self.dt[1:]))
            for k in range(self.RM.RTS.ngroups-2):
                self.covmat[k,k+1] = self.covmat[k+1,k] =  (self.mean_electron_rate * self.RM.RTS.group_times[k+1] * (1.-1./self.RM.RTS.nframes)
                                                            -2*self.mean_electron_rate/np.square(self.RM.RTS.nframes)*self.RM.RTS.lower_triangle_sum[k+1]
                                                            )/self.dt[k+1]/self.dt[k+2]

            if np.array_equal(self.good_intervals,new_good_intervals):
                check_CRs = 0
            else: 
                self.good_intervals = new_good_intervals
                electron_rates = (self.x_new[1:]-self.x_new[:-1])/self.dt[1:]
                
                covmat_h    = self.covmat[np.ix_(self.good_intervals,self.good_intervals)]
                invcovmat_h = np.linalg.inv(covmat_h)
                self.sigmasq_h   = 1./np.sum(invcovmat_h)
                self.mean_electron_rate = self.sigmasq_h*np.sum(np.matmul(invcovmat_h,electron_rates[self.good_intervals])) 
                
            if (self.crloops_counter > maxCRiter):
                self.error = 3
                return self.error, self.counter, self.good_intervals, self.crloops_counter

            
        return self.error, self.counter, self.good_intervals, self.crloops_counter




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

        if np.sum(self.good_intervals) < 1:
            self.gof_stat = -np.inf
            self.gof_pval = 0.

        else:
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

            elif mode == 'Squared-deviations-nocov':
                var_signal_per_diff = (self.RM.RTS.group_times[1:]/self.RM.RTS.nframes
                                       + self.RM.RTS.group_times[:-1]*(1./self.RM.RTS.nframes -2)
                                       + 2./np.square(self.RM.RTS.nframes)*self.triangle_sums[1:]
                                       ) * self.mean_electron_rate
            
                variance = var_signal_per_diff[self.good_intervals]+self.var_RON_per_diff+self.var_quant_per_diff
                variance = variance / np.square(self.RM.gain)
                dof   = np.sum(self.good_intervals) - 1            
                g = np.sum(np.square(f_obs-f_exp)/variance)
                p = chi2.sf(g,dof)      
            
            elif mode == 'Squared-deviations':
                var_signal_per_diff = (self.RM.RTS.group_times[1:]/self.RM.RTS.nframes
                                       + self.RM.RTS.group_times[:-1]*(1./self.RM.RTS.nframes -2)
                                       + 2./np.square(self.RM.RTS.nframes)*self.triangle_sums[1:]
                                       ) * self.mean_electron_rate
            
                variance = var_signal_per_diff+self.var_RON_per_diff+self.var_quant_per_diff
                covmat = np.diag(variance)
                for k in range(self.stddev.size-1):
                    for l in range(k+1,self.stddev.size):
                        covmat[k,l] = covmat[l,k] =  (self.mean_electron_rate * self.RM.RTS.group_times[1+k] * (1.-1./self.RM.RTS.nframes)
                                                      -2*self.mean_electron_rate/np.square(self.RM.RTS.nframes)*self.RM.RTS.lower_triangle_sum[1+k]
                                                      -np.square(self.RM.RON_e)/self.RM.RTS.nframes
                                                      -1./12.*np.square(self.RM.gain*self.RM.RTS.nframes)
                                                      )
                                                            
                covmat    = covmat/np.square(self.RM.gain)
                covmat    = covmat[np.ix_(self.good_intervals,self.good_intervals)]
                invcovmat = np.linalg.inv(covmat)
                dof       = np.sum(self.good_intervals) - 1            
#            x0 = self.dt[1:][self.good_intervals]
#            H_p1 = 1/np.matmul(x0,np.matmul(invcovmat,x0))
#            H_p2 = np.matmul(x0,invcovmat)
#            H    = np.outer(x0,H_p1*H_p2)
#            I    = np.diag(np.ones(H.shape[0]))
#            M1 = I-H
#            M2 = M1.T
#            dof = np.trace(np.matmul(M2,M1))           
                g = np.matmul((f_obs-f_exp),np.matmul(invcovmat,(f_obs-f_exp)))
                p = chi2.sf(g,dof) 
            
            elif mode == 'poisson-likelihood':

                poisson_lpmf = np.empty_like(self.dt,dtype=np.float_)
                for i in range(self.RM.RTS.ngroups):
                    if i == 0:
                        poisson_lpmf[i] = 0.
                    else:
                        if self.x_new[i] < self.x_new[i-1]:
                            poisson_lpmf[i] = np.inf
                        else:
                            poisson_lpmf[i] = self.poisson_distr[i].logpmf(self.x_new[i]-self.x_new[i-1])
            
                g = np.sum(poisson_lpmf[1:][self.good_intervals])
                ncompare = 10000
                lpmfs = np.empty([ncompare,np.sum(self.good_intervals)])
             
                i = 0
                for k in np.nonzero(self.good_intervals)[0]:
                    rv = self.poisson_distr[k+1].rvs(size=ncompare)
                    lpmfs[:,i] = self.poisson_distr[k+1].logpmf(rv) 
                    i = i+1
             
                loglik_compare = np.sum(lpmfs,axis=1)
                BM = g > loglik_compare
                p = np.sum(BM).astype(np.float_)/ncompare

            elif mode == 'full-likelihood':

                poisson_lpmf = np.empty_like(self.dt,dtype=np.float_)
                gaussian_lpdf = np.empty_like(self.dt,dtype=np.float_)

                for i in range(self.RM.RTS.ngroups):
                    if i == 0:
                        poisson_lpmf[i] = 0.
                        gaussian_lpdf[i] = 0.
                    else:
                        if self.x_new[i] < self.x_new[i-1]:
                            poisson_lpmf[i] = -np.inf
                            gaussian_lpdf[i] = -np.inf
                           
                        else:
                            poisson_lpmf[i] = self.poisson_distr[i].logpmf(np.round(self.x_new[i]-self.x_new[i-1])) #Mario:rounded version
                            gauss_distr = norm(loc=self.RM.gain*(self.RM.noisy_counts[i]-self.RM.noisy_counts[i-1]),
                                           scale=np.sqrt(2)*self.RM.RON_e)
                            gaussian_lpdf[i] = gauss_distr.logpdf(self.x_new[i]-self.x_new[i-1])


                g = np.sum((poisson_lpmf+gaussian_lpdf)[1:][self.good_intervals])
                ncompare = 10000
                llkls = np.empty([ncompare,np.sum(self.good_intervals)])

                i = 0
                nh = norm(loc=0.,scale=np.sqrt(2)*self.RM.RON_e)
#                print(self.good_intervals)
#                print(self.error)
#                print(self.mean_electron_rate)
#                print(self.dt[i])
#                sys.stdout.flush()
                for k in np.nonzero(self.good_intervals)[0]:
#                    print('k',k)
                    sys.stdout.flush()
                    rv = self.poisson_distr[k+1].rvs(size=ncompare)
                    noise= nh.rvs(size=ncompare)                
                    llkls[:,i] = self.poisson_distr[k+1].logpmf(rv) + nh.logpdf(noise)
                    i = i+1

                loglik_compare = np.sum(llkls,axis=1)
                BM = g > loglik_compare
                p = np.sum(BM).astype(np.float_)/ncompare


            else:
                print('Goodness of fit test type not supported')
                assert False

            self.gof_stat = g
            self.gof_pval = p

    def test_plot(self):
        '''
        Method to plot the fit results
        '''
        f,ax = plt.subplots(1,2,figsize=(12,4),sharex='row')
        ax[0].scatter(self.RM.RTS.group_times,self.RM.noisy_counts,label='Noisy Counts',s=100,marker='*')
        ax[0].scatter(self.RM.RTS.group_times,self.x_new/self.RM.gain,label='Convergence counts',s=25)
        ax[0].scatter(self.RM.RTS.group_times,self.RM.noisy_counts-self.RM.RON_effective/self.RM.gain,label='Noiseless Counts + \n Bias + KTC + CRs',s=25)
        ax[0].scatter(self.RM.RTS.group_times,self.RM.noisy_counts-(self.RM.RON_effective-np.mean(self.RM.RON_effective))/self.RM.gain,label='Noiseless Counts + \n Bias + KTC +\n mean RON',s=25)
        
        ax[0].plot(self.RM.RTS.group_times,(self.x_new[0]+self.mean_electron_rate*(self.RM.RTS.group_times-self.RM.RTS.group_times[0]))/self.RM.gain)

        for j,gi in enumerate(self.good_intervals):
            if ~gi:
                ax[0].axvline(0.5*(self.RM.RTS.group_times[j]+self.RM.RTS.group_times[j+1]),color='#bbbbbb',linestyle='--')
            
        ax[0].legend()
        ax[0].set_xlabel('Time [s]')
        ax[0].set_ylabel('Counts')
        ax[0].set_title('Cumulative counts')

        mt = 0.5*(self.RM.RTS.group_times[1:]+self.RM.RTS.group_times[:-1])
        dt = self.RM.RTS.group_times[1:]-self.RM.RTS.group_times[:-1]

        y = self.RM.noisy_counts
        y = (y[1:]-y[:-1])/dt
        ax[1].scatter(mt,y,label='Noisy Counts',s=100,marker='*')
        
        y = self.x_new/self.RM.gain
        y = (y[1:]-y[:-1])/dt
        ax[1].scatter(mt,y,label='Convergence counts',s=25)
        
        y = self.RM.noisy_counts-self.RM.RON_effective/self.RM.gain
        y = (y[1:]-y[:-1])/dt
        ax[1].scatter(mt,y,label='Noiseless Counts + \n Bias + KTC + CRs',s=25)
        
        y = self.RM.noisy_counts-(self.RM.RON_effective-np.mean(self.RM.RON_effective))/self.RM.gain
        y = (y[1:]-y[:-1])/dt        
        ax[1].scatter(mt,y,label='Noiseless Counts + \n Bias + KTC +\n mean RON',s=25)
        
        ax[1].plot(mt,self.mean_electron_rate*np.ones_like(mt)/self.RM.gain)
        ax[1].errorbar(mt,self.mean_electron_rate*np.ones_like(mt)/self.RM.gain,yerr=self.stddev/self.RM.gain/dt)

        for j,gi in enumerate(self.good_intervals):
            if ~gi:
                ax[1].axvline(mt[j],color='#bbbbbb',linestyle='--')
            
        ax[1].legend()
        ax[1].set_xlabel('Time [s]')
        ax[1].set_ylabel('Count rate')

        ax[1].set_title('Differential counts')



        f.tight_layout()


