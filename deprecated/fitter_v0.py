import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson, norm, uniform, gamma
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


    def loglikelihood(self,x,index):
        '''
        Function that returns minus the log-likelihood of the measured counts, given a flux
        and the noise characteristics. 
        '''

        xr = np.round(x).astype(np.int_)
        
        gaussian_pdf = norm.pdf(xr,loc=self.RM.noisy_counts[index],scale=self.RM.RON_adu)
        #uniform_pdf  = uniform.pdf(x,loc=self.x_hat[index],scale=1)
        uniform_pdf = 1.
        if index == 0:
            return -1.*np.log(gaussian_pdf*uniform_pdf)
        else:
            #            poisson_pmf  = poisson.pmf(xr-self.x_hat[index-1],mu=self.mean_countrate*self.dt[index])
            poisson_pmf  = poisson.pmf(xr-self.x_new[index-1],mu=self.mean_countrate*self.dt[index])
            return -1.*np.log(gaussian_pdf * poisson_pmf * uniform_pdf)            

    def loglikelihood_all(self,x):
        '''
        Function that returns minus the log-likelihood of the measured counts, given a flux
        and the noise characteristics. Does all reads in one
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

        #for i in range(self.x_hat.size):
        #    self.x_new[i] = np.round(minimize(self.loglikelihood,self.x_hat[i],(i),method='Powell')['x'])

        self.x_new = np.round(minimize(self.loglikelihood_all,self.x_hat,method=self.fitpars['one_iteration_method'])['x'])


        countrates = (self.x_new[1:]-self.x_new[:-1])/self.dt[1:]
        self.mean_countrate = np.mean(countrates[self.good_intervals])
        if self.mean_countrate <= 0.:
            self.mean_countrate = 1e-5
        self.x_hat = self.x_new


    def fitter_loop(self,thr=None,maxiter=50,CRthr=4.):

        if thr is None:
            thr = 1./self.RM.FWC_adu/self.RM.RTS.group_times[-1] 
            
        self.good_intervals = np.ones_like(self.x_hat[1:],dtype=np.bool_)
        stddev = np.sqrt(self.mean_countrate*self.dt[1:]+2*np.square(self.RM.RON_adu))
        self.good_intervals = np.fabs((self.RM.noisy_counts[1:]-self.RM.noisy_counts[:-1] - self.mean_countrate*self.dt[1:])/stddev) < CRthr
        old_mean_countrate = self.mean_countrate

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
                #print(counter, old_mean_countrate)
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

        return error, counter, self.good_intervals, crloops_counter


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


    def MCMC_fitter(self, nwalkers=100, nruns=1000,nburn=500,nthin=25,ncores=1,lnpfun = 'ln_post_CR'):
        '''
        A fitting method based on the same likelihood, but using MCMC
        to find the best fit flux and zero-read counts rather than the iterative minimization
        '''

        if lnpfun == 'ln_post':
            ndim = 2
            startflux = np.max([self.mean_countrate,0.])
            p0arr = np.array([startflux,self.RM.noisy_counts[0]])
            spreadarr = np.array([np.sqrt(np.sum((np.array([startflux,np.square(self.RM.RON_adu)]) ))) , self.RM.RON_adu ])
            pos = [p0arr + spreadarr*np.random.randn(ndim) for j in range(nwalkers)]
            for j in range(nwalkers):
                p0 = pos[j][0] 
                if p0 < 0.:
                    pos[j][0] = np.random.uniform()
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_post, threads=ncores)
        elif lnpfun == 'ln_post_CR':
            ndim = 1
            startflux = np.max([self.mean_countrate,0.])
            p0arr = startflux
            spreadarr = np.sqrt(np.sum((np.array([startflux,np.square(self.RM.RON_adu)]) )))
            pos = [p0arr + spreadarr*np.random.randn(ndim) for j in range(nwalkers)]
            for j in range(nwalkers):
                p0 = pos[j][0] 
                if p0 < 0.:
                    pos[j][0] = np.random.uniform()

            print(startflux)
            print(pos)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_post_CR, threads=ncores)
            


        #run emcee
        ts = time.time()
        for i, result in enumerate(sampler.sample(pos, iterations=nruns)):
            if (i+1) % (nruns//10) == 0:
                print("{0:5.1%}".format(float(i+1) / nruns))

        self.samples = sampler.chain[:, nburn:-1:nthin, :].reshape((-1, ndim))

        print('Elapsed Time',time.time()-ts)


    def ln_post(self,pars):
        '''
        The log-posterior to be used in the MCMC_fitter. This version cannot account for CRs.
        '''

        if np.any(pars< 0):
            return -np.inf
        
        ln_post = 0.
        for i in range(self.x_hat.size):
        
            if i == 0:
                lli =  np.log(norm.pdf(self.RM.noisy_counts[i],loc=pars[1],scale=self.RM.RON_adu))
            else:
                mc  = pars[0] * self.RM.RTS.group_times[i]

                nsig = 2
                mincnt = np.max([0,np.round(mc-nsig*np.sqrt(mc)).astype(np.int_)])
                maxcnt = np.max([mincnt+1,(mc+nsig*np.sqrt(mc)).astype(np.int_)])

                xi = np.arange(mincnt,maxcnt)
                poisson_pmf  = poisson.pmf(xi,mu=mc)/(poisson.cdf(maxcnt,mu=mc)-poisson.cdf(mincnt,mu=mc))
                gaussian_pdf = norm.pdf(xi,loc=self.RM.noisy_counts[i]-self.RM.noisy_counts[0],scale=self.RM.RON_adu)
                lli = np.log(np.sum(gaussian_pdf * poisson_pmf))            

            ln_post = ln_post + lli

        return ln_post

    def ln_post_CR(self,pars):
        '''
        The log-posterior to be used in the MCMC_fitter. This version accounts for CRs.
        '''

        if np.any(pars<= 0):
            return -np.inf
        
        ln_post = 0.
        nsig = 2
        for i in range(1,self.x_hat.size,1):
        
            mc  = pars[0] * (self.RM.RTS.group_times[i]-self.RM.RTS.group_times[i-1])
            sigmaintv = nsig*np.sqrt(mc+np.square(self.RM.RON_adu))

            mincnt = 0      # np.max([0,np.round(mc-sigmaintv).astype(np.int_)])
            maxcnt = 100 #np.max([mincnt+10,np.round(mc+sigmaintv).astype(np.int_)])

            xi = np.arange(mincnt,maxcnt)
            poisson_pmf  = poisson.pmf(xi,mu=mc)/(poisson.cdf(maxcnt,mu=mc)-poisson.cdf(mincnt,mu=mc))
            gaussian_pdf = norm.pdf(xi,loc=self.RM.noisy_counts[i]-self.RM.noisy_counts[i-1],scale=np.sqrt(2)*self.RM.RON_adu)
            lli = np.log(np.sum((gaussian_pdf * poisson_pmf)))            

            ln_post = ln_post + lli

        return ln_post

    def plot_MCMC(self):
        avgs = np.average(self.samples,axis=0)
        print('Mean parameters',avgs)

        plt.style.use('bmh')
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 14
        plt.rcParams['axes.labelsize'] = 15
        plt.rcParams['axes.labelweight'] = 'normal'
        plt.rcParams['xtick.labelsize'] = 13
        plt.rcParams['ytick.labelsize'] = 13
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 11

        
        fig = corner.corner(self.samples, labels=[['Count_rate','C_0'][i] for i in range(self.samples.shape[1])], show_titles=True, quantiles=[0.16,0.50,0.84], figsize=(13,13), title_kwargs={'fontsize':15},
                            truths = [self.RM.flux/self.RM.gain,self.RM.bias_adu+self.RM.KTC_actual/self.RM.gain])

        for ax in fig.axes:
            ax.set_axis_bgcolor('#FFFFFF')
