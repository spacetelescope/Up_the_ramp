import numpy as np
import matplotlib.pyplot as plt

class RampTimeSeq(object):

    '''
    A class to describe a basic infrared detector ramp time sequence, 
    for an individual pixel in a generic IR detector working in MultiAccum mode.
    The ramp is parametrized by the number of groups, number of averaged reads and skipped frames per group, as well as the timing of each of the reads.
    This can be used to describe WFC3/IR or JWST detectors ramps, and some defaults are set for the avaialble read sequences on WFC3/IR.

    :detector:
        a string. Supported values are 'GENERIC', 'HST/WFC3/IR', 'JWST/NIRCam' 
        If detector == 'HST/WFC3/IR' or 'JWST/NIRCam' some parameters are set to defaults values among the available read sequences

    :ngroups:
        a np.int_
        Number of actual measurements returned by the detector for each ramp. ngroups includes the 0-th read,
        thus it is always greater than 1

    :nframes:
        a np.int_
        Number of individual consecutive reads averaged together to obtain a single group

    :nskips:
        a np.int_
        Number of consecutive reads to skip between each group

    :read_times:
        an array of np.float_ type
        the length of this array is checked and should be equal to (nframes+nskips)xngroups

    :samp_seq:
        a string
        For detector == 'WFC3/IR' or 'JWST/NIRCam' these are prefdefined read sequences,
        for which only the number of groups needs to be specified

    '''
    
    def __init__(self,detector, ngroups, nframes=None, nskips=None, read_times=None, samp_seq=None):

        self.detector   = detector
        self.ngroups    = ngroups

        if detector == 'GENERIC':
            '''
            Some sanity checks
            '''
            if nframes is None:
                print('nframes needs to be specified for a GENERIC detector')
                assert False
            if nskips is None:
                print('nskips needs to be specified for a GENERIC detector')
                assert False
            if read_times is None:
                print('read_times needs to be specified for a GENERIC detector')
                assert False
            if len(read_times) != ngroups*(nframes+nskips):
                print('len(read_times) should be equal to (nframes+nskips) x ngroups')
                assert False

            self.nframes    = nframes
            self.nskips     = nskips
            self.read_times = read_times
            
        elif detector == 'HST/WFC3/IR':
            self.nframes    = 1
            self.nskips     = 0

            if samp_seq == 'RAPID' :
                read_times = np.array([   0.0000,   2.9320,   5.8650,   8.7970,  11.7290,  14.6610,  17.5940,  20.5260,  23.4580,  26.3910,  29.3230,  32.2550,  35.1870,  38.1200,  41.0520,  43.9840], dtype=np.float_)
            if samp_seq == 'SPARS5' :
                read_times = np.array([   0.0000,   2.9320,   7.9330,  12.9340,  17.9350,  22.9350,  27.9360,  32.9370,  37.9380,  42.9380,  47.9390,  52.9400,  57.9410,  62.9420,  67.9420,  72.9430], dtype=np.float_)
            if samp_seq == 'SPARS10' :
                read_times = np.array([   0.0000,   2.9320,  12.9330,  22.9340,  32.9350,  42.9360,  52.9370,  62.9380,  72.9390,  82.9400,  92.9410, 102.9420, 112.9430, 122.9440, 132.9450, 142.9460], dtype=np.float_)
            if samp_seq == 'SPARS25' :
                read_times = np.array([   0.0000,   2.9320,  27.9330,  52.9330,  77.9340, 102.9340, 127.9350, 152.9350, 177.9360, 202.9360, 227.9370, 252.9370, 277.9380, 302.9380, 327.9390, 352.9400], dtype=np.float_)
            if samp_seq == 'SPARS50' :
                read_times = np.array([   0.0000,   2.9320,  52.9330, 102.9330, 152.9340, 202.9340, 252.9350, 302.9350, 352.9350, 402.9360, 452.9360, 502.9370, 552.9370, 602.9380, 652.9380, 702.9390], dtype=np.float_)
            if samp_seq == 'SPARS100' :
                read_times = np.array([   0.0000,   2.9320, 102.9330, 202.9330, 302.9330, 402.9340, 502.9340, 602.9340, 702.9350, 802.9350, 902.9350,1002.9360,1102.9360,1202.9360,1302.9360,1402.9370], dtype=np.float_)
            if samp_seq == 'SPARS200' :
                read_times = np.array([   0.0000,   2.9320, 202.9320, 402.9320, 602.9320, 802.9330,1002.9330,1202.9330,1402.9330,1602.9330,1802.9330,2002.9330,2202.9330,2402.9330,2602.9330,2802.9330], dtype=np.float_)
            if samp_seq == 'STEP25' :
                read_times = np.array([   0.0000,   2.9320,   5.8650,   8.7970,  11.7290,  24.2300,  49.2300,  74.2310,  99.2310, 124.2320, 149.2320, 174.2330, 199.2330, 224.2340, 249.2340, 274.2350], dtype=np.float_)
            if samp_seq == 'STEP50' :
                read_times = np.array([   0.0000,   2.9320,   5.8650,   8.7970,  11.7290,  24.2300,  49.2300,  99.2310, 149.2310, 199.2320, 249.2320, 299.2320, 349.2330, 399.2330, 449.2340, 499.2340], dtype=np.float_)
            if samp_seq == 'STEP100' :
                read_times = np.array([   0.0000,   2.9320,   5.8650,   8.7970,  11.7290,  24.2300,  49.2300,  99.2310, 199.2310, 299.2310, 399.2320, 499.2320, 599.2320, 699.2330, 799.2330, 899.2330], dtype=np.float_)
            if samp_seq == 'STEP200' :
                read_times = np.array([   0.0000,   2.9320,   5.8650,   8.7970,  11.7290,  24.2300,  49.2300,  99.2310, 199.2310, 399.2310, 599.2310, 799.2310, 999.2310,1199.2310,1399.2310,1599.2310], dtype=np.float_)
            if samp_seq == 'STEP400' :
                read_times = np.array([   0.0000,   2.9320,   5.8650,   8.7970,  11.7290,  24.2300,  49.2300,  99.2310, 199.2310, 399.2310, 799.2320,1199.2320,1599.2330,1999.2330,2399.2340,2799.2350], dtype=np.float_)

            self.read_times = read_times[:self.ngroups]

            
        elif detector == 'JWST/NIRCam':
            print(" 'JWST/NIRCam' not implemented yet ")
            assert False

        else:
            print("Allowed values for the detector parameter are  'GENERIC', 'HST/WFC3/IR', 'JWST/NIRCam' ")
            assert False

        '''
        Define the group times as the average of the read times of the kept frames for that group and define, lt, the sum of 
        the lower-triangular matrix formed by the kept-reads times for that group. This term appears as (proportional to) the double
        sum in eq (4) of Robberto et al. (2010): JWST-STScI-002161, and is important for calculating the noise of averaged groups
        '''

        self.group_times = np.empty(self.ngroups)
        self.lower_triangle_sum = np.empty(self.ngroups)
        
        self.kept_reads = np.zeros_like(self.read_times,dtype=np.bool_)
        for i in range(self.ngroups):
            tt = self.read_times[i*(self.nframes+self.nskips):i*(self.nframes+self.nskips)+self.nframes]
            self.kept_reads[i*(self.nframes+self.nskips):i*(self.nframes+self.nskips)+self.nframes] = True
            self.group_times[i] = np.mean(tt)
            self.lower_triangle_sum[i] = np.sum(np.tril(np.tile(tt,(tt.size,1)),k=-1))


    def test_plot(self):
        '''
        Method to plot the read sequence
        '''
        f,ax = plt.subplots(1,1,figsize=(10,4))
        ax.scatter(self.read_times[self.kept_reads],np.ones_like(self.read_times[self.kept_reads]),c='b',s=4,label='Individual frames read times')
        ax.scatter(self.read_times[~self.kept_reads],np.ones_like(self.read_times[~self.kept_reads]), c='r',s=4,label='Individual skipped frames times')
        ax.scatter(self.group_times,1.1*np.ones_like(self.group_times),c='black',marker='x',s=30,label='Group average times')
        ax.set_xlabel('Time [s]')
        ax.set_ylim(0.8,1.3)
        ax.legend()
        f.tight_layout()




class RampMeasurement(object):

    '''
    A class to describe a full ramp measurement sequence (in counts), with errors on individual reads,
    average of frames
    
    :RTS:
        an instance of the RampTimeSeq class.

    :flux:
        a np.float_
        the flux in e-/s

    :gain:
        a np.float_
        The detector gain in e-/ADU

    :RON:
        a np.float_
        the standard deviation of read out noise for individual frames in e-

    :KTC:
        a np.float
        the standard deviation of the KTC noise for the 0th frame in e-

    :bias:
        a np.int
        the mean of the bias level in adu
        

    '''
    def __init__(self,RTS,flux,gain=None,RON=None,KTC=None,bias=None,full_well=None,CRdict=None):

        self.RTS = RTS
        self.flux = 1.*flux
        self.CRdict = CRdict

        if self.RTS.detector == 'GENERIC':
            '''
            Some sanity checks
            '''
            if gain is None:
                print('gain (e/adu) needs to be specified for a GENERIC detector')
                assert False
            if RON is None:
                print('the RON (electrons) needs to be specified for a GENERIC detector')
                assert False
            if KTC is None:
                print('the KTC noise (electrons) needs to be specified for a GENERIC detector')
                assert False
            if bias is None:
                print('The bias (counts) needs to be specified for a GENERIC detector ')
                assert False
            if full_well is None:
                print('The full well capacity (electrons) needs to be specified for a GENERIC detector ')
                assert False


            self.gain = gain
            
            self.bias_adu = bias
            self.bias_e   = np.rint(self.bias_adu*self.gain)

            self.KTC_e = KTC
            self.KTC_adu = self.KTC_e/self.gain

            self.RON_e = RON
            self.RON_adu = self.RON_e/self.gain

            self.FWC_e = full_well
            self.FWC_adu = self.FWC_e/self.gain



        
        if self.RTS.detector == 'HST/WFC3/IR':

            '''
            The gain (electrons/adu) comes from version 4.0 of the WFC3/IR DHB and is the avergae of the gain for the 4 amplifiers
            '''
            self.gain = 0.25*(2.252+2.203+2.188+2.265)
            
            
            '''
            This bias value (in counts) is the np.median of the superzero extension ['ZSCI'] in the
            u1k1727mi_lin.fits nlinfit file
            '''
            self.bias_adu = 11075 
            self.bias_e   = np.rint(self.bias_adu*self.gain)

            '''
            This KTC noise (in electrons) comes from Massimo Robberto (priv. comm.) estimate of the
            WFC3 pixel capacity: 0.064 pF, using k_B =1.38e-23, and K = 145 Kelvin (for WFC3).
            With V = sqrt(KTC) the amplitude of the noise in Volts, divided by the charge of the electron (1.602 e-19 Coulomb)
            '''
            
            self.KTC_e = np.sqrt(1.380648e-23*145*0.064e-12)/1.602e-19  
            self.KTC_adu = self.KTC_e/self.gain

            '''
            The RON in electrons comes from Instrument Science Report WFC3 2009-21 where the CDS noise is reported.
            We take the average CDS RON among the quadrants from their table 1 (first row, SMOV)  and divide that by sqrt(2)
            to get the noise in 1 read
            '''

            self.RON_e = 0.25*(20.99+19.80+21.53+21.98) / np.sqrt(2)
            self.RON_adu = self.RON_e/self.gain

            '''
            We use 80000 electrons as the full well for WFC3
            '''

            self.FWC_e = 80000
            self.FWC_adu = self.FWC_e/self.gain


        '''
        Get the effective readnoise from ISR NICMOS 98-008, page 7
        '''
        
        
        dt = np.zeros_like(self.RTS.read_times)
        for i in range(dt.size):
            dt[i] = self.RTS.read_times[i] - self.RTS.read_times[i-1] 
        
        num = np.square(dt[1])+np.square(dt[-1]) + np.sum(np.square(dt[1:-1]-dt[2:]))
        den = 2*np.sum(np.square(dt[1:]))
        self.effRON_e = np.sqrt(num/den)*self.RON_e


        self.generate_electrons()
        self.add_noise()
        self.add_CR_hits()
        self.average_groups()

    def generate_electrons(self):

        '''
        Given the timing sequence and the flux generate a ramp of number of detected electrons.
        These electrons are generated via a random poisson process, thus contain
        the intrinsic noise due to the source photon statistics. Note that the electrons_read attribute
        an array of integers 
        '''
        
        self.electrons_reads = np.zeros_like(self.RTS.read_times,dtype=np.int_)
        
        for i in range(1,self.electrons_reads.size,1):
            delta_t = self.RTS.read_times[i] - self.RTS.read_times[i-1]
            self.electrons_reads[i] = self.electrons_reads[i-1]+np.random.poisson(lam=delta_t*self.flux)

        '''
        For convenience create a noiseless_counts attribute that is purely equal to the electrons
        divided by the gain (no errors, no discretization)
        '''
            
        self.noiseless_counts_reads = self.electrons_reads/self.gain

    def add_noise(self):

        '''
        Add the various sources of noise. This step creates an other convenience variable: noisy_electrons_reads,
        which represent the noisy version of the phot-electrons, and is a float (even though electrons are 
        integers). The final measured variable, the noisy counts is instead an integer as it should be
        '''

        '''
        Add bias as a constant
        '''
        self.noisy_electrons_reads = (self.electrons_reads + self.bias_e).astype(np.float_)

        '''
        Add KTC noise as random gaussian (since KTC is added at reset, it is the same value for all reads)
        '''
        self.KTC_actual = np.random.normal(scale=self.KTC_e)
        self.noisy_electrons_reads = self.noisy_electrons_reads + self.KTC_actual

        '''
        Add read noise as random gaussian (a different random number per read)
        '''
        self.RON_actual_reads = np.random.normal(scale=self.RON_e,size=self.noisy_electrons_reads.size)
        self.noisy_electrons_reads = self.noisy_electrons_reads + self.RON_actual_reads

        '''
        Convert to counts using the gain and rounding to nearest lower integer
        '''
        self.noisy_counts_reads = np.floor(self.noisy_electrons_reads/self.gain).astype(np.int_)


    def add_CR_hits(self):
        '''
        Method to add CR hits to a ramp

        :CRdict:
            a dictionary containing the following keys:

           :'times':
               a list of times when the CR hit

           :'counts':
               the "energy" deposited by each CR (in counts)
        
        '''

        self.cum_CR_counts_reads = np.zeros_like(self.RTS.read_times,dtype=np.int_)
        if self.CRdict is not None:
            for t,c in zip(self.CRdict['times'],self.CRdict['counts']):
                '''First find out which of the ramp frames the CR falls in'''
                msk = self.RTS.read_times >= t
                self.cum_CR_counts_reads[msk] = self.cum_CR_counts_reads[msk]+np.floor(c).astype(np.int_)
                
            self.noisy_counts_reads = self.noisy_counts_reads + self.cum_CR_counts_reads

    def average_groups(self):
        '''
        Method to take the individual reads and average them
        '''

        self.noisy_counts = np.zeros(self.RTS.ngroups,dtype=np.int_)
        self.noiseless_counts = np.zeros(self.RTS.ngroups)
        self.cum_CR_counts = np.zeros(self.RTS.ngroups)
        self.RON_effective = np.zeros(self.RTS.ngroups)

        for i in range(self.RTS.ngroups):

            stuff_to_average = self.noisy_counts_reads[i*(self.RTS.nframes+self.RTS.nskips):i*(self.RTS.nframes+self.RTS.nskips)+self.RTS.nframes]
            self.noisy_counts[i] = np.floor(np.mean(stuff_to_average)).astype(np.int_)

            stuff_to_average = self.noiseless_counts_reads[i*(self.RTS.nframes+self.RTS.nskips):i*(self.RTS.nframes+self.RTS.nskips)+self.RTS.nframes]
            self.noiseless_counts[i] = np.mean(stuff_to_average)

            stuff_to_average = self.cum_CR_counts_reads[i*(self.RTS.nframes+self.RTS.nskips):i*(self.RTS.nframes+self.RTS.nskips)+self.RTS.nframes]
            self.cum_CR_counts[i] = np.mean(stuff_to_average)

            stuff_to_average = self.RON_actual_reads[i*(self.RTS.nframes+self.RTS.nskips):i*(self.RTS.nframes+self.RTS.nskips)+self.RTS.nframes]
            self.RON_effective[i] = np.mean(stuff_to_average)
        
            
    def test_plot(self):
        '''
        Method to plot the simulated counts
        '''

        f,ax = plt.subplots(3,1,figsize=(10,10),sharex='col')
        ax[0].scatter(self.RTS.read_times[self.RTS.kept_reads],self.noiseless_counts_reads[self.RTS.kept_reads],label='Noiseless Counts',s=15)
        ax[0].scatter(self.RTS.read_times[~self.RTS.kept_reads],self.noiseless_counts_reads[~self.RTS.kept_reads],label=None,s=1)
                
        if self.RTS.nframes > 1:
            ax[0].scatter(self.RTS.group_times,self.noiseless_counts,label='Noiseless Counts -- gr. avg.',marker='x',s=100)
        ax[0].legend()

        ax[1].scatter(self.RTS.read_times[self.RTS.kept_reads],self.noiseless_counts_reads[self.RTS.kept_reads]+self.cum_CR_counts_reads[self.RTS.kept_reads]+self.bias_adu+self.KTC_actual/self.gain,
                      label='Noiseless Counts + \n Bias + KTC + CRs',s=15)
        ax[1].scatter(self.RTS.read_times[~self.RTS.kept_reads],self.noiseless_counts_reads[~self.RTS.kept_reads]+self.cum_CR_counts_reads[~self.RTS.kept_reads]+self.bias_adu+self.KTC_actual/self.gain,
                      label=None,s=1)
        if self.RTS.nframes > 1:
            ax[1].scatter(self.RTS.group_times,self.noiseless_counts+self.cum_CR_counts+self.bias_adu+self.KTC_actual/self.gain,
                          label='Noiseless Counts + \n Bias + KTC -- gr. avg.',s=100,marker='x')

        ax[1].scatter(self.RTS.read_times[self.RTS.kept_reads],self.noisy_counts_reads[self.RTS.kept_reads],label='Noisy Counts',s=15)
        ax[1].scatter(self.RTS.read_times[~self.RTS.kept_reads],self.noisy_counts_reads[~self.RTS.kept_reads],label=None,s=1)
        if self.RTS.nframes > 1:
            ax[1].scatter(self.RTS.group_times,self.noisy_counts,label='Noisy Counts -- gr. avg.',s=100,marker='x')
        ax[1].legend()

        ydw,yup = ax[1].get_ylim()

        if self.CRdict is not None:
            cmax = 0.
            for t,c in zip(self.CRdict['times'],self.CRdict['counts']):
                ax[1].plot([t,t],[ydw,ydw-c],label=None,c='#555555')
                if c > cmax:
                    cmax = c
                
            ax[1].set_ylim(ydw-1.1*cmax,yup)
                
        ax[2].scatter(self.RTS.read_times[self.RTS.kept_reads],self.RON_actual_reads[self.RTS.kept_reads]/self.gain,label='RON (counts)',s=15)
        ax[2].scatter(self.RTS.read_times[~self.RTS.kept_reads],self.RON_actual_reads[~self.RTS.kept_reads]/self.gain,label=None,s=1)
        if self.RTS.nframes > 1:
            ax[2].scatter(self.RTS.group_times,self.RON_effective/self.gain,label='RON (counts) -- gr.avg.',s=100,marker='x')
        ax[2].set_xlabel('Time [s]')
        ax[2].set_ylim(-3*self.RON_adu,3*self.RON_adu)
        ax[2].legend()

        f.tight_layout()


    def add_background(self,extra_bg):

        '''
        Method to add extra background at the single read level.
        It modifies the "noisy_electrons_reads" attribute and then
        adds the CRhits and perfoms the group-average and conversion to counts/s.
        
        :extra_bg:
            a np.array of integers of the same size as RTS.read_times.
            It contains the cumulative electrons from the extra background source
        '''
        
        if extra_bg.dtype != np.int_:
            print('The extra background array must be of np.int type')
            assert False
        
        if (extra_bg.size != self.RTS.read_times.size):
            print('The extra background array must be of the same length as the RTS.read_times array')
            assert False
        else:
            self.electrons_reads = self.electrons_reads + extra_bg
            self.noiseless_counts_reads = self.electrons_reads/self.gain
            self.noisy_electrons_reads = self.noisy_electrons_reads + extra_bg
            self.noisy_counts_reads = np.floor(self.noisy_electrons_reads/self.gain).astype(np.int_)
            
        self.add_CR_hits()
        self.average_groups()
        
        
        
        