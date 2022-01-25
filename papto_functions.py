import sys
import mne
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
import matplotlib.pyplot as plt
from fooof import FOOOF
from fooof import FOOOFGroup




def find_papto_bursts(s, Fs):


	#################################################################
	#Inputs:
	#s = timecourse = numpy with shape (# of timepoints,)
	#Fs = sampling frequency in Hz 
	#################################################################

	s = np.expand_dims(s, axis=0)

	# frequencies for spectral domain analysis 
	fmin = 0.25
	fmax = 80
	fstep = 0.25
	fVec = np.arange(fmin, fmax, fstep)


	#################################################################
	# [1] compute TFR via Morlet wavelet convolution
	# technique based on: Tallon-Baudry et al., J. Neuroscience, 1997
	# code adapted from: Shin et al, eLife, 2017
	##################################################################
	
	TFR, tVec = TFR_via_morlet_wavelet(s, Fs, fVec)


	###################################################################
	# [2] FOOOF modeling to obtain offset and exponent 
	# see github.com/fooof-tools/fooof
	####################################################################

	# PSD via welch method
	PSD_fVec, PSD = signal.welch(s, fs=Fs, nfft=1000, noverlap=900, nperseg=1000)

	# get rid of f = 0 Hz in PSD for fooof fitting
	PSD_fVec = np.delete(PSD_fVec, 0)
	PSD = np.delete(PSD, 0)

	# notch filter to remove power line noise is appropriate here 

	# FOOOF modeling
	fm = FOOOF(peak_width_limits=(2,10), max_n_peaks=4, aperiodic_mode='fixed', min_peak_height=0.05, peak_threshold=1.5)
	fm.fit(PSD_fVec, np.squeeze(PSD), [fmin,fmax])
	exponent = fm.get_params('aperiodic_params', 'exponent')
	offset = fm.get_params('aperiodic_params', 'offset')


	###################################################################
	# [3] calculate TFR normalization factor and apply it to TFR
	####################################################################

	# generate normalization factor n_c from aperiodic offset and exponent
	nc = (fVec**(exponent)) / (10**offset)

	# normalize the TFR
	TFR = TFR * nc[None,:,None] 


	###################################################################
	# [4] find events from normalized TFR
	####################################################################

	eventsList = get_spectral_events(TFR, Fs, fVec, tVec, fstep)

	# take only supra-threshold events
	eventsList = eventsList[eventsList['Peak Power'] >= 8]
	
	return eventsList, fm




def get_spectral_events(TFR, Fs, fVec, tVec,fstep):

	########################################################
	# Adapted from Shin et al., eLife, 2017
	########################################################

    # Find transient spectral events based on TFR
    findMethod = 1
    thrFOM = 1 # This is irrelevant for this script b/c we dont take outlier events
    numTrials = TFR.shape[0] 
    classLabels = [1 for x in range(numTrials)]
    neighbourhood_size = (4,160)
    threshold = 0 # i.e., no thresholding setting 
    spectralEvents = spectralevents_find(findMethod, thrFOM, tVec,
                fVec, TFR, classLabels, neighbourhood_size, threshold, Fs)
    df = pd.DataFrame(spectralEvents)

    #modify burst propterties
    allEvents = df.copy()
    allEvents['periods'] = allEvents['Event Duration']*allEvents['Peak Frequency']
    allEvents['Frequency Span'] = allEvents['Frequency Span']*fstep 

    # very high power (>50) events are erroneous
    allEvents = allEvents[allEvents['Peak Power']<=50]
    

    allEvents = allEvents.drop(['Trial', 'Hit/Miss', 'Normalized Peak Power', 'Outlier Event'], axis=1)

    return allEvents





def energyvec(f,s,Fs,width):
    
	########################################################
	# Adapted from Shin et al., eLife, 2017
	########################################################

    # Return a vector containing the energy as a
    # function of time for frequency f. The energy
    # is calculated using Morlet's wavelets. 
    # s : signal
    # Fs: sampling frequency
    # width : width of Morlet wavelet (>= 5 suggested).


    dt = 1/Fs
    sf = f/width
    st = 1/(2 * np.pi * sf)

    t= np.arange(-3.5*st, 3.5*st, dt)
    m = morlet(f, t, width)

    y = np.convolve(s, m)
    y = (dt * np.abs(y))**2
    lowerLimit = int(np.ceil(len(m)/2))
    upperLimit = int(len(y)-np.floor(len(m)/2)+1)
    y = y[lowerLimit:upperLimit]

    return y

def morlet(f,t,width):

	########################################################
	# Adapted from Shin et al., eLife, 2017
	########################################################

    # Morlet's wavelet for frequency f and time t. 
    # The wavelet will be normalized so the total energy is 1.
    # width defines the ``width'' of the wavelet. 
    # A value >= 5 is suggested.
    #
    # Ref: Tallon-Baudry et al., J. Neurosci. 15, 722-734 (1997)

    sf = f/width
    st = 1/(2 * np.pi * sf)
    A = 1/np.sqrt((st/2 * np.sqrt(np.pi))) 
    y = A * np.exp(-t**2 / (2 * st**2)) * np.exp(1j * 2 * np.pi * f * t)

    return y



def TFR_via_morlet_wavelet(s, Fs, fVec):

	#####################################################################
	# Adapted from spectralevents_ts2tfr function from Shin et al (2017)
	#####################################################################

	# width of morlet wavelet 
	width = 10

	# obtain time vector (tVec) from timecourse (tVec starting with t=0s)
	numSamples = s.shape[1] 
	tVec = np.arange(numSamples)/Fs

	# find number of frequencies for convolution
	numFrequencies = len(fVec)

	# generate TFR row by row
	TFR = []
	B = np.zeros((numFrequencies, numSamples))
	# Frequency loop
	for j in np.arange(numFrequencies):
		B[j,:] = energyvec(fVec[j], signal.detrend(s[0,:]), Fs, width)
	TFR.append(B)

	return TFR, tVec




def fwhm_lower_upper_bound1(vec, peakInd, peakValue):

	########################################################
	# Adapted from Shin et al., eLife, 2017
	########################################################

    # Function to find the lower and upper indices within which the vector is less than the FWHM
    #   with some rather complicated boundary rules (Shin, eLife, 2017)
    halfMax = peakValue/2

    # Extract data before the peak only (data should be rising at the end of the new array)
    vec1 = vec[0:peakInd]
    # Find indices less than half the max
    vec1_underThreshold = np.where(vec1<halfMax)[0]
    if len(vec1_underThreshold)==0:
        # There are no indices less than half the max, so we have to estimate the lower edge
        estimateLowerEdge = True
    else:
        # There are indices less than half the max, take the last one under halfMax as the lower edge
        estimateLowerEdge = False
        lowerEdgeIndex = vec1_underThreshold[-1]

    # Extract data following the peak only (data should be falling at the start of the new array)
    vec2 = vec[peakInd:]
    # Find indices less than half the max
    vec2_underThreshold = np.where(vec2<halfMax)[0]
    if len(vec2_underThreshold)==0:
        # There are no indices less than half the max, so we have to estimate the upper edge
        estimateUpperEdge = True
    else:
        # There are indices less than half the max, take the first one under halfMax as the upper edge
        estimateUpperEdge = False
        upperEdgeIndex = vec2_underThreshold[0] + len(vec1)

    if not estimateLowerEdge:
        if not estimateUpperEdge:
            # FWHM fits in the range, so pick off the edges of the FWHM
            lowerInd = lowerEdgeIndex
            upperInd = upperEdgeIndex
            FWHM = upperInd - lowerInd
        if estimateUpperEdge:
            # FWHM fits in on the low end, but hits the edge on the high end
            lowerInd = lowerEdgeIndex
            upperInd = len(vec)-1
            FWHM = 2 * (peakInd - lowerInd + 1)
    else:
        if not estimateUpperEdge:
            # FWHM hits the edge on the low end, but fits on the high end
            lowerInd = 0
            upperInd = upperEdgeIndex
            FWHM = 2 * (upperInd - peakInd + 1)
        if estimateUpperEdge:
            # FWHM hits the edge on the low end and the high end
            lowerInd = 0
            upperInd = len(vec)-1
            FWHM = 2*len(vec)

    return lowerInd, upperInd, FWHM


def find_localmax_method_1(TFR, fVec, tVec, eventThresholdByFrequency, classLabels, medianPower, neighbourhood_size,  threshold, Fs):
    
    # 1st event-finding method (primary event detection method in Shin et
    # al. eLife 2017): Find spectral events by first retrieving all local
    # maxima in un-normalized TFR using imregionalmax, then selecting
    # suprathreshold peaks within the frequency band of interest. This
    # method allows for multiple, overlapping events to occur in a given
    # suprathreshold region and does not guarantee the presence of
    # within-band, suprathreshold activity in any given trial will render
    # an event.

    # spectralEvents: 12 column matrix for storing local max event metrics:
    #        trial index,            hit/miss,         maxima frequency,
    #        lowerbound frequency,     upperbound frequency,
    #        frequency span,         maxima timing,     event onset timing,
    #        event offset timing,     event duration, maxima power,
    #        maxima/median power
    # Number of elements in discrete frequency spectrum
    flength = TFR.shape[1]
    # Number of point in time
    tlength = TFR.shape[2]
    # Number of trials
    numTrials = TFR.shape[0]

    spectralEvents = []

    # Retrieve all local maxima in TFR using python equivalent of imregionalmax
    for ti in range(numTrials):

        # Get TFR data for this trial [frequency x time]
        thisTFR = TFR[ti, :, :]

        # Find local maxima in the TFR data
        data = thisTFR
        data_max = filters.maximum_filter(data, neighbourhood_size)
        maxima = (data == data_max)
        data_min = filters.minimum_filter(data, neighbourhood_size)
        diff = ((data_max - data_min) > threshold)
        maxima[diff == 0] = 0
        labeled, num_objects = ndimage.label(maxima)
        xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1)))

        numPeaks = len(xy)

        peakF = []
        peakT = []
        peakPower = []
        for thisXY in xy:
            peakF.append(int(thisXY[0]))
            peakT.append(int(thisXY[1]))
            peakPower.append(thisTFR[peakF[-1], peakT[-1]])

        # Find local maxima lowerbound, upperbound, and full width at half max
        #    for both frequency and time
        Ffwhm = []
        Tfwhm = []
        for lmi in range(numPeaks):
            thisPeakF = peakF[lmi]
            thisPeakT = peakT[lmi]
            thisPeakPower = peakPower[lmi]

            # Indices of TFR frequencies < half max power at the time of a given local peak
            TFRFrequencies = thisTFR[:, thisPeakT]
            lowerInd, upperInd, FWHM = fwhm_lower_upper_bound1(TFRFrequencies,
                                                               thisPeakF, thisPeakPower)
            lowerEdgeFreq = fVec[lowerInd]
            upperEdgeFreq = fVec[upperInd]
            FWHMFreq = FWHM

            # Indices of TFR times < half max power at the frequency of a given local peak
            TFRTimes = thisTFR[thisPeakF, :]
            lowerInd, upperInd, FWHM = fwhm_lower_upper_bound1(TFRTimes,
                                                               thisPeakT, thisPeakPower)
            lowerEdgeTime = tVec[lowerInd]
            upperEdgeTime = tVec[upperInd]
            FWHMTime = FWHM / Fs

            # Put peak characteristics to a dictionary
            #        trial index,            hit/miss,         maxima frequency,
            #        lowerbound frequency,     upperbound frequency,
            #        frequency span,         maxima timing,     event onset timing,
            #        event offset timing,     event duration, maxima power,
            #        maxima/median power
            peakParameters = {
                'Trial': ti,
                'Hit/Miss': classLabels[ti],
                'Peak Frequency': fVec[thisPeakF],
                'Lower Frequency Bound': lowerEdgeFreq,
                'Upper Frequency Bound': upperEdgeFreq,
                'Frequency Span': FWHMFreq,
                'Peak Time': tVec[thisPeakT],
                'Event Onset Time': lowerEdgeTime,
                'Event Offset Time': upperEdgeTime,
                'Event Duration': FWHMTime,
                'Peak Power': thisPeakPower,
                'Normalized Peak Power': thisPeakPower / medianPower[thisPeakF],
                'Outlier Event': thisPeakPower > eventThresholdByFrequency[thisPeakF]
            }

            # Build a list of dictionaries
            spectralEvents.append(peakParameters)

    return spectralEvents




def spectralevents_find (findMethod, thrFOM, tVec, fVec, TFR, classLabels, neighbourhood_size, threshold, Fs):
	
	########################################################
	# Adapted from Shin et al., eLife, 2017
	########################################################   

    # SPECTRALEVENTS_FIND Algorithm for finding and calculating spectral 
    #   events on a trial-by-trial basis of of a single subject/session. Uses 
    #   one of three methods before further analyzing and organizing event 
    #   features:
    #
    #   1) (Primary event detection method in Shin et al. eLife 2017): Find 
    #      spectral events by first retrieving all local maxima in 
    #      un-normalized TFR using imregionalmax, then selecting suprathreshold
    #      peaks within the frequency band of interest. This method allows for 
    #      multiple, overlapping events to occur in a given suprathreshold 
    #      region and does not guarantee the presence of within-band, 
    #      suprathreshold activity in any given trial will render an event.
    #   2) Find spectral events by first thresholding
    #      entire normalize TFR (over all frequencies), then finding local 
    #      maxima. Discard those of lesser magnitude in each suprathreshold 
    #      region, respectively, s.t. only the greatest local maximum in each 
    #      region survives (when more than one local maxima in a region have 
    #      the same greatest value, their respective event timing, freq. 
    #      location, and boundaries at full-width half-max are calculated 
    #      separately and averaged). This method does not allow for overlapping
    #      events to occur in a given suprathreshold region and does not 
    #      guarantee the presence of within-band, suprathreshold activity in 
    #      any given trial will render an event.
    #   3) Find spectral events by first thresholding 
    #      normalized TFR in frequency band of interest, then finding local 
    #      maxima. Discard those of lesser magnitude in each suprathreshold region,
    #      respectively, s.t. only the greatest local maximum in each region
    #      survives (when more than one local maxima in a region have the same 
    #      greatest value, their respective event timing, freq. location, and 
    #      boundaries at full-width half-max are calculated separately and 
    #      averaged). This method does not allow for overlapping events to occur in
    #      a given suprathreshold region and ensures the presence of 
    #      within-band, suprathreshold activity in any given trial will render 
    #      an event.
    #
    # specEv_struct = spectralevents_find(findMethod,eventBand,thrFOM,tVec,fVec,TFR,classLabels)
    # 
    # Inputs:
    #   findMethod - integer value specifying which event-finding method 
    #       function to run. Note that the method specifies how much overlap 
    #       exists between events.
    #   eventBand - range of frequencies ([Fmin_event Fmax_event]; Hz) over 
    #       which above-threshold spectral power events are classified.
    #   thrFOM - factors of median threshold; positive real number used to
    #       threshold local maxima and classify events (see Shin et al. eLife 
    #       2017 for discussion concerning this value).
    #   tVec - time vector (s) over which the time-frequency response (TFR) is 
    #       calcuated.
    #   fVec - frequency vector (Hz) over which the time-frequency response 
    #       (TFR) is calcuated.
    #   TFR - time-frequency response (TFR) (trial-frequency-time) for a
    #       single subject/session.
    #   classLabels - numeric or logical 1-row array of trial classification 
    #       labels; associates each trial of the given subject/session to an 
    #       experimental condition/outcome/state (e.g., hit or miss, detect or 
    #       non-detect, attend-to or attend away).
    #
    # Outputs:
    #   specEv_struct - event feature structure with three main sub-structures:
    #       TrialSummary (trial-level features), Events (individual event 
    #       characteristics), and IEI (inter-event intervals from all trials 
    #       and those associated with only a given class label).
    #
    # See also SPECTRALEVENTS, SPECTRALEVENTS_FIND, SPECTRALEVENTS_TS2TFR, SPECTRALEVENTS_VIS.

    # Initialize general data parameters
    # Number of elements in discrete frequency spectrum
    flength = TFR.shape[1]
    # Number of point in time
    tlength = TFR.shape[2]
    # Number of trials
    numTrials = TFR.shape[0]
    classes = np.unique(classLabels)

    # Median power at each frequency across all trials
    TFRpermute = np.transpose(TFR, [1, 2, 0]) # freq x time x trial
    TFRreshape = np.reshape(TFRpermute, (flength, tlength*numTrials))
    medianPower = np.median(TFRreshape, axis=1)

    # Spectral event threshold for each frequency value
    eventThresholdByFrequency = thrFOM*medianPower

    # Validate consistency of parameter dimensions
    if flength != len(fVec):
        sys.exit('Mismatch in frequency dimensions!')
    if tlength != len(tVec): 
        sys.exit('Mismatch in time dimensions!')
    if numTrials != len(classLabels): 
        sys.exit('Mismatch in number of trials!')

    # Find spectral events using appropriate method
    #    Implementing one for now
    if findMethod == 1:
        spectralEvents = find_localmax_method_1(TFR, fVec, tVec, eventThresholdByFrequency, classLabels, medianPower, neighbourhood_size, threshold, Fs)
    elif findMethod == 2:
        spectralEvents = find_localmax_method_2 # HACK!!!!
    elif findMethod == 3:
        spectralEvents = find_localmax_method_3 # HACK!!!!

    return spectralEvents


















				







			 


