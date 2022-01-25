import papto_functions as papto
import mne
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":

	##############################
	
	# Inputs:
	#
	# s = (cleaned) timeseires data in shape (#timepoints,)
	#
	# Fs = sampling frequency of s


	# Outputs:
	#
	# eventsList = pandas dataframe of all supra-threshold (Peak Power > 8 * aperiodic activity) PAPTO events with
	#			   peak frequency between 0.25 and 80 Hz. One row per event. Columns are:
	#			
	#			    'Event Duration' : duation of event in seconds (FWHM_time). Time between event onset and offset.
	#				'Event Offset Time' : peak time of event + FWHM_time/2
	#				'Event Onset Time' : peak time of event - FWHM_time/2 
	#				'Frequency Span' : FWHM_freq in Hz
	#				'Lower Frequency Bound' : Peak Frequency - FWHM_freq/2
	#				'Peak Frequency' : Peak frequency of event in Hz
	#				'Peak Power' : Peak power of event in multiples of aperiodic activity 
	#				'Peak Times' : Peak times of event in seconds (relative to start of s)
	#				'Upper Frequency Bound' : Peak Frequency + FWHM_freq/2
	#				'periods' : number of oscillations within the event (Event Duration * Peak Frequency)
	#
	#
	#
	# fooofModel = fooof object. see https://fooof-tools.github.io/fooof/generated/fooof.FOOOF.html#fooof.FOOOF for attributes

	###############################
	

	# load in timeseries data and set sampling frequency
	s = np.loadtxt('example_data_250Hz.txt')
	Fs = 250

	# run PAPTO 
	eventsList, fooofModel = papto.find_papto_bursts(s, Fs)

	# take only beta events
	betaEvents = eventsList[(eventsList['Peak Frequency']>=15)&(eventsList['Peak Frequency']<=30)]

	# generate Figures
	fooofModel.save_report('fooof_report.pdf')


	fig, axes = plt.subplots(2, 3)
	sns.histplot(ax=axes[0,0], data=betaEvents, x='Peak Time')
	sns.histplot(ax=axes[0,1], data=betaEvents, x='Event Duration')
	sns.histplot(ax=axes[0,2], data=betaEvents, x='Frequency Span')
	sns.histplot(ax=axes[1,0], data=betaEvents, x='Peak Power')
	sns.histplot(ax=axes[1,1], data=betaEvents, x='Peak Frequency')
	sns.histplot(ax=axes[1,2], data=betaEvents, x='periods')
	plt.tight_layout(pad=1.0)
	plt.savefig('papto_beta_event_characteristics.pdf')
	plt.close()
