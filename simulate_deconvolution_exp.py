#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:58:09 2021

@author: joshuafoster
"""

#%%

import numpy as np
import matplotlib.pyplot as plt
import deconvolveTools as dcvl

# simulation parameters
TR = 1 # duration of TR
startBuffer = 4 # REVIEW: not in use
endBuffer = 16 # null period after start of sequence
noiseSD = 0.1 # SD of guassian noise added to simulated time series

# HRF parameters
tau = 2 

# make HRF
t = np.arange(0,20,TR)
HRF = TR*(t/tau)**2 * np.exp(-t/tau) / 2*tau # REVIEW: still need to include the delta param...
HRF = HRF/np.max(HRF)


#%% load and concatenate paradigm files
    
file_dir = '/Users/joshuafoster/Foster/repos/FBA_CRF_WEDGES/Subject Data/S013/ParFiles/'
run_names = ['PAR-001.par','PAR-002.par','PAR-003.par','PAR-004.par','PAR-005.par','PAR-006.par','PAR-007.par','PAR-008.par','PAR-009.par','PAR-010.par']
nRuns =  len(run_names)

for r in range(nRuns):
    
    filepath = file_dir + run_names[r]

    eventTimes, condNums, eventDurs, condNames, nConds, seqDur =  dcvl.read_paradigm_file(filepath)
    
    if r == 0:
        stimTimes = eventTimes
        cond = condNums
        totalDur = seqDur + endBuffer
    else:
        stimTimes = np.concatenate((stimTimes,totalDur+eventTimes))
        cond = np.concatenate((cond,condNums))
        totalDur = totalDur + seqDur + endBuffer
        
nTRs = totalDur # convert duration to TRs



#%% simulate fMRI data

# make design matrix
designMatrix = np.zeros([nTRs,nConds])
for c in range(nConds):
    ts = np.zeros(nTRs)
    condStimTimes = stimTimes[cond == c + 1] 
    ts[condStimTimes.astype('int')] = 1
    tmp = np.convolve(ts,HRF)
    designMatrix[:,c] = tmp[0:nTRs]

# # plot design matrix
# plt.imshow(designMatrix,cmap='viridis',aspect = 0.01)

# generate betas for each condition
betas = np.linspace(-1,1,nConds)

# simulate fMRI time series by multiplying deisgn matrix by random betas and adding gaissina noise
fMRI = np.matmul(designMatrix,betas) + np.random.standard_normal(nTRs)*noiseSD

#from scipy import stats
#fMRI = stats.zscore(fMRI)


#%% perfom deconvolution

# to estimate betas for each condition, we solve (via least squares)
# the linear system given the BOLD time series (fMRI) and the design matrix
designMatrix_pinv = np.linalg.pinv(designMatrix) 
betas_est = np.dot(designMatrix_pinv,fMRI)

# calculate the predicted fMRI response given the estimated beta weights
pred = np.matmul(designMatrix,betas_est)

plt.figure()
plt.plot(range(nTRs), fMRI, label='data')
plt.plot(range(nTRs),pred, label='fit')
plt.xlabel('Time (s)')
plt.legend(['BOLD time-series', 'fit'])
#plt.xlim([0,1000])
    

# build a design matrix for deconvolution
nTimes = 20  # number of time points to deconvolve
designMatrix, condIdx = dcvl.design_matrix_deconvolve(cond, stimTimes, nTRs, nTimes)

# do deconvolution!
designMatrix_pinv = np.linalg.pinv(designMatrix) 
betas_est = np.dot(designMatrix_pinv,fMRI)

deconvolvedResp = np.zeros([nConds,nTimes])
for c in range(nConds):
   deconvolvedResp[c,:] = betas_est[condIdx == c+1]
    

import matplotlib.pylab as pl
colors = pl.cm.rainbow(np.linspace(0,1,nConds))

plt.figure()
for c in range(nConds):
    ground_truth = betas[c]*HRF
    plt.plot(range(nTimes),ground_truth,color=colors[c])
    plt.plot(range(nTimes),deconvolvedResp[c,:],color=colors[c],linestyle='--')
    
    plt.xlabel('Times (s)')
    # set x-ticks
    # - simulated response, -- deconvolved response
    # y label
