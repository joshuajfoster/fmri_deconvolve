#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 18:58:09 2021

@author: joshuafoster
"""

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import deconvolve as dcvl

# simulation parameters
TR = 1 # duration of TR
endBuffer = 16 # null period following sequence specified in par file
noiseSD = 0.1 # SD of guassian noise added to simulated time series

# HRF parameters
tau = 2 

# make HRF
t = np.arange(0,20,TR)
HRF = TR*(t/tau)**2 * np.exp(-t/tau) / 2*tau # REVIEW: still need to include the delta param...
HRF = HRF/np.max(HRF)


#%% load and concatenate paradigm files
    

file_dir = os.getcwd() + '/example_paradigm_files/'
run_numbers = list(range(1,9+1))
run_names = dcvl.create_parfile_list(run_numbers,'par')


stimTimes, cond, runNum, totalDur = dcvl.compile_paradigm_files(file_dir,run_names,endBuffer)

nRuns = len(np.unique(runNum))
nConds = len(np.unique(cond))
nTRs = int(totalDur/TR)


#%% simulate fMRI data

designMatrix, condIdx = dcvl.buildDesignMatrix_paramEst(cond,stimTimes,runNum,HRF)

# generate betas for each condition
betas = np.linspace(-1,1,nConds)
betas = np.concatenate((betas,np.zeros(nRuns))) # add zeros for run terms in design matrix

# simulate fMRI time series by multiplying deisgn matrix by random betas and adding gaissina noise
fMRI = np.matmul(designMatrix,betas) + np.random.standard_normal(nTRs)*noiseSD

# plot simulated BOLD time series
#plt.figure()
#plt.plot(range(nTRs), fMRI)
#plt.xlabel('Time (s)')


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
plt.legend(['data time-series', 'fit'])
plt.title('Full time-series')
    

# build a design matrix for deconvolution
nTimes = 20  # number of time points to deconvolve
designMatrix, condIdx = dcvl.buildDesignMatrix_deconvolve(cond, stimTimes, runNum, nTimes)

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
plt.xlim([0,nTimes-1])
plt.xlabel('Times (s)')
plt.ylabel('Response')
plt.legend(['ground truth','deconvolved response'])
