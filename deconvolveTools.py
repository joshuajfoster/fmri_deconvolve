#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
deconvolveTools: functions for performing deconvolution of fMRI data

Created on Tue Jul 20 18:42:18 2021

@author: joshuafoster
"""

#%%

# REVIEW: maybe read_paradigm_file belongs in a different module?? Not really deconvolution

def read_paradigm_file(filepath):
    """
    returns the content of a Freesurfer par file as numpy arrays. 
    Numpy should be imported as np before function this is called.
    
    Joshua Foster, 7.20.2021

    Parameters
    ----------
    filepath : full filepath including directory and filename. 
    You can provide just the filename if you're in the directory that contains the file.

    Returns
    -------
    eventTimes : numpy array of non-null events.
    cond : numpy array of condition numbers.
    eventDurs : numpy array of event durations.
    nConds : number of conditions.
    seqDur : duration of sequence in seconds.
    
    """
      
    import numpy as np # REVIEW: better to do this, or provide np as an input to the function
    
    # read in par file, grab event times and conditions
    f=open(filepath,"r")
    lines=f.readlines()
    eventTimes = []
    condNums = []
    eventDurs = []
    condNames = []
    for x in lines:
        eventTimes.append(x.split()[0]) # event times from 1st column
        condNums.append(x.split()[1]) # condition numbers from 2nd column
        eventDurs.append(x.split()[2]) # event durations from 3rd column
        condNames.append(x.split()[4])
    f.close()
    
    # convert to np arrays
    eventTimes = np.array(eventTimes).astype(np.float)
    condNums = np.array(condNums).astype(np.int)
    eventDurs = np.array(eventDurs).astype(np.float)
    condNames = np.array(condNames).astype(np.str)
    
    # calculate number of TRs in sequence
    seqDur = eventDurs.sum() # REVIEW: I'm assuming TR = 1, fix this later!!
    seqDur = seqDur.astype('int')
    
    # remove rows for null events
    eventTimes = eventTimes[condNums != 0]
    eventDurs = eventDurs[condNums != 0]
    condNames = condNames[condNums != 0]
    condNums = condNums[condNums != 0]
    nConds = np.unique(condNums).size
    
    return eventTimes, condNums, eventDurs, condNames, nConds, seqDur

#%%
   
def buildDesignMatrix_paramEst(cond,stimTimes,runNum,HRF):

    import numpy as np
    
    nConds = np.unique(cond).shape[0] # number of conditions
    nRuns = np.unique(runNum).shape[0] # number of runs
    nTRs = runNum.shape[0] # number of TRs in full time series
    nTRsPerRun = nTRs/nRuns
    
    # make design matrix # REVIEW: should make a design matrix function given an HRF in deconvolve tools
    designMatrix = np.zeros([nTRs,nConds])
    for c in range(nConds):
        ts = np.zeros(nTRs)
        condStimTimes = stimTimes[cond == c + 1] 
        ts[condStimTimes.astype('int')] = 1
        tmp = np.convolve(ts,HRF)
        designMatrix[:,c] = tmp[0:nTRs]
                  
    # run specific terms
    runTerms = np.zeros([nTRs,nRuns])
    for r in range(nRuns):
        runTerms[runNum == r + 1,r] = 1
    
    # concatenate the run terms to the design matrix
    designMatrix = np.concatenate((designMatrix,runTerms),axis=1)
    
    # create condition index
    condIdx = np.concatenate((np.array(range(nConds))+1,np.zeros(nRuns)))
        
    return designMatrix, condIdx


#%%

def buildDesignMatrix_deconvolve(cond,stimTimes,runNum,nTimes):
    
    import numpy as np # REVIEW: again, better to do this or to provide np as an input?
    
    nConds = np.unique(cond).shape[0] # number of conditions
    nRuns = np.unique(runNum).shape[0] # number of runs
    nTRs = runNum.shape[0] # number of TRs in full time series
    nTRsPerRun = nTRs/nRuns

    for c in range(nConds):
        ts = np.zeros(nTRs)
        condStimTimes = stimTimes[cond == c + 1] 
        ts[condStimTimes.astype('int')] = 1
       
        dm = np.zeros([nTRs,nTimes])
        for t in range(nTimes):
            dm[:,t] = ts
            ts = np.concatenate((np.zeros(1),ts))
            ts = ts[range(nTRs)]
            
        if c == 0:
            designMatrix = dm
            condIdx = np.ones(nTimes)
        else:
            designMatrix = np.concatenate((designMatrix,dm),axis=1)
            condIdx = np.concatenate((condIdx,(c+1)*np.ones(nTimes)))
            
    # run specific terms
    runTerms = np.zeros([nTRs,nRuns])
    for r in range(nRuns):
        runTerms[runNum == r + 1,r] = 1
    
    # concatenate the run terms to the design matrix
    designMatrix = np.concatenate((designMatrix,runTerms),axis=1)
    condIdx = np.concatenate((condIdx,np.zeros(nRuns)))

   
    return designMatrix, condIdx