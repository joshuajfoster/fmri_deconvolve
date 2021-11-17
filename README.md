# fmri_deconvolve

**fmri_deconvolve** is a python module for deconvolution of fMRI responses in event-related designs.

The modules includes the following functions:

`buildDesignMatrix_betaEst` : build a design matrix given an assumed HRF to estimate beta weights for each condition.

`buildDesignmatrix_deconvolve`: build a design matrix to estimate the full HRF for each condition (i.e. not assuming an HRF).

`deconvolution_example.ipynb` is an IPython notebook that provide an example of the functions in action.

The module also includes some functions for extracting info from paradigm files generated with FreeSurfer optseq2

Note: the **example_paradigm_files** folder includes some paradigm files that specify the timing of trials in a hypothetical event-related fMRI experiment. These are used by `deconvolution_example.ipynb`. The paradigm files were generated using FreeSurfer's optseq2, which you can find [here](https://surfer.nmr.mgh.harvard.edu/fswiki/optseq2).





