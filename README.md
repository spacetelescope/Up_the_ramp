Project for up-the-ramp fitting of IR exposures

# Iterative maximum likelihood fit of infrared detectors ramps

This package contains the code described in Gennaro and Khandrika (2020) - **ref goes here** to perform the fit of infrared detector ramps.
The code can be applied to ramps with arbitrary number of groups, averaged frames per group and skipped frames.


## Content

The main routines for simulating and fitting ramps are contained in the ramp_utils package.
 - `ramp.py` defines the `RampTimeSeq` class and `RampMeasurement` class, than can be used together to created simulated ramps
 - `fitter.py` defines the `IterativeFitter` class which is the workhorse code for fitting the ramps and obtaining an estimate of the true incident flux

The Test_notebooks folder contains a series of examples to help the user familiarize with the code and its functionalities


## Authors

 - [Mario Gennaro](https://github.com/mgennaro)
