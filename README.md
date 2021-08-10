# new_aachen_data
Working with the data from Baumhofer 2014, to reproduce the results in their paper
(https://doi.org/10.1016/j.jpowsour.2020.228863)<br>

## Notebook Outline
<b>'01. baumhofer_pre_processing.ipynb'</b><br>
<b>Note this notebook IS NOT REQUIRED if you only want to use the first 100 cycles of data</b><br>
If you are only interested in the first 100 cycles, this is the baumhofer_first_100_cycles.pkl file located 
[HERE](https://www.dropbox.com/sh/jdbib6xx2p31vyr/AAAxvFHDNhnp6mtLuXm4WWUja?dl=0)<br>

Notebook summary:
- Process the time series data (V, I, time_elapsed, T) from the raw CSV files, for <b>all cycles</b>.<br>
- Clean the data - remove cycles with erroneous values


<b>'02. CLEAN_baumhofer_first_100_interpolated.ipynb'</b>
- Load the time series data (V, I, time_elapsed, T) from pkl file
- Do the pre-processing to get train, validation and test sets, with the data split <b>at the cell level</b>

<b>'03. Rough_model_experimentation.ipynb'</b>
- Initial evaluation of 1D CNN model using the Baumhofer data. This does not yet implement any hyperparameter tuning or k-fold cross validation.

## Module Outline

baumhofer_utils.py<br>
A collection of functions that are used for pre-processing.

knee_finder.py<br>
Contains the KneeFinder class, used to extract relevant information from capacity fade / IR rise curves.
