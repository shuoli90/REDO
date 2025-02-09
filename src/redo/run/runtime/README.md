# Overview
This folder is for running REDO on STA and compute performance statistics. The readme file contains instructions for setting up the dataset; and running experiments.

# First step:
Set up dataset: follow instructions mentioned in this repo [STA]{https://github.com/google-research/runtime-error-prediction}

# Generating 
- The error and safe splits are saved under *collected/runtime*; 
- modify the file path in *google_bench.py* and *google_bench_nocontext.py* to point to the collected files
- There are two key arugument:
**split*: which split of sampled data to run (1,2,3,4)
** *seed*: which random seed was used when sampling the safe instances (only valid for safe instance, available seeds are 21, 42, 84)
- Generated results will be saved in the current folder

# Static analysis tools
- Static analysis tools (pyright and pyflakes) are implemented separately in *google_bench_pyright.py* and *google_bench_pyflakes.py*
- Both files have three key parameters:
** *split*: same as above
** *seed*: same as above
** *no_error*: if set, instances without runtime errors will be analyzed; otherwise, instances with runtime errors will be analyzed.

# Evaluation
- *analysis.py* computes quantitative results basing on the collected results
- It has two key arguments:
** *-e*: only instances with runtime errors are considered (used for E.F1)
** *-n*: Without context scenario is considered

# Plotting Figure 1
- Update the stats in the notebook
- run the cell and the figure will be saved under this folder