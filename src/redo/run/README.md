# Overview
This folder contains REDO implementation, and scripts used to analyze the results. Files outsides *runtime* are used to run on SWEDE; while files inside runtime are used to run REDO on STA.

# File Descriptions:
- *error_detection_pyright.py*: detecting runtime errors using only pyright
- *error_detection_pyflakes.py*: detecting runtime errors using only pyflakes
- *runtime_error_detection.py*: detecting runtime errors using LLM-based
- *whole_pipeline.py*: the whole REDO pipeline, including differential analysis and LLM-based detection.
- *analysis_external_dependencies.ipynb*: compute the average number of dependencies under the parent folder of the modified files
- *compute_error_average.ipynb*: compute the average count of different types of errors on the top-6 coding agents
- *end2end.py*: compute evaluation stats reported in Table 2
- *ensemble.py*: run the patch ensemble algorithm
- *external_dependencies.py*: identify external dependencies of different coding agents using code2flow
- *fix_patch.py*: implementation of the patch-fixing algorithm proposed in section 5.2
- *qualitative_analysis.ipynb*: visualization of qualitative results reported in section 5.2

# Steps to run
- Setup SWE-bench-lite repository [SWE-bench](https://github.com/swe-bench/experiments/tree/main)
- Setup the virtual environment following the requirement.txt
- specify one key argument:
-- *predictions_file*: the path to predictions file generated after evaluating the coding agent
- run corresponding files, e.g., 
```python 
python error_detection_pyright.py --predictions_file [PATH_TO_PREDICTIONS_FILE]
```
