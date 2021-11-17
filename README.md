# STEM flyback correction
This repository contains scripts to diagnose and correct flyback distortion from scanning transmission electron microscope (STEM) images.

## Usage
All code is written in python (tested with Python 3.7) and is configured to use DigitalMicrograph .dm3 or .dm4 files and tags. The two main scripts are the diagnose.py and correct.py that are used to measure the correction parameters and apply them, respectively. These files define several variables that are to be set by the user.