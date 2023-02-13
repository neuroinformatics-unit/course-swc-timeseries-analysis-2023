# Introduction to timeseries analysis in Python (2023)

[Link to the slides (online version)](https://docs.google.com/presentation/d/1auKX6nvKOJgY_fg_Su7TFEesEQs9YK7aesjoYnJAGy8/edit?usp=sharing)

This repo contains materials for the course Timeseries analysis in Python (2023) at the 
Sainsbury Wellcome Centre.

It contains material for attending the course, and also all materials for the production
of course figures / animations for those interested.

### Materials related to attending the course:

- environment_check.py : Run this python script to ensure your environment is setup correctly. 
			 The script contains a list of dependencies that need to be installed.
			 A plot should be displayed if everything is working. 

- sub-001_drug-cch_rawdata.csv : the raw data we will be analysing in the course




### Materials related to development of the course:

walkthrough/

This folder contains a complete walkthrough of the course (i.e. working copy of all 
code that will be written during the course)

figures/ and animations/

code for producing the figures and animations from the course

data_generation/ 

the data used in the course was generated from real data. These
scritps take the raw data (two recordings, one with and one without drug),
concatenates them and adds high frequency noise.
