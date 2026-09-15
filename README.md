# Brain-to-brain synchrony during shared reminiscing predicts connection and memory in romantic couples

This repository contains data and analysis code for our manuscript, *Brain-to-brain synchrony during shared reminiscing predicts connection and memory in romantic couples*. If you have any questions or encounter any bugs/broken links, please email me at jadynpark@uchicago.edu.

## 1. fNIRS data

The .mat files contain preprocessed, normalized oxygenated hemoglobin time courses.  

The dimensions are: nSamples x nChannels x nSubjects = 6104 x 20 x 80  

(note that one couple was omitted due to device failure)  

## 2. Behavioral data

The .csv file contains self-report measures of post-origin story shared reality, closeness, and relationship satisfaction. It also includes measures of origin story centrality, how often they revisit their origin story, and relationship duration.

## 3. Analysis scripts

a_calc_ISC.m: Script for performing intersubject correlation analysis  
b_create_nulldist_ISC.m: Script for creating null distribution for ISC  
c_convo_recall_similarity.py: Script for performing conversation-recall semantic similarity analysis  


