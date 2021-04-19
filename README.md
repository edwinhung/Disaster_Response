# Installation
Libraries needed in the code are: nltk, lightgbm, joblib.

If you are using Anaconda, you can run 
> conda install nltk lightgbm joblib

The rest of libraries used in the code should be in Anaconda distribution of Python 3.8.

# Project Motivation
Filtering disaster messages to pull out the most relevant and important information is vital to disaster response. This project build a NLP pipeline to process text message and train a supervised learning model for 36 categories. The model can be run on a Web API to classify input message. 

# Dataset
Data used here are message and categories datasets, which contain text messages and the corresponding categories. Both datasets are from Figure Eight.

# File Descriptions
### 1. Jupyter notebooks
ETL_Pipeline_Preparation and ML_Pipeline_Preparation notebooks interactively show case steps to build pipelines and machine learning model.
### 2. Python script files
- process_data.py: load and process datasets, which is then saved in DisasterResponse database
- train_classifier.py: transform text data and feed into LightGBM model.

# How to interact with this project
Executing run.py would set up and run Web API, which is capable of showing classification result for message inputed by users. 

# Acknowledgements
Two datasets, message and cateogires, come from Figure Eight. 
