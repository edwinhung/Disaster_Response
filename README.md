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
- run.py: set up Web API with Flask.

# How to interact with this project
In app folder, executing run.py would set up and run Web API as the following
> python run.py

Then, go to http://0.0.0.0:3001/

Web API has visualization that shows distributions of the training set. 
Users can also input text into message box and click Classify button to see
predictions of the model on 36 categories. 

Note: to run Web API in Windows, set host='127.0.0.1' in run.py file, and use 
http://127.0.0.1:3001/ instead.

If you would like to preprocess data and build the model from scratch , remove DisasterResponse.db first and execute the following lines in terminal:
> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
# Acknowledgements
Figure Eight is the source of datasets used here.
