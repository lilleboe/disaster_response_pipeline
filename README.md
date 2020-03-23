# Disaster Response Pipeline

## Summary
The project contains code for a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.  The repository contains a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

## Running Instructions
### ***Run ETL Process***
1. In the data directory run the following command:
`python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`

### ***Run ML Pipeline***
1. In the models directory run the following command:
`python train_classifier.py ../data/DisasterResponse.db classifier.pkl`

### ***Run Web Application***
1. Run the following command in the app directory:
    `python run.py`
2. Go to http://0.0.0.0:3001/


## Files
* **ETL Pipeline Preparation.ipynb**: 
* **ML Pipeline Preparation.ipynb**: 
* **process_data.py**: 
* **train_classifier.py**: 
* **data**: 
* **app**: 

## Web App UI

***App Interface***

***App Results Page***

## Note
The data sourced for this program was provided by [Udacity](https://www.udacity.com) and [Figure Eight](https://www.figure-eight.com/).  The dataset is not very evenly distributed between postive and negative examples.  This can skew the results of the classifier and provide misleading accuracy.
